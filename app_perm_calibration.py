"""
Permeability Auto-Calibration Tool
===================================
Streamlit app terpisah untuk optimasi parameter permeabilitas empiris
menggunakan data core sebagai ground truth. Scope: per well.

Metode perhitungan PERM (dari LLS):
1. Coates-Dumanoir   : PERM = ((C / W^4) * (PHIE^W / SW^W))^2     → param: C, W
2. Coates FFI        : PERM = (C * PHIE^2 * (1-SW)/SW)^2           → param: C
3. Wyllie-Rose       : PERM = (C * PHIE^D / SW^E)^2                → param: C, D, E

Metode pencarian parameter (optimizer):
1. Differential Evolution  — global stochastic optimizer
2. Dual Annealing          — simulated annealing + local refinement
3. Basin-Hopping           — random perturbation + local optimization
4. Multi-Start L-BFGS-B    — banyak titik awal acak + gradient-based
5. Grid Search             — exhaustive grid search di seluruh bounds

Jalankan: streamlit run app_perm_calibration.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import lasio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import (
    differential_evolution, dual_annealing, basinhopping,
    minimize, brute,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io
import warnings
import re

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Perm Auto-Calibration",
    page_icon="🔧",
    layout="wide",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

NULL_VAL = -999.25


def _normalize_well_name(name: str) -> str:
    """Normalisasi nama sumur → PREFIX-NNN[SUFFIX]."""
    if not isinstance(name, str):
        return str(name)
    m = re.match(r'^([A-Za-z]+[-_])(\d+)([A-Za-z]*)$', name.strip())
    if m:
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        return f"{prefix.upper()}{digits.zfill(3)}{suffix.upper()}"
    return name.upper()


def read_zone_csv(file_obj) -> pd.DataFrame:
    raw = file_obj.read() if hasattr(file_obj, 'read') else open(file_obj, 'rb').read()
    text = raw.decode('utf-8', errors='replace')
    sep = ';' if text.count(';') > text.count(',') / 2 else ','
    if sep == ';':
        lines = []
        for line in text.splitlines():
            parts = line.split(';')
            fixed = []
            for p in parts:
                p = p.strip()
                if ',' in p:
                    try:
                        float(p.replace(',', '.'))
                        p = p.replace(',', '.')
                    except ValueError:
                        pass
                fixed.append(p)
            lines.append(sep.join(fixed))
        text = '\n'.join(lines)

    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    well_col = next((c for c in df.columns if 'WELL' in c or 'UWI' in c), None)
    depth_col = next((c for c in df.columns if c in (
        'MD', 'DEPTH', 'DEPTH_MD', 'TVD') or 'DEPTH' in c), None)
    zone_col = next((c for c in df.columns if c in (
        'SURFACE', 'ZONE', 'FORMATION', 'FORM', 'UNIT', 'LAYER', 'NAME', 'MARKER')), None)

    if not all([well_col, depth_col, zone_col]):
        cols = df.columns.tolist()
        if len(cols) >= 3:
            well_col, depth_col, zone_col = cols[0], cols[1], cols[2]
        else:
            return pd.DataFrame()

    df = df[[well_col, depth_col, zone_col]].copy()
    df.columns = ['WELL_NAME', 'MD', 'ZONE_NAME']
    df['WELL_NAME'] = df['WELL_NAME'].str.strip().apply(_normalize_well_name)
    df['MD'] = pd.to_numeric(df['MD'].str.strip().str.replace(
        ',', '.', regex=False), errors='coerce')
    df['ZONE_NAME'] = df['ZONE_NAME'].str.strip()

    records = []
    for well, grp in df.groupby('WELL_NAME'):
        grp = grp.sort_values('MD').reset_index(drop=True)
        valid = grp[grp['ZONE_NAME'].notna() & (
            grp['ZONE_NAME'] != '')].reset_index(drop=True)
        for i, row in valid.iterrows():
            top = row['MD']
            nxt = grp[grp['MD'] > top]['MD']
            bot = float(nxt.iloc[0]) if len(nxt) > 0 else 99999.0
            records.append({'WELL_NAME': well, 'DEPTH_TOP': float(
                top), 'DEPTH_BOT': bot, 'ZONE': row['ZONE_NAME']})
    return pd.DataFrame(records)


def merge_zone(df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'ZONE' not in df.columns:
        df['ZONE'] = 'UNKNOWN'
    if zone_df is None or len(zone_df) == 0:
        return df

    # Identify well name
    wname = None
    if 'WELL_NAME' in df.columns:
        wname = df['WELL_NAME'].iloc[0]
    elif 'WELL' in df.columns:
        wname = df['WELL'].iloc[0]

    if not wname:
        return df

    zw = zone_df[zone_df['WELL_NAME'] == _normalize_well_name(wname)]
    for _, r in zw.iterrows():
        m = (df['DEPTH'] >= r['DEPTH_TOP']) & (df['DEPTH'] < r['DEPTH_BOT'])
        df.loc[m, 'ZONE'] = r['ZONE']
    return df


def safe_log10(x):
    return np.log10(np.clip(x, 1e-6, None))


def get_plot_theme():
    """
    Ambil warna dari theme Streamlit agar plot mengikuti light/dark mode user.
    """
    base = st.get_option("theme.base") or "light"
    bg = st.get_option("theme.backgroundColor")
    sec_bg = st.get_option("theme.secondaryBackgroundColor")
    text = st.get_option("theme.textColor")

    if not bg:
        bg = "#0E1117" if base == "dark" else "#FFFFFF"
    if not sec_bg:
        sec_bg = "#262730" if base == "dark" else "#F0F2F6"
    if not text:
        text = "#FAFAFA" if base == "dark" else "#31333F"

    return {
        "base": base,
        "template": "plotly_dark" if base == "dark" else "plotly_white",
        "paper_bg": bg,
        "plot_bg": sec_bg,
        "font_color": text,
        "grid_color": "rgba(255,255,255,0.20)" if base == "dark" else "rgba(0,0,0,0.20)",
        "axis_color": "rgba(255,255,255,0.60)" if base == "dark" else "rgba(0,0,0,0.60)",
        "anno_bg": "rgba(15,23,42,0.80)" if base == "dark" else "rgba(255,255,255,0.85)",
        "anno_border": "rgba(255,255,255,0.35)" if base == "dark" else "rgba(0,0,0,0.35)",
    }

# def get_shared_perm_axis_settings(results, min_exp=-2, max_exp=4):
#     """
#     Ambil range log bersama untuk semua crossplot dalam satu well.
#     Default minimal 10^-2 sampai 10^4 agar tampil mirip contoh.
#     Plotly log-axis range memakai exponent, bukan nilai asli.
#     """
#     vals = []

#     for mres in results.values():
#         for key in ['perm_core_used', 'perm_pred']:
#             arr = np.asarray(mres.get(key, []), dtype=float)
#             arr = arr[np.isfinite(arr) & (arr > 0)]
#             if arr.size > 0:
#                 vals.append(arr)

#     if not vals:
#         lo_exp, hi_exp = min_exp, max_exp
#     else:
#         all_vals = np.concatenate(vals)
#         lo_exp = min(min_exp, int(np.floor(np.log10(all_vals.min()))))
#         hi_exp = max(max_exp, int(np.ceil(np.log10(all_vals.max()))))

#     tickvals = [10.0 ** i for i in range(lo_exp, hi_exp + 1)]
#     ticktext = [f"{v:g}" for v in tickvals]

#     return {
#         'range': [lo_exp, hi_exp],   # penting: exponent untuk plotly log axis
#         'tickvals': tickvals,
#         'ticktext': ticktext,
#     }


def get_shared_perm_axis_settings(min_val=0.01, max_val=1000.0):
    """
    Fixed log axis range untuk crossplot:
    0.01 sampai 1000 mD.
    Plotly log-axis range memakai exponent.
    """
    lo_exp = np.log10(min_val)   # -2
    hi_exp = np.log10(max_val)  # 3

    tickvals = [0.01, 0.1, 1, 10, 100, 1000]
    ticktext = ["0.01", "0.1", "1", "10", "100", "1000"]

    return {
        "range": [lo_exp, hi_exp],
        "tickvals": tickvals,
        "ticktext": ticktext,
    }


def compute_crossplot_log_stats(core, pred, axis_cfg=None):
    """
    Hitung statistik crossplot di log-space:
    - CC = Pearson correlation coefficient dari log10(core) vs log10(pred)
    - regression line log-log
    """
    core = np.asarray(core, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mask = (core > 0) & (pred > 0) & np.isfinite(core) & np.isfinite(pred)
    core = core[mask]
    pred = pred[mask]

    if core.size < 3:
        return None

    xlog = safe_log10(core)
    ylog = safe_log10(pred)

    cc = float(np.corrcoef(xlog, ylog)[0, 1])
    slope, intercept = np.polyfit(xlog, ylog, 1)

    if axis_cfg is None:
        all_vals = np.concatenate([core, pred])
        lo_exp = min(-2, int(np.floor(np.log10(all_vals.min()))))
        hi_exp = max(4, int(np.ceil(np.log10(all_vals.max()))))
    else:
        lo_exp, hi_exp = axis_cfg['range']

    x_line = np.logspace(lo_exp, hi_exp, 200)
    y_reg = 10 ** (intercept + slope * np.log10(x_line))

    return {
        'core': core,
        'pred': pred,
        'cc': cc,
        'slope': float(slope),
        'intercept': float(intercept),
        'x_line': x_line,
        'y_reg': y_reg,
    }


# ── Permeability equations (dari LLS) ────────────────────────────────────────

def perm_coates_dumanoir(phie, sw, C, W):
    """PERM = ((C / W^4) * (PHIE^W / SW^W))^2"""
    inner = (C / W**4) * (phie**W / np.clip(sw, 1e-6, None)**W)
    return inner**2


def perm_coates_ffi(phie, sw, C):
    """PERM = (C * PHIE^2 * (1 - SW) / SW)^2"""
    sw_safe = np.clip(sw, 1e-6, None)
    inner = C * phie**2 * (1 - sw_safe) / sw_safe
    return inner**2


def perm_wyllie_rose(phie, sw, C, D, E):
    """PERM = (C * PHIE^D / SW^E)^2"""
    sw_safe = np.clip(sw, 1e-6, None)
    inner = C * phie**D / sw_safe**E
    return inner**2


# ── Objective functions (log-space MSE) ──────────────────────────────────────

def _obj_cd(params, phie, sw, pc_log):
    C, W = params
    pred = perm_coates_dumanoir(phie, sw, C, W)
    return np.mean((safe_log10(pred) - pc_log)**2)


def _obj_ffi(params, phie, sw, pc_log):
    pred = perm_coates_ffi(phie, sw, params[0])
    return np.mean((safe_log10(pred) - pc_log)**2)


def _obj_wr(params, phie, sw, pc_log):
    C, D, E = params
    pred = perm_wyllie_rose(phie, sw, C, D, E)
    return np.mean((safe_log10(pred) - pc_log)**2)


# ── Method registry (LLS methods only) ──────────────────────────────────────

METHOD_INFO = {
    'Coates-Dumanoir': {
        'params':   ['C', 'W'],
        'defaults': {'C': 300.0, 'W': 2.0},
        'bounds':   [(250, 400), (1.0, 5.0)],
        'obj':      _obj_cd,
        'compute': lambda ph, sw, p: perm_coates_dumanoir(ph, sw, p['C'], p['W']),
        'desc':     'k = ((C/W⁴)·(φ^W/Sw^W))²',
        'lls_output': 'PERM_COATES_DUM',
    },
    'Coates FFI': {
        'params':   ['C'],
        'defaults': {'C': 70.0},
        'bounds':   [(50, 100)],
        'obj':      _obj_ffi,
        'compute': lambda ph, sw, p: perm_coates_ffi(ph, sw, p['C']),
        'desc':     'k = (C·φ²·(1-Sw)/Sw)²',
        'lls_output': 'PERM_COATES_FFI',
    },

    # ── Wyllie-Rose family dipisah sesuai LLS ──────────────────────────
    'WR Morris-Biggs Gas': {
        'params':   ['C', 'D', 'E'],
        'defaults': {'C': 79.0, 'D': 3.0, 'E': 1.0},
        'bounds':   [(50, 100), (1.0, 5.0), (0.5, 3.0)],
        'obj':      _obj_wr,
        'compute': lambda ph, sw, p: perm_wyllie_rose(ph, sw, p['C'], p['D'], p['E']),
        'desc':     'k = (C·φ^D / Sw^E)² | Morris-Biggs Gas',
        'lls_output': 'PERM_MB_GAS',
    },
    'WR Morris-Biggs Oil': {
        'params':   ['C', 'D', 'E'],
        'defaults': {'C': 250.0, 'D': 3.0, 'E': 1.0},
        'bounds':   [(100, 300), (1.0, 5.0), (0.5, 3.0)],
        'obj':      _obj_wr,
        'compute': lambda ph, sw, p: perm_wyllie_rose(ph, sw, p['C'], p['D'], p['E']),
        'desc':     'k = (C·φ^D / Sw^E)² | Morris-Biggs Oil',
        'lls_output': 'PERM_MB_OIL',
    },
    'WR Timur': {
        'params':   ['C', 'D', 'E'],
        'defaults': {'C': 92.63, 'D': 2.2, 'E': 1.0},
        'bounds':   [(70, 200), (1.0, 5.0), (0.5, 3.0)],
        'obj':      _obj_wr,
        'compute': lambda ph, sw, p: perm_wyllie_rose(ph, sw, p['C'], p['D'], p['E']),
        'desc':     'k = (C·φ^D / Sw^E)² | Timur',
        'lls_output': 'PERM_TIM',
    },
    'WR Tixier': {
        'params':   ['C', 'D', 'E'],
        'defaults': {'C': 250.0, 'D': 3.0, 'E': 1.0},
        'bounds':   [(100, 300), (1.0, 5.0), (0.5, 3.0)],
        'obj':      _obj_wr,
        'compute': lambda ph, sw, p: perm_wyllie_rose(ph, sw, p['C'], p['D'], p['E']),
        'desc':     'k = (C·φ^D / Sw^E)² | Tixier',
        'lls_output': 'PERM_TIX',
    },
}

# Wyllie-Rose default variants (untuk perbandingan)
WR_DEFAULT_VARIANTS = {
    'WR Morris-Biggs Gas': {'C': 79.0,  'D': 3.0,  'E': 1.0},
    'WR Morris-Biggs Oil': {'C': 250.0, 'D': 3.0,  'E': 1.0},
    'WR Timur':            {'C': 100.0, 'D': 2.25, 'E': 1.0},
}


def compute_perm(method, phie, sw, params):
    info = METHOD_INFO.get(method)
    if info is None:
        raise KeyError(f"Unknown method: {method}")
    return info['compute'](phie, sw, params)


# ── Optimizer registry ───────────────────────────────────────────────────────

OPTIMIZER_INFO = {
    'Differential Evolution': {
        'desc': 'Global stochastic optimizer. Populasi kandidat di-evolve secara iteratif. Robust untuk multi-modal.',
        'short': 'DE',
    },
    'Dual Annealing': {
        'desc': 'Simulated annealing + local search (L-BFGS-B). Baik untuk landscape yang rugged.',
        'short': 'DA',
    },
    'Basin-Hopping': {
        'desc': 'Random perturbation + local minimization. Efektif untuk banyak local minima.',
        'short': 'BH',
    },
    'Multi-Start L-BFGS-B': {
        'desc': 'Gradient-based optimizer dari banyak titik awal acak. Cepat, baik jika landscape relatif smooth.',
        'short': 'MS',
    },
    'Grid Search': {
        'desc': 'Exhaustive grid search. Paling lambat tapi paling thorough. Resolusi grid: 20 titik per parameter.',
        'short': 'GS',
    },
}


def _run_differential_evolution(obj, bounds, args):
    res = differential_evolution(
        obj, bounds, args=args,
        seed=42, maxiter=500, tol=1e-8, polish=True,
    )
    return res.x, res.fun


def _run_dual_annealing(obj, bounds, args):
    res = dual_annealing(
        obj, bounds, args=args,
        seed=42, maxiter=500,
    )
    return res.x, res.fun


def _run_basin_hopping(obj, bounds, args):
    # Start from midpoint of bounds
    x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'args': args,
    }
    res = basinhopping(
        obj, x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=200, seed=42,
    )
    return res.x, res.fun


def _run_multistart_lbfgsb(obj, bounds, args, n_starts=50):
    rng = np.random.RandomState(42)
    best_x, best_fun = None, np.inf
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        try:
            res = minimize(obj, x0, method='L-BFGS-B',
                           bounds=bounds, args=args)
            if res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x
        except Exception:
            continue
    if best_x is None:
        raise RuntimeError("All starts failed")
    return best_x, best_fun


def _run_grid_search(obj, bounds, args, grid_points=20):
    # Build grid ranges for brute
    ranges = [slice(b[0], b[1], complex(0, grid_points)) for b in bounds]
    result = brute(obj, ranges, args=args, finish=minimize, full_output=True)
    return result[0], result[1]


OPTIMIZER_FUNCS = {
    'Differential Evolution': _run_differential_evolution,
    'Dual Annealing': _run_dual_annealing,
    'Basin-Hopping': _run_basin_hopping,
    'Multi-Start L-BFGS-B': _run_multistart_lbfgsb,
    'Grid Search': _run_grid_search,
}


# ── Main optimisation runner ────────────────────────────────────────────────

def optimise_method(method, phie, sw, perm_core, optimizers):
    """
    Run multiple optimizers for a given perm method.
    Returns dict: {optimizer_name: result_dict} and best_optimizer_name.
    """
    mask = (perm_core > 0) & np.isfinite(
        phie) & np.isfinite(sw) & (phie > 0) & (sw > 0)
    if mask.sum() < 3:
        return None, None

    ph = phie[mask].values if hasattr(phie, 'values') else phie[mask]
    sw_arr = sw[mask].values if hasattr(sw, 'values') else sw[mask]
    pc = perm_core[mask].values if hasattr(
        perm_core, 'values') else perm_core[mask]
    pc_log = safe_log10(pc)

    info = METHOD_INFO[method]
    obj_func = info['obj']
    bounds = info['bounds']
    param_names = info['params']

    results_by_opt = {}
    best_r2 = -np.inf
    best_opt = None

    for opt_name in optimizers:
        opt_func = OPTIMIZER_FUNCS[opt_name]
        try:
            opt_vals, opt_fun = opt_func(
                obj_func, bounds, (ph, sw_arr, pc_log))
        except Exception:
            continue

        # Clip to bounds
        opt_vals = np.clip(opt_vals,
                           [b[0] for b in bounds],
                           [b[1] for b in bounds])

        par = {k: v for k, v in zip(param_names, opt_vals)}
        pred = info['compute'](ph, sw_arr, par)
        pred_log = safe_log10(pred)

        r2 = r2_score(pc_log, pred_log)
        rmse = np.sqrt(mean_squared_error(pc_log, pred_log))
        mae = mean_absolute_error(pc_log, pred_log)

        cp_stats = compute_crossplot_log_stats(pc, pred)

        results_by_opt[opt_name] = {
            'params': par,
            'r2': r2,
            'rmse_log': rmse,
            'mae_log': mae,
            'cc_log': np.nan if cp_stats is None else cp_stats['cc'],
            'reg_slope_log': np.nan if cp_stats is None else cp_stats['slope'],
            'reg_intercept_log': np.nan if cp_stats is None else cp_stats['intercept'],
            'obj_value': float(opt_fun),
            'perm_pred': pred,
            'perm_core_used': pc,
            'phie_used': ph,
            'swirr_used': sw_arr,
        }

        if r2 > best_r2:
            best_r2 = r2
            best_opt = opt_name

    return results_by_opt, best_opt


def eval_default(method, phie_vals, sw_vals, perm_core_vals, defaults, label):
    mask = (perm_core_vals > 0) & np.isfinite(phie_vals) & np.isfinite(sw_vals)
    mask = mask & (phie_vals > 0) & (sw_vals > 0)
    if mask.sum() < 3:
        return None

    ph = phie_vals[mask]
    sw_a = sw_vals[mask]
    pc = perm_core_vals[mask]

    pred = compute_perm(
        method if method in METHOD_INFO else 'Wyllie-Rose',
        ph, sw_a, defaults
    )

    pred_log = safe_log10(pred)
    pc_log = safe_log10(pc)
    valid = np.isfinite(pred_log) & np.isfinite(pc_log)
    if valid.sum() < 3:
        return None

    pc_valid = pc[valid]
    pred_valid = pred[valid]

    cp_stats = compute_crossplot_log_stats(pc_valid, pred_valid)

    return {
        'params': defaults,
        'r2': r2_score(pc_log[valid], pred_log[valid]),
        'rmse_log': np.sqrt(mean_squared_error(pc_log[valid], pred_log[valid])),
        'mae_log': mean_absolute_error(pc_log[valid], pred_log[valid]),
        'cc_log': np.nan if cp_stats is None else cp_stats['cc'],
        'reg_slope_log': np.nan if cp_stats is None else cp_stats['slope'],
        'reg_intercept_log': np.nan if cp_stats is None else cp_stats['intercept'],
        'perm_pred': pred_valid,
        'perm_core_used': pc_valid,
        'phie_used': ph[valid],
        'swirr_used': sw_a[valid],
    }

# ── LAS loading ─────────────────────────────────────────────────────────────


def load_las(uploaded_file):
    raw = uploaded_file.read()
    try:
        las = lasio.read(io.BytesIO(raw))
    except Exception:
        las = lasio.read(io.StringIO(raw.decode('latin-1')))
    df = las.df().reset_index()
    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: 'DEPTH'})
    df.columns = [c.upper() for c in df.columns]
    df = df.replace(NULL_VAL, np.nan)
    well_name = ''
    try:
        well_name = las.well['WELL'].value
    except Exception:
        pass
    if not well_name:
        well_name = uploaded_file.name.replace('.las', '').replace('.LAS', '')
    well_name = _normalize_well_name(well_name)
    df['WELL_NAME'] = well_name
    return well_name, df


def load_core_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df[df['WELL'].apply(lambda x: isinstance(
        x, str) and len(x.strip()) > 0)]
    df['WELL'] = df['WELL'].apply(_normalize_well_name)
    df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce')
    df['PERM_CORE'] = pd.to_numeric(df['PERM_CORE'], errors='coerce')
    if 'PORE_CORE' in df.columns:
        df['PORE_CORE'] = pd.to_numeric(df['PORE_CORE'], errors='coerce')
    df = df.replace(NULL_VAL, np.nan)
    df = df.dropna(subset=['WELL', 'DEPTH', 'PERM_CORE'])
    df = df[df['PERM_CORE'] > 0]
    return df


def merge_core_to_log(log_df, core_df, depth_tol=0.5):
    merged_rows = []
    log_depths = log_df['DEPTH'].values
    for _, row in core_df.iterrows():
        cd = row['DEPTH']
        idx = np.argmin(np.abs(log_depths - cd))
        if np.abs(log_depths[idx] - cd) <= depth_tol:
            log_row = log_df.iloc[idx].to_dict()
            log_row['PERM_CORE'] = row['PERM_CORE']
            log_row['CORE_DEPTH'] = cd
            if 'PORE_CORE' in row.index:
                log_row['PORE_CORE'] = row['PORE_CORE']
            merged_rows.append(log_row)
    if not merged_rows:
        return pd.DataFrame()
    return pd.DataFrame(merged_rows)


# ── Plot functions ───────────────────────────────────────────────────────────

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']


def plot_log_perm(well_name, log_df, core_df, results, phie_col, sw_col):
    n_methods = len(results)
    fig = make_subplots(
        rows=1, cols=3 + n_methods,
        shared_yaxes=True,
        column_widths=[1] * (3 + n_methods),
        horizontal_spacing=0.02,
    )
    depths = log_df['DEPTH']

    fig.add_trace(go.Scatter(
        x=log_df[phie_col], y=depths, name='PHIE',
        line=dict(color='green', width=1),
    ), row=1, col=1)
    fig.update_xaxes(title_text='PHIE (v/v)', row=1, col=1, range=[0, 0.4])

    fig.add_trace(go.Scatter(
        x=log_df[sw_col], y=depths, name='SW',
        line=dict(color='blue', width=1),
    ), row=1, col=2)
    fig.update_xaxes(title_text='SW (v/v)', row=1, col=2, range=[0, 1])

    if core_df is not None and len(core_df) > 0:
        fig.add_trace(go.Scatter(
            x=core_df['PERM_CORE'], y=core_df['CORE_DEPTH'],
            mode='markers', name='PERM_CORE',
            marker=dict(color='red', size=6, symbol='circle'),
        ), row=1, col=3)
    fig.update_xaxes(title_text='PERM Core', type='log', row=1, col=3)

    for i, (mname, mres) in enumerate(results.items()):
        col = 4 + i
        phie_full = log_df[phie_col].values
        sw_full = log_df[sw_col].values
        perm_full = compute_perm(mname, phie_full, sw_full, mres['params'])
        perm_full = np.where(np.isfinite(perm_full) & (
            perm_full > 0), perm_full, np.nan)

        r2_val = mres['r2']
        fig.add_trace(go.Scatter(
            x=perm_full, y=depths,
            name=f'{mname} (R²={r2_val:.2f})',
            line=dict(color=COLORS[i % len(COLORS)], width=1),
        ), row=1, col=col)

        if core_df is not None and len(core_df) > 0:
            fig.add_trace(go.Scatter(
                x=core_df['PERM_CORE'], y=core_df['CORE_DEPTH'],
                mode='markers', name='Core',
                marker=dict(color='red', size=5, symbol='circle'),
                showlegend=False,
            ), row=1, col=col)

        fig.update_xaxes(title_text=f'{mname[:18]}<br>R²={r2_val:.2f}',
                         type='log', row=1, col=col)

    fig.update_yaxes(autorange='reversed', title_text='DEPTH', row=1, col=1)
    fig.update_layout(
        height=800,
        title=f"Permeability Log — {well_name}",
        showlegend=True,
        legend=dict(orientation='h', y=-0.05),
    )
    return fig


def plot_crossplot_grid(well_name, results):
    n = len(results)
    if n == 0:
        return None

    theme = get_plot_theme()
    axis_cfg = get_shared_perm_axis_settings()

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    method_names = list(results.keys())

    subplot_titles = []
    for m in method_names:
        cc_val = results[m].get('cc_log', np.nan)
        r2_val = results[m].get('r2', np.nan)
        subplot_titles.append(f'{m}<br>CC={cc_val:.4f} | R²={r2_val:.4f}')

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    for idx, mname in enumerate(method_names):
        mres = results[mname]
        r = idx // ncols + 1
        c = idx % ncols + 1

        stats = compute_crossplot_log_stats(
            mres['perm_core_used'],
            mres['perm_pred'],
            axis_cfg=axis_cfg
        )
        if stats is None:
            continue

        # scatter points
        fig.add_trace(
            go.Scatter(
                x=stats['core'],
                y=stats['pred'],
                mode='markers',
                marker=dict(
                    color='#4C78FF',
                    size=7,
                    opacity=0.78,
                    line=dict(width=0)
                ),
                name=mname,
                showlegend=False,
            ),
            row=r, col=c
        )

        # 1:1 line
        fig.add_trace(
            go.Scatter(
                x=stats['x_line'],
                y=stats['x_line'],
                mode='lines',
                line=dict(color=theme["font_color"], width=1.2),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=r, col=c
        )

        # regression line
        fig.add_trace(
            go.Scatter(
                x=stats['x_line'],
                y=stats['y_reg'],
                mode='lines',
                line=dict(color='#FF5A5A', width=1.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=r, col=c
        )

        fig.update_xaxes(
            type='log',
            title_text='PERM Core (mD)',
            range=axis_cfg['range'],
            tickmode='array',
            tickvals=axis_cfg['tickvals'],
            ticktext=axis_cfg['ticktext'],
            showgrid=True,
            gridcolor=theme["grid_color"],
            gridwidth=0.7,
            showline=True,
            linecolor=theme["axis_color"],
            mirror=True,
            zeroline=False,
            row=r, col=c
        )

        fig.update_yaxes(
            type='log',
            title_text='PERM Pred (mD)',
            range=axis_cfg['range'],
            tickmode='array',
            tickvals=axis_cfg['tickvals'],
            ticktext=axis_cfg['ticktext'],
            showgrid=True,
            gridcolor=theme["grid_color"],
            gridwidth=0.7,
            showline=True,
            linecolor=theme["axis_color"],
            mirror=True,
            zeroline=False,
            row=r, col=c
        )

        axis_no = idx + 1
        xref = 'x domain' if axis_no == 1 else f'x{axis_no} domain'
        yref = 'y domain' if axis_no == 1 else f'y{axis_no} domain'

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref=xref,
            yref=yref,
            text=(
                f"CC(log) = {stats['cc']:.6f}<br>"
                f"y = 10^({stats['intercept']:.5f} + {stats['slope']:.5f}·log10(x))"
            ),
            showarrow=False,
            align='left',
            font=dict(size=10, color=theme["font_color"]),
            bgcolor=theme["anno_bg"],
            bordercolor=theme["anno_border"],
            borderwidth=0.7
        )

    fig.update_layout(
        height=max(420, 420 * nrows),
        title=f"Crossplot per Method — {well_name}",
        showlegend=False,
        template=theme["template"],
        paper_bgcolor=theme["paper_bg"],
        plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["font_color"]),
    )
    return fig


def plot_crossplot_combined(well_name, results):
    theme = get_plot_theme()
    axis_cfg = get_shared_perm_axis_settings()

    fig = go.Figure()
    all_core = []
    all_pred = []

    for i, (mname, mres) in enumerate(results.items()):
        stats = compute_crossplot_log_stats(
            mres['perm_core_used'],
            mres['perm_pred'],
            axis_cfg=axis_cfg
        )
        if stats is None:
            continue

        all_core.append(stats['core'])
        all_pred.append(stats['pred'])

        fig.add_trace(
            go.Scatter(
                x=stats['core'],
                y=stats['pred'],
                mode='markers',
                name=f'{mname} | CC={stats["cc"]:.3f}',
                marker=dict(
                    color=COLORS[i % len(COLORS)],
                    size=7,
                    opacity=0.72
                ),
            )
        )

    if all_core:
        all_core = np.concatenate(all_core)
        all_pred = np.concatenate(all_pred)
        overall = compute_crossplot_log_stats(
            all_core, all_pred, axis_cfg=axis_cfg)

        fig.add_trace(
            go.Scatter(
                x=overall['x_line'],
                y=overall['x_line'],
                mode='lines',
                name='1:1',
                line=dict(color=theme["font_color"], width=1.2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=overall['x_line'],
                y=overall['y_reg'],
                mode='lines',
                name=f'Overall regression | CC={overall["cc"]:.3f}',
                line=dict(color='#FF5A5A', width=1.6, dash='dot'),
            )
        )

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=(
                f"Overall CC(log) = {overall['cc']:.6f}<br>"
                f"y = 10^({overall['intercept']:.5f} + {overall['slope']:.5f}·log10(x))"
            ),
            showarrow=False,
            align='left',
            font=dict(size=11, color=theme["font_color"]),
            bgcolor=theme["anno_bg"],
            bordercolor=theme["anno_border"],
            borderwidth=0.7
        )

    fig.update_xaxes(
        type='log',
        title_text='PERM Core (mD)',
        range=axis_cfg['range'],
        tickmode='array',
        tickvals=axis_cfg['tickvals'],
        ticktext=axis_cfg['ticktext'],
        showgrid=True,
        gridcolor=theme["grid_color"],
        gridwidth=0.7,
        showline=True,
        linecolor=theme["axis_color"],
        mirror=True,
        zeroline=False,
    )
    fig.update_yaxes(
        type='log',
        title_text='PERM Predicted (mD)',
        range=axis_cfg['range'],
        tickmode='array',
        tickvals=axis_cfg['tickvals'],
        ticktext=axis_cfg['ticktext'],
        showgrid=True,
        gridcolor=theme["grid_color"],
        gridwidth=0.7,
        showline=True,
        linecolor=theme["axis_color"],
        mirror=True,
        zeroline=False,
    )
    fig.update_layout(
        title=f"Crossplot Combined — {well_name}",
        height=650,
        template=theme["template"],
        paper_bgcolor=theme["paper_bg"],
        plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["font_color"]),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════


st.title("🔧 Permeability Auto-Calibration")
st.markdown(
    "Optimasi parameter permeabilitas empiris terhadap data **core** — per well. "
    "6 metode PERM (2 Coates + 4 Wyllie-Rose Methode) × 5 algoritma pencarian parameter."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Data")
    las_files = st.file_uploader(
        "Upload LAS file(s)", type=['las', 'LAS'],
        accept_multiple_files=True,
        help="LAS harus memiliki kurva PHIE dan SW/SWE",
    )
    core_file = st.file_uploader(
        "Upload Core CSV",
        type=['csv'],
        help="Format: WELL, DEPTH, PERM_CORE (dan opsional PORE_CORE)",
    )
    zone_file = st.file_uploader(
        "Upload Zone CSV (opsional)",
        type=['csv'],
        help="Format: WELL, MD, ZONE. Jika diunggah, kalibrasi dilakukan per zona."
    )

    st.divider()
    st.header("⚙️ Settings")

    depth_tol = st.slider("Depth tolerance (m)", 0.1, 2.0, 0.5, 0.1,
                          help="Toleransi jarak depth untuk matching core ke log")

    st.subheader("Kolom Input")
    phie_col_opt = st.text_input("Nama kolom PHIE", value="PHIE")
    sw_col_opt = st.text_input("Nama kolom SW", value="SWE",
                               help="Bisa SWE_IRR, SW, SWIRR, SWE dll.")

    st.divider()
    st.subheader("Metode Perhitungan PERM")
    all_method_names = list(METHOD_INFO.keys())
    methods_selected = st.multiselect(
        "Pilih metode",
        all_method_names,
        default=all_method_names,
    )

    with st.expander("ℹ️ Detail Rumus"):
        for m, info in METHOD_INFO.items():
            st.markdown(f"**{m}**: `{info['desc']}`")
            defaults_str = ', '.join(
                f'{k}={v}' for k, v in info['defaults'].items())
            bounds_str = ', '.join(f'{k}=[{b[0]},{b[1]}]'
                                   for k, b in zip(info['params'], info['bounds']))
            st.caption(f"Default: {defaults_str} | Bounds: {bounds_str}")

    st.divider()
    st.subheader("Metode Pencarian Parameter")
    all_opt_names = list(OPTIMIZER_INFO.keys())
    optimizers_selected = st.multiselect(
        "Pilih optimizer",
        all_opt_names,
        default=['Differential Evolution',
                 'Dual Annealing', 'Multi-Start L-BFGS-B'],
        help="Semakin banyak optimizer → semakin besar peluang menemukan R² terbaik, tapi lebih lama.",
    )

    with st.expander("ℹ️ Detail Optimizer"):
        for oname, oinfo in OPTIMIZER_INFO.items():
            st.markdown(f"**{oname}** ({oinfo['short']})")
            st.caption(oinfo['desc'])

    use_default = st.checkbox(
        "Bandingkan dengan parameter default", value=True)

    st.divider()
    run_btn = st.button("🚀 Run Optimisation",
                        type='primary', use_container_width=True)

# ── Main area ────────────────────────────────────────────────────────────────

if not las_files or not core_file:
    st.info("Upload LAS file(s) dan Core CSV di sidebar untuk memulai.")
    st.stop()

core_df = load_core_csv(core_file)
st.sidebar.success(
    f"Core: {len(core_df)} data points, {core_df['WELL'].nunique()} wells")

zone_df = None
if zone_file:
    try:
        zone_df = read_zone_csv(zone_file)
        st.sidebar.success(f"Zone: {len(zone_df)} intervals loaded")
    except Exception as e:
        st.sidebar.error(f"Gagal baca Zone CSV: {e}")

well_data = {}
for f in las_files:
    wn, df = load_las(f)
    well_data[wn] = df

st.sidebar.success(f"LAS: {len(well_data)} well(s) loaded")

wells_in_las = list(well_data.keys())
wells_in_core = sorted(core_df['WELL'].unique().tolist())

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_info, tab_preview, tab_results = st.tabs(
    ["📖 Konsep & Workflow", "📊 Data Preview", "🎯 Optimisation Results"])

with tab_info:
    st.subheader("Konsep Dasar")
    st.markdown("""
Permeabilitas reservoir dihitung secara empiris dari **PHIE** dan **SW** menggunakan
rumus Coates-Dumanoir, Coates FFI, dan family Wyllie-Rose yang dipisah menjadi:
Morris-Biggs Gas, Morris-Biggs Oil, Timur, dan Tixier.

Masalahnya: konstanta default tiap rumus **tidak selalu cocok** untuk formasi lokal.
Tool ini mencari **parameter optimal** per well menggunakan **beberapa algoritma pencarian**
sekaligus, lalu memilih yang menghasilkan **R² tertinggi** terhadap data core.
    """)

    st.divider()
    st.subheader("Persamaan Permeabilitas (dari LLS)")

    st.markdown("**1. Coates-Dumanoir**")
    st.latex(r"k = \left(\frac{C}{W^4} \cdot \frac{\phi_e^W}{S_w^W}\right)^2")
    st.caption("Parameter: C, W")

    st.markdown("**2. Coates FFI**")
    st.latex(r"k = \left(C \cdot \phi_e^2 \cdot \frac{1-S_w}{S_w}\right)^2")
    st.caption("Parameter: C")

    st.markdown("**3. WR Morris-Biggs Gas**")
    st.latex(r"k = \left(\frac{C \cdot \phi_e^D}{S_w^E}\right)^2")
    st.caption("Default: C=79, D=3, E=1")

    st.markdown("**4. WR Morris-Biggs Oil**")
    st.latex(r"k = \left(\frac{C \cdot \phi_e^D}{S_w^E}\right)^2")
    st.caption("Default: C=250, D=3, E=1")

    st.markdown("**5. WR Timur**")
    st.latex(r"k = \left(\frac{C \cdot \phi_e^D}{S_w^E}\right)^2")
    st.caption("Default: C=92.63, D=2.2, E=1")

    st.markdown("**6. WR Tixier**")
    st.latex(r"k = \left(\frac{C \cdot \phi_e^D}{S_w^E}\right)^2")
    st.caption("Default: C=250, D=3, E=1")

    st.divider()
    st.subheader("Algoritma Pencarian Parameter")
    st.markdown("""
Setiap algoritma punya **strategi berbeda** untuk menjelajahi ruang parameter.
Dengan menjalankan **beberapa optimizer sekaligus**, peluang menemukan parameter yang
menghasilkan R² terbaik menjadi lebih besar.

| Optimizer | Strategi | Kecepatan | Ketepatan |
|-----------|----------|-----------|-----------|
| **Differential Evolution** | Populasi solusi di-evolve (mutasi + crossover) | Sedang | Tinggi — robust untuk global |
| **Dual Annealing** | Simulated annealing + L-BFGS-B refinement | Sedang | Tinggi — bagus untuk rugged landscape |
| **Basin-Hopping** | Random jump + local optimization berulang | Sedang | Tinggi — banyak local minima |
| **Multi-Start L-BFGS-B** | 50 titik awal acak + gradient-based | Cepat | Sedang — bisa miss jika sangat multi-modal |
| **Grid Search** | Coba semua kombinasi di grid 20×20×... | Lambat | Pasti menemukan — tapi resolusi terbatas |

Tool akan menjalankan **semua optimizer yang dipilih** untuk setiap metode PERM, lalu
menampilkan perbandingan: parameter mana dari optimizer mana yang menghasilkan **R² tertinggi**.
    """)

    st.divider()
    st.subheader("Workflow")
    st.markdown("""
```
┌──────────────────────────────────────────────────────────────────────┐
│  1. INPUT                                                            │
│     ├── LAS file(s)  →  PHIE, SW per well                           │
│     └── Core CSV     →  PERM_CORE per well                          │
├──────────────────────────────────────────────────────────────────────┤
│  2. MATCHING  (per well)                                             │
│     └── Merge core ke log by nearest DEPTH (± tolerance)            │
├──────────────────────────────────────────────────────────────────────┤
│  3. OPTIMASI  (per well × per metode PERM × per optimizer)           │
│     ├── Objective: minimasi MSE(log10(k_pred) − log10(k_core))      │
│     ├── Jalankan N optimizer secara paralel                          │
│     └── Pilih parameter dari optimizer dengan R² tertinggi           │
├──────────────────────────────────────────────────────────────────────┤
│  4. EVALUASI & VISUALISASI                                           │
│     ├── Tabel perbandingan: metode × optimizer × R²                 │
│     ├── Log plot + Core dots                                         │
│     └── Crossplot PERM_core vs PERM_pred per metode                 │
├──────────────────────────────────────────────────────────────────────┤
│  5. EXPORT                                                           │
│     ├── CSV parameter optimal + R²                                   │
│     └── CSV full-well PERM dari best method                         │
└──────────────────────────────────────────────────────────────────────┘
```
    """)

with tab_preview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Wells in LAS")
        for w in wells_in_las:
            df = well_data[w]
            cols_avail = [c.upper() for c in df.columns]
            has_phie = phie_col_opt.upper() in cols_avail
            has_sw = sw_col_opt.upper() in cols_avail
            status = "✅" if (has_phie and has_sw) else "⚠️"
            st.write(f"{status} **{w}** — {len(df)} rows, "
                     f"PHIE={'✓' if has_phie else '✗'}, "
                     f"SW={'✓' if has_sw else '✗'}")
    with col2:
        st.subheader("Wells in Core CSV")
        for w in wells_in_core:
            n = len(core_df[core_df['WELL'] == w])
            match = '✅' if w in wells_in_las else '❌ (no LAS)'
            st.write(f"{match} **{w}** — {n} core points")

    st.subheader("Core Data Summary")
    summary = core_df.groupby('WELL').agg(
        Count=('PERM_CORE', 'count'),
        Min_Perm=('PERM_CORE', 'min'),
        Max_Perm=('PERM_CORE', 'max'),
        Median_Perm=('PERM_CORE', 'median'),
        Depth_Min=('DEPTH', 'min'),
        Depth_Max=('DEPTH', 'max'),
    ).round(2)
    st.dataframe(summary, use_container_width=True)

# ── Run optimisation ─────────────────────────────────────────────────────────

with tab_results:
    if not run_btn:
        st.info("Klik **Run Optimisation** di sidebar untuk memulai.")
        st.stop()

    if not methods_selected:
        st.warning("Pilih minimal 1 metode PERM di sidebar.")
        st.stop()
    if not optimizers_selected:
        st.warning("Pilih minimal 1 optimizer di sidebar.")
        st.stop()

    matching_wells = [w for w in wells_in_las if w in wells_in_core]
    if not matching_wells:
        st.error("Tidak ada well yang cocok antara LAS dan Core CSV.")
        st.stop()

    n_combos = len(matching_wells) * len(methods_selected) * \
        len(optimizers_selected)
    st.subheader(
        f"Optimising: {len(matching_wells)} well × "
        f"{len(methods_selected)} metode × "
        f"{len(optimizers_selected)} optimizer = {n_combos} runs"
    )

    all_results = {}
    progress = st.progress(0)
    total = len(matching_wells) * len(methods_selected)
    step = 0

    for well in matching_wells:
        log_df = well_data[well]
        log_df = merge_zone(log_df, zone_df)

        col_map = {c.upper(): c for c in log_df.columns}
        phie_actual = col_map.get(phie_col_opt.upper())
        sw_actual = col_map.get(sw_col_opt.upper())

        if not phie_actual or not sw_actual:
            st.warning(f"⚠️ **{well}**: kolom tidak ditemukan. Skip.")
            step += len(methods_selected)
            progress.progress(min(step / total, 1.0))
            continue

        well_core = core_df[core_df['WELL'] == well].copy()
        merged = merge_core_to_log(log_df, well_core, depth_tol=depth_tol)
        if not merged.empty and 'ZONE' not in merged.columns:
            merged = merge_zone(merged, zone_df)

        if len(merged) < 3:
            st.warning(
                f"⚠️ **{well}**: hanya {len(merged)} core match (min 3). Skip.")
            step += len(methods_selected)
            progress.progress(min(step / total, 1.0))
            continue

        well_results = {}       # {method_zone: best_result}
        well_opt_detail = {}    # {method_zone: {opt_name: result}}

        zones_in_well = merged['ZONE'].unique() if 'ZONE' in merged.columns else [
            'UNKNOWN']

        for zone in zones_in_well:
            z_merged = merged[merged['ZONE'] ==
                              zone] if 'ZONE' in merged.columns else merged
            if len(z_merged) < 3:
                continue

            for method in methods_selected:
                step += (1 / len(zones_in_well)
                         ) if len(zones_in_well) > 0 else 1
                progress.progress(min(step / total, 1.0))

                opt_results, best_opt = optimise_method(
                    method,
                    z_merged[phie_actual],
                    z_merged[sw_actual],
                    z_merged['PERM_CORE'],
                    optimizers_selected,
                )

                if opt_results:
                    res_key = f"{method} [{zone}]" if zone != 'UNKNOWN' else method
                    well_opt_detail[res_key] = opt_results
                    best_res = opt_results[best_opt].copy()
                    best_res['best_optimizer'] = best_opt
                    best_res['zone'] = zone
                    best_res['method'] = method
                    well_results[res_key] = best_res

                # Default comparison
                if use_default:
                    info = METHOD_INFO[method]
                    def_res = eval_default(
                        method,
                        z_merged[phie_actual].values,
                        z_merged[sw_actual].values,
                        z_merged['PERM_CORE'].values,
                        info['defaults'],
                        f'{method} (Default)',
                    )
                    if def_res is not None:
                        res_key_def = f"{method} (Default) [{zone}]" if zone != 'UNKNOWN' else f"{method} (Default)"
                        def_res['best_optimizer'] = 'Default'
                        def_res['zone'] = zone
                        def_res['method'] = method
                        well_results[res_key_def] = def_res

        if well_results:
            all_results[well] = {
                'results': well_results,
                'opt_detail': well_opt_detail,
                'log_df': log_df,
                'merged': merged,
                'phie_col': phie_actual,
                'sw_col': sw_actual,
            }

    progress.empty()

    if not all_results:
        st.error("Tidak ada hasil. Periksa data input.")
        st.stop()

    # ── Summary ──────────────────────────────────────────────────────────────

    st.subheader("📋 Summary — Best per Well")
    summary_rows = []
    for well, wd in all_results.items():
        opt_only = {k: v for k,
                    v in wd['results'].items() if '(Default)' not in k}
        if opt_only:
            best_name = max(opt_only, key=lambda k: opt_only[k]['r2'])
            best = opt_only[best_name]
            row = {
                'Well': well,
                'Best Method': best_name,
                'Best Optimizer': best.get('best_optimizer', ''),
                'CC (log)': round(best.get('cc_log', np.nan), 4),
                'R² (log)': round(best['r2'], 4),
                'RMSE (log)': round(best['rmse_log'], 4),
                'N Core': len(best['perm_core_used']),
            }
            row.update({f'{k}': round(v, 4)
                       for k, v in best['params'].items()})
            summary_rows.append(row)

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # ── Per-well detail ──────────────────────────────────────────────────────

    for well, wd in all_results.items():
        st.divider()
        st.subheader(f"🔍 Well: {well}")

        results = wd['results']
        opt_detail = wd['opt_detail']
        log_df = wd['log_df']
        merged = wd['merged']
        phie_actual = wd['phie_col']
        sw_actual = wd['sw_col']

        # ── Optimizer comparison table per method ────────────────────────────
        if opt_detail:
            st.markdown("**Perbandingan Optimizer per Metode PERM**")
            for method, opt_res in opt_detail.items():
                rows = []
                for oname, ores in opt_res.items():
                    row = {
                        'Optimizer': oname,
                        'CC (log)': round(ores.get('cc_log', np.nan), 4),
                        'R² (log)': round(ores['r2'], 4),
                        'RMSE (log)': round(ores['rmse_log'], 4),
                        'MAE (log)': round(ores['mae_log'], 4),
                        'Obj Value': round(ores['obj_value'], 6),
                    }
                    row.update({k: round(v, 4)
                               for k, v in ores['params'].items()})
                    rows.append(row)
                cmp_df = pd.DataFrame(rows).sort_values(
                    'R² (log)', ascending=False)
                best_opt_name = cmp_df.iloc[0]['Optimizer']
                st.markdown(f"*{method}* — Best: **{best_opt_name}**")
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # ── Overall metrics table ────────────────────────────────────────────
        st.markdown("**Hasil Terbaik per Metode (+ Defaults)**")
        metric_rows = []
        for mname, mres in results.items():
            row = {
                'Method': mname,
                'Optimizer': mres.get('best_optimizer', ''),
                'CC (log)': round(mres.get('cc_log', np.nan), 4),
                'R² (log)': round(mres['r2'], 4),
                'RMSE (log)': round(mres['rmse_log'], 4),
                'MAE (log)': round(mres['mae_log'], 4),
            }
            row.update({k: round(v, 4) for k, v in mres['params'].items()})
            metric_rows.append(row)

        metrics_df = pd.DataFrame(metric_rows).sort_values(
            'R² (log)', ascending=False)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ── Plots ────────────────────────────────────────────────────────────
        opt_only = {k: v for k, v in results.items()
                    if '(Default)' not in k and not k.startswith('WR ')}

        if opt_only:
            st.markdown("**Log Plot (Best dari tiap metode)**")
            fig_log = plot_log_perm(well, log_df, merged, opt_only,
                                    phie_actual, sw_actual)
            st.plotly_chart(fig_log, use_container_width=True)

        if opt_only:
            st.markdown("**Crossplot per Metode**")
            fig_grid = plot_crossplot_grid(well, opt_only)
            if fig_grid:
                st.plotly_chart(fig_grid, use_container_width=True)

        if len(opt_only) > 1:
            st.markdown("**Crossplot Combined**")
            fig_comb = plot_crossplot_combined(well, opt_only)
            st.plotly_chart(fig_comb, use_container_width=True)

    # ── Export ───────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("📥 Export Results")

    # Full detail export (all optimizers × all methods)
    export_rows = []
    for well, wd in all_results.items():
        for method, opt_res in wd.get('opt_detail', {}).items():
            for oname, ores in opt_res.items():
                row = {
                    'Well': well,
                    'Method': method,
                    'Optimizer': oname,
                    'R2_log': ores['r2'],
                    'RMSE_log': ores['rmse_log'],
                    'MAE_log': ores['mae_log'],
                    'Obj_Value': ores['obj_value'],
                }
                row.update(ores['params'])
                export_rows.append(row)
        # Defaults
        for mname, mres in wd['results'].items():
            if '(Default)' in mname or mname.startswith('WR '):
                row = {
                    'Well': well,
                    'Method': mname,
                    'Optimizer': 'Default',
                    'R2_log': mres['r2'],
                    'RMSE_log': mres['rmse_log'],
                    'MAE_log': mres['mae_log'],
                }
                row.update(mres['params'])
                export_rows.append(row)

    if export_rows:
        export_df = pd.DataFrame(export_rows)
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Results — All Optimizers (CSV)",
            data=csv_bytes,
            file_name="perm_calibration_full_results.csv",
            mime="text/csv",
        )

    st.markdown("**Export computed permeability (full-well):**")
    export_well = st.selectbox("Pilih well", list(all_results.keys()))
    if export_well:
        wd = all_results[export_well]
        results_well = wd['results']
        opt_only = {k: v for k, v in results_well.items()
                    if '(Default)' not in k}

        if opt_only:
            log_df_exp = wd['log_df'].copy()
            phie_full = log_df_exp[wd['phie_col']].values
            sw_full = log_df_exp[wd['sw_col']].values

            # Calculate PERM per zone using best method/params for that zone
            perm_computed = np.full(len(log_df_exp), np.nan)

            # Group by zone to apply per-zone optimal params
            zones_exp = log_df_exp['ZONE'].unique() if 'ZONE' in log_df_exp.columns else [
                'UNKNOWN']

            for z in zones_exp:
                z_mask = (log_df_exp['ZONE'] == z) if 'ZONE' in log_df_exp.columns else np.ones(
                    len(log_df_exp), dtype=bool)
                # Find best performing method for this specific zone
                z_res_list = {k: v for k, v in opt_only.items()
                              if v.get('zone') == z}
                if not z_res_list:
                    # Fallback to absolute best if zone-specific not found
                    best_key = max(opt_only, key=lambda k: opt_only[k]['r2'])
                    z_res = opt_only[best_key]
                else:
                    best_z_key = max(
                        z_res_list, key=lambda k: z_res_list[k]['r2'])
                    z_res = z_res_list[best_z_key]

                z_method = z_res['method']
                z_params = z_res['params']

                ph_z = phie_full[z_mask]
                sw_z = sw_full[z_mask]

                if len(ph_z) > 0:
                    perm_computed[z_mask] = compute_perm(
                        z_method, ph_z, sw_z, z_params)

            log_df_exp['PERM_COMPUTED'] = perm_computed

            csv2 = log_df_exp[['DEPTH', 'ZONE', wd['phie_col'], wd['sw_col'],
                               'PERM_COMPUTED']].to_csv(index=False).encode('utf-8')
            st.download_button(
                f"⬇️ Download {export_well} — Computed PERM (CSV)",
                data=csv2,
                file_name=f"perm_computed_{export_well}.csv",
                mime="text/csv",
                help="Menggunakan parameter terbaik per zona yang telah dikalibrasi."
            )
