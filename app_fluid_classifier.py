"""
Fluid·ML  —  Two-Stage Fluid Classification Dashboard
═════════════════════════════════════════════════════════════════════════
Pipeline (terpisah):
    Stage-1  →  HC (Hydrocarbon = Oil + Gas)  vs  W (Water)
    Stage-2  →  O (Oil)  vs  G (Gas)   (hanya pada baris HC)

Setiap stage punya konfigurasi sendiri:
    - log predictor (features) yang berbeda boleh
    - algoritma model & hyperparameter sendiri
    - strategi imbalance handling sendiri
    - tombol training sendiri

Input log dari LAS:
    GR, GR_NORM, RT, NPHI, RHOB, VSH, PHIE, SW,
    RGSA, NGSA, DGSA, RPBE, RGBE, SWGRAD,
    ZONE, LITHO_CODE, DNS, DNBE, SPBE, IQUAL

Label fluid (FLUID atau FLUID_CODE):
    FLUID huruf  : G, O, W, OP, GP
    FLUID_CODE   : 7=Gas, 8=GP, 9=Oil, 10=OP, 11=Water
    Aturan:
      OP → O   (possible oil dijadikan oil)
      GP → G   (possible gas dijadikan gas)

LITHO_CODE   : 1=Sand (kuning), 4=Coal (hitam), 6=Shale (abu2)

Catatan:
    * Hanya baris dengan IQUAL > 0 yang dipakai (sisanya tidak punya label).
    * Class imbalance ditangani: class_weight='balanced', undersample, atau oversample.
    * Param model di-translate per algoritma (CatBoost: iterations/depth, RF: no LR).
"""

import io
import zipfile
import hashlib
import warnings
import re
import json
import pickle

import numpy as np
import pandas as pd
import lasio

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fluid·ML",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;
  --border:#30363d;--border2:#21262d;
  --accent:#58a6ff;--a2:#f0a500;--a3:#3fb950;
  --danger:#f85149;--warn:#e3b341;
  --text:#e6edf3;--muted:#7d8590;--muted2:#484f58;
  --r:6px;
}
html,body,[class*="css"]{
  font-family:'IBM Plex Sans',sans-serif;
  background:var(--bg)!important;color:var(--text)!important;
}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] label{font-size:0.79rem!important;}
.main .block-container{padding:1.4rem 2rem;max-width:100%;}
.logo{font-family:'IBM Plex Mono',monospace;font-size:1.85rem;font-weight:600;color:var(--accent);letter-spacing:-1.5px;}
.logo-sub{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;}
.sec{font-family:'IBM Plex Mono',monospace;font-size:0.64rem;color:var(--muted2);text-transform:uppercase;letter-spacing:2px;padding:.65rem 0 .25rem;border-top:1px solid var(--border2);margin-top:.5rem;}
.stage-hdr{font-family:'IBM Plex Mono',monospace;font-size:.78rem;font-weight:600;color:var(--accent);padding:.4rem .6rem;background:var(--bg3);border-radius:var(--r);border:1px solid var(--border);margin:.3rem 0 .5rem;}
.kpi-row{display:flex;gap:10px;flex-wrap:wrap;margin:.7rem 0;}
.kpi{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:11px 16px;flex:1;min-width:120px;}
.kl{font-family:'IBM Plex Mono',monospace;font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;}
.kv{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;}
.ks{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:var(--muted);margin-top:1px;}
.good{color:var(--a3)!important;}.ok{color:var(--a2)!important;}.bad{color:var(--danger)!important;}.na{color:var(--muted)!important;}
.pill{display:inline-block;background:var(--bg3);border:1px solid var(--border);border-radius:20px;padding:2px 9px;font-family:'IBM Plex Mono',monospace;font-size:.67rem;color:var(--accent);margin:2px;}
.ibox{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:var(--r);padding:8px 13px;font-size:.78rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;margin:.35rem 0;line-height:1.6;}
.wbox{border-left-color:var(--warn)!important;}
.ebox{border-left-color:var(--danger)!important;}
.stButton>button{background:var(--accent)!important;color:#000!important;border:none!important;font-family:'IBM Plex Mono',monospace!important;font-weight:600!important;font-size:.82rem!important;border-radius:var(--r)!important;padding:.47rem 1.3rem!important;}
.stButton>button:hover{background:#79b8ff!important;transform:translateY(-1px);}
.stButton>button:disabled{background:var(--muted2)!important;color:var(--muted)!important;}
[data-testid="stTabs"] [role="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:var(--muted)!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
[data-testid="stExpander"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;}
hr{border-color:var(--border2)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS — Mapping fluid, litho, warna, log alias
# ══════════════════════════════════════════════════════════════════
NULL_VALUES = [-9999.25, -9999., -999.25, -
               999., -9998., 9999., 9998., 1e30, -1e30]

MNEMONIC_MAP = {
    'DEPT': 'DEPTH', 'MD': 'DEPTH',
    'GAMMA': 'GR', 'GRD': 'GR', 'SGR': 'GR', 'CGR': 'GR', 'GRN': 'GR_NORM',
    'CNPHI': 'NPHI', 'TNPH': 'NPHI', 'NPOR': 'NPHI', 'CNCF': 'NPHI', 'NEU': 'NPHI',
    'RHOZ': 'RHOB', 'DEN': 'RHOB', 'RHOC': 'RHOB', 'ZDEN': 'RHOB', 'RHOG': 'RHOB',
    'ILD': 'RT', 'LLD': 'RT', 'M2RX': 'RT', 'HDRS': 'RT', 'AT90': 'RT',
    'RDEEP': 'RT', 'RDEP': 'RT', 'RD': 'RT',
    'VCL': 'VSH', 'VSHALE': 'VSH', 'VSH_D': 'VSH',
    'PHIT': 'PHIE', 'PORE': 'PHIE', 'POR': 'PHIE', 'PHI': 'PHIE',
    'SWT': 'SW', 'SWE': 'SW', 'SW_D': 'SW',
    'FORM': 'ZONE', 'FORMATION': 'ZONE',
}

# Mapping FLUID huruf → FLUID_CODE numerik (sesuai instruksi user)
FLUID_LETTER_TO_CODE = {'G': 7, 'GP': 8, 'O': 9, 'OP': 10, 'W': 11}

# Code → kategori 3-kelas akhir (setelah OP→O, GP→G)
FLUID_CODE_TO_CLASS3 = {7: 'G', 8: 'G', 9: 'O', 10: 'O', 11: 'W'}

# Stage-1 / Stage-2 mapping
CLASS3_TO_HC_W = {'G': 'HC', 'O': 'HC', 'W': 'W'}
CLASS3_TO_OG = {'O': 'O', 'G': 'G'}

# Warna fluid (track plot)
FLUID_COLOR = {
    'O': '#3fb950',   # Oil   = hijau
    'W': '#58a6ff',   # Water = biru
    'G': '#f85149',   # Gas   = merah
    'OP': '#22d3ee',  # OP    = cyan
    'GP': '#ec4899',  # GP    = pink
    'HC': '#e3b341',  # HC    = kuning (fallback)
}

# Litho mapping
LITHO_CODE_TO_NAME = {1: 'Sand', 4: 'Coal', 6: 'Shale'}
LITHO_COLOR = {'Sand': '#f0d000', 'Coal': '#000000', 'Shale': '#9aa0a6'}

# Log dasar yang dipakai sebagai fitur kandidat (termasuk engineered)
ALL_FEATURE_LOGS = [
    'GR', 'GR_NORM',
    'RT', 'LOG_RT',
    'NPHI', 'RHOB',
    'VSH', 'PHIE', 'SW',
    'RGSA', 'LOG_RGSA', 'NGSA', 'DGSA',
    'RGSAA', 'NGSAA', 'DGSAA',
    'RPBE', 'RGBE', 'SWGRAD',
    'LITHO_CODE',
    'DNS', 'DNSV', 'DNS_CUTOFF', 'DNBE', 'SPBE',
]


# ══════════════════════════════════════════════════════════════════
# DNS_CUTOFF Equations (port dari lls/dns_dnsv.py)
# Format: {STRUKTUR: {ZONE: {"var": "PHIE"|"GR", "intercept": float, "slope": float}}}
# DNS_CUTOFF = intercept + slope * (PHIE atau GR)
# ══════════════════════════════════════════════════════════════════
DNS_CUTOFF_EQUATIONS = {
    "Gunung Kemala": {
        "TAF1": {"var": "PHIE", "intercept": -0.053, "slope": 0.89},
        "TAF2": {"var": "PHIE", "intercept": -0.01, "slope": 0.61},
    },
    "Abab": {
        "BRF": {"var": "GR", "intercept": 0.14, "slope": -0.001},
        "TAF": {"var": "GR", "intercept": 0.22, "slope": -0.002},
    },
    "Benuang": {"TAF": {"var": "PHIE", "intercept": -0.06, "slope": 1.25}},
    "Mangun Jaya": {"TAF": {"var": "PHIE", "intercept": -0.06, "slope": 0.24}},
    "Lembak": {"TAF": {"var": "PHIE", "intercept": -0.08, "slope": 1.02}},
    "Karangan": {"TAF": {"var": "PHIE", "intercept": -0.18, "slope": 1.30}},
    "Talang Jimar": {"TAF": {"var": "PHIE", "intercept": -0.13, "slope": 1.28}},
    "Belimbing": {"TAF1": {"var": "PHIE", "intercept": -0.06, "slope": 1.09}},
    "Bentayan": {"TAF": {"var": "PHIE", "intercept": 0.039, "slope": 0.61}},
    "Benakat Barat": {
        "GUF":   {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_1": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_2": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_3": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_4": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
    },
    "Limau Barat": {
        "TAF1": {"var": "PHIE", "intercept": -0.081, "slope": 0.796},
        "TAF2": {"var": "PHIE", "intercept": -0.012, "slope": 0.70},
    },
    "Limau Tengah": {"TAF": {"var": "PHIE", "intercept": -0.05, "slope": 0.8}},
    "Beringin": {
        "GUF": {"var": "PHIE", "intercept": -0.2579, "slope": 1.357},
        "BRF": {"var": "PHIE", "intercept": -0.037, "slope": 0.67},
        "TAF": {"var": "PHIE", "intercept": -0.078, "slope": 1.34},
    },
    "Prabumenang": {"BRF": {"var": "PHIE", "intercept": 0.09, "slope": 0.574}},
    "Musi": {"BRF": {"var": "PHIE", "intercept": 0.016, "slope": 0.72}},
    "Betung": {"TAF": {"var": "PHIE", "intercept": -0.10, "slope": 0.99}},
    "Niru": {"TAF": {"var": "PHIE", "intercept": -0.02, "slope": 0.79}},
    "Prabumulih Barat": {"TAF": {"var": "PHIE", "intercept": -0.005, "slope": 0.87}},
}

PLOTLY_BASE = dict(
    paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
    font=dict(family='IBM Plex Mono', color='#e6edf3', size=10),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k, v in {
    'all_wells': {},
    'zone_df': None,
    'combined_df': None,
    'qc_log': None,
    'normalized': False,
    'gr_norm_params': {},
    # Multi-struktur state: list of {name, wells, zone_df, zip_hash, zone_hash}
    'structures': [],
    'n_structures': 1,
    # Per-stage results
    'fc_s1_results': None,
    'fc_s2_results': None,
    'fc_s1_cfg': {},
    'fc_s2_cfg': {},
    'fc_test_wells': [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
# LAS LOADING
# ══════════════════════════════════════════════════════════════════
def _normalize_well_name(name: str) -> str:
    """Normalisasi nama sumur → PREFIX-NNN[SUFFIX] (misal BN-63 → BN-063)."""
    m = re.match(r'^([A-Za-z]+[-_])(\d+)([A-Za-z]*)$', name.strip())
    if m:
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        return f"{prefix.upper()}{digits.zfill(3)}{suffix.upper()}"
    return name.upper()


def _auto_convert_rhob_to_gcc(df: pd.DataFrame) -> pd.DataFrame:
    """RHOB kg/m3 → g/cc bila median > 100."""
    df = df.copy()
    if 'RHOB' not in df.columns:
        return df
    rhob_num = pd.to_numeric(df['RHOB'], errors='coerce')
    valid = rhob_num.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return df
    if float(valid.median()) > 100:
        df['RHOB'] = rhob_num / 1000.0
    return df


def _coerce_fluid_label(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan kolom FLUID_CODE ada (FLUID huruf → code bila perlu)."""
    df = df.copy()
    has_code = 'FLUID_CODE' in df.columns
    has_letter = 'FLUID' in df.columns
    if has_code:
        df['FLUID_CODE'] = pd.to_numeric(df['FLUID_CODE'], errors='coerce')
    elif has_letter:
        s = df['FLUID'].astype(str).str.strip().str.upper()
        df['FLUID_CODE'] = s.map(FLUID_LETTER_TO_CODE).astype(float)
    else:
        df['FLUID_CODE'] = np.nan
    return df


def read_las_bytes(content: bytes, well_name: str):
    """Baca LAS dari bytes → DataFrame standar."""
    def _parse(las_str: str):
        try:
            return lasio.read(io.StringIO(las_str),
                              ignore_header_errors=True,
                              read_policy='quick',
                              null_policy='none')
        except Exception:
            return lasio.read(io.StringIO(las_str), ignore_header_errors=True)

    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            las_str = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        las_str = content.decode('utf-8', errors='replace')

    try:
        las = _parse(las_str)
    except Exception as e:
        st.warning(f"⚠ Skip {well_name}: {e}")
        return None

    df = las.df().reset_index()
    df.columns = [c.strip().upper() for c in df.columns]
    df.rename(columns={k: v for k, v in MNEMONIC_MAP.items() if k in df.columns},
              inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    if 'DEPTH' not in df.columns:
        df.rename(columns={df.columns[0]: 'DEPTH'}, inplace=True)

    try:
        nv = float(las.well.NULL.value)
    except Exception:
        nv = -9999.25
    df.replace(nv, np.nan, inplace=True)
    for nv2 in NULL_VALUES:
        df.replace(nv2, np.nan, inplace=True)

    if 'RT' in df.columns:
        df.loc[df['RT'] <= 0, 'RT'] = np.nan
    df = _auto_convert_rhob_to_gcc(df)
    df = _coerce_fluid_label(df)

    if 'ZONE' not in df.columns:
        df['ZONE'] = 'UNKNOWN'
    if 'IQUAL' not in df.columns:
        df['IQUAL'] = np.nan

    df['WELL_NAME'] = _normalize_well_name(well_name)
    return df.sort_values('DEPTH').reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_zip_cached(file_bytes: bytes) -> dict:
    wells = {}
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        las_names = sorted([f for f in zf.namelist()
                            if f.lower().endswith('.las')
                            and not f.startswith('__')])
        for fname in las_names:
            with zf.open(fname) as f:
                content = f.read()
            base = fname.split('/')[-1]
            wname = base.rsplit('.', 1)[0].upper()
            df = read_las_bytes(content, wname)
            if df is not None:
                wells[wname] = df
    return wells


# ── ZONE CSV ──────────────────────────────────────────────────────
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
    depth_col = next((c for c in df.columns
                      if c in ('MD', 'DEPTH', 'DEPTH_MD', 'TVD') or 'DEPTH' in c), None)
    zone_col = next((c for c in df.columns
                     if c in ('SURFACE', 'ZONE', 'FORMATION', 'FORM',
                              'UNIT', 'LAYER', 'NAME', 'MARKER')), None)
    if not all([well_col, depth_col, zone_col]):
        cols = df.columns.tolist()
        if len(cols) >= 3:
            well_col, depth_col, zone_col = cols[0], cols[1], cols[2]
        else:
            raise ValueError(f"Zone CSV tidak punya kolom yang cukup: {cols}")

    df = df[[well_col, depth_col, zone_col]].copy()
    df.columns = ['WELL_NAME', 'MD', 'ZONE_NAME']
    df['WELL_NAME'] = df['WELL_NAME'].str.strip().apply(_normalize_well_name)
    df['MD'] = pd.to_numeric(df['MD'].str.strip().str.replace(',', '.', regex=False),
                             errors='coerce')
    df['ZONE_NAME'] = df['ZONE_NAME'].str.strip()

    records = []
    for well, grp in df.groupby('WELL_NAME'):
        grp = grp.sort_values('MD').reset_index(drop=True)
        valid = grp[grp['ZONE_NAME'].notna() & (grp['ZONE_NAME'] != '')
                    & (grp['ZONE_NAME'].str.upper() != 'NAN')].reset_index(drop=True)
        for _, row in valid.iterrows():
            top = row['MD']
            nxt = grp[grp['MD'] > top]['MD']
            bot = float(nxt.iloc[0]) if len(nxt) > 0 else 99999.0
            records.append({'WELL_NAME': well, 'DEPTH_TOP': float(top),
                            'DEPTH_BOT': bot, 'ZONE': row['ZONE_NAME']})
    return pd.DataFrame(records)


def merge_zone(df: pd.DataFrame, zone_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if zone_df is None or len(zone_df) == 0:
        return df
    wname = _normalize_well_name(df['WELL_NAME'].iloc[0])
    zw = zone_df[zone_df['WELL_NAME'] == wname]
    for _, r in zw.iterrows():
        m = (df['DEPTH'] >= r['DEPTH_TOP']) & (df['DEPTH'] < r['DEPTH_BOT'])
        df.loc[m, 'ZONE'] = r['ZONE']
    return df


def build_combined(wells: dict, zone_df) -> pd.DataFrame:
    dfs = [merge_zone(df, zone_df) for df in wells.values()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_combined_multi(structures: list) -> pd.DataFrame:
    """Multi-struktur combiner: tag setiap baris dengan kolom STRUKTUR."""
    parts = []
    for s in structures:
        wells = s.get('wells') or {}
        if not wells:
            continue
        sname = (s.get('name') or 'UNKNOWN').strip()
        zdf = s.get('zone_df')
        for _wname, wdf in wells.items():
            merged = merge_zone(wdf, zdf).copy()
            merged['STRUKTUR'] = sname
            parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — RGSAA / NGSAA / DGSAA, log RT/RGSA, DNS/DNSV/DNS_CUTOFF/FLUID_DNS
# ══════════════════════════════════════════════════════════════════
def _dns(rhob, nphi):
    """DNS = ((2.71 - RHOB)/1.71) - NPHI."""
    return ((2.71 - rhob) / 1.71) - nphi


def _dnsv(rhob, nphi, rhob_sh, nphi_sh, vsh):
    """DNSV: DNS dengan koreksi shale-volume."""
    rhob_corv = rhob + vsh * (2.65 - rhob_sh)
    nphi_corv = nphi + vsh * (0 - nphi_sh)
    return ((2.71 - rhob_corv) / 1.71) - nphi_corv


def _shale_point(df: pd.DataFrame):
    """Estimasi (nphi_sh, rhob_sh) dari data: prefer titik VSH > 0.7;
    fallback ke percentile tinggi NPHI/RHOB."""
    nphi = pd.to_numeric(df.get('NPHI'), errors='coerce')
    rhob = pd.to_numeric(df.get('RHOB'), errors='coerce')
    valid = nphi.notna() & rhob.notna()
    if not valid.any():
        return 0.40, 2.60
    if 'VSH' in df.columns:
        vsh = pd.to_numeric(df['VSH'], errors='coerce')
        sh = valid & (vsh > 0.7)
        if sh.sum() > 10:
            return float(np.nanmedian(nphi[sh])), float(np.nanmedian(rhob[sh]))
    return (float(np.nanpercentile(nphi[valid], 95)),
            float(np.nanpercentile(rhob[valid], 95)))


def _calc_dns_cutoff_vec(df: pd.DataFrame,
                         struktur_col: str = 'STRUKTUR',
                         zone_col: str = 'ZONE',
                         phie_col: str = 'PHIE',
                         gr_col: str = 'GR') -> pd.Series:
    """Lookup DNS_CUTOFF berdasar (STRUKTUR, ZONE) → intercept + slope*(PHIE|GR)."""
    cutoff = pd.Series(np.nan, index=df.index)
    if struktur_col not in df.columns or zone_col not in df.columns:
        return cutoff
    eq_lookup = {}
    for sname, zones in DNS_CUTOFF_EQUATIONS.items():
        for zname, eq in zones.items():
            eq_lookup[(sname.upper(), zname.upper())] = eq
    su = df[struktur_col].astype(str).str.strip().str.upper()
    zu = df[zone_col].astype(str).str.strip().str.upper()
    for (s, z), eq in eq_lookup.items():
        m = (su == s) & (zu == z)
        if not m.any():
            continue
        var = phie_col if eq['var'] == 'PHIE' else gr_col
        if var not in df.columns:
            continue
        vals = pd.to_numeric(df.loc[m, var], errors='coerce')
        cutoff.loc[m] = eq['intercept'] + eq['slope'] * vals
    return cutoff


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom turunan:
       LOG_RT, LOG_RGSA, RGSAA, NGSAA, DGSAA, DNS, DNSV, DNS_CUTOFF, FLUID_DNS.
       Dihitung sebelum QC sehingga dapat dipilih sebagai feature di sidebar.
       DNS_CUTOFF butuh STRUKTUR + ZONE; FLUID_DNS butuh DNS & DNS_CUTOFF (IQUAL>0)."""
    if df is None or df.empty:
        return df
    df = df.copy()

    # ── Log transforms (RT & RGSA → log10) ─────────────────────────
    if 'RT' in df.columns:
        rt_num = pd.to_numeric(df['RT'], errors='coerce')
        df['LOG_RT'] = np.log10(rt_num.where(rt_num > 0))
    if 'RGSA' in df.columns:
        rgsa_num = pd.to_numeric(df['RGSA'], errors='coerce')
        df['LOG_RGSA'] = np.log10(rgsa_num.where(rgsa_num > 0))

    # ── Selisih log standar vs sintetik (XGSAA = X_in - XGSA) ──────
    if 'RT' in df.columns and 'RGSA' in df.columns:
        df['RGSAA'] = (pd.to_numeric(df['RT'], errors='coerce')
                       - pd.to_numeric(df['RGSA'], errors='coerce'))
    if 'NPHI' in df.columns and 'NGSA' in df.columns:
        df['NGSAA'] = (pd.to_numeric(df['NPHI'], errors='coerce')
                       - pd.to_numeric(df['NGSA'], errors='coerce'))
    if 'RHOB' in df.columns and 'DGSA' in df.columns:
        df['DGSAA'] = (pd.to_numeric(df['RHOB'], errors='coerce')
                       - pd.to_numeric(df['DGSA'], errors='coerce'))

    # ── DNS & DNSV (gunakan VSH bila ada; shale-point per STRUKTUR) ─
    if 'RHOB' in df.columns and 'NPHI' in df.columns:
        rhob = pd.to_numeric(df['RHOB'], errors='coerce')
        nphi = pd.to_numeric(df['NPHI'], errors='coerce')
        df['DNS'] = _dns(rhob, nphi)

        if 'VSH' in df.columns:
            vsh = pd.to_numeric(df['VSH'], errors='coerce').fillna(0)
            df['DNSV'] = np.nan
            if 'STRUKTUR' in df.columns:
                for _s, idx in df.groupby('STRUKTUR').groups.items():
                    sub = df.loc[idx]
                    nphi_sh, rhob_sh = _shale_point(sub)
                    df.loc[idx, 'DNSV'] = _dnsv(
                        rhob.loc[idx], nphi.loc[idx],
                        rhob_sh, nphi_sh, vsh.loc[idx])
            else:
                nphi_sh, rhob_sh = _shale_point(df)
                df['DNSV'] = _dnsv(rhob, nphi, rhob_sh, nphi_sh, vsh)

    # ── DNS_CUTOFF dari (STRUKTUR, ZONE) ───────────────────────────
    df['DNS_CUTOFF'] = _calc_dns_cutoff_vec(df)

    # ── FLUID_DNS: G bila DNS > CUTOFF, O bila DNS < CUTOFF (IQUAL>0) ─
    if 'DNS' in df.columns and 'DNS_CUTOFF' in df.columns:
        df['FLUID_DNS'] = pd.NA
        valid = df['DNS'].notna() & df['DNS_CUTOFF'].notna()
        if 'IQUAL' in df.columns:
            iq = pd.to_numeric(df['IQUAL'], errors='coerce').fillna(0)
            valid = valid & (iq > 0)
        df.loc[valid & (df['DNS'] > df['DNS_CUTOFF']), 'FLUID_DNS'] = 'G'
        df.loc[valid & (df['DNS'] < df['DNS_CUTOFF']), 'FLUID_DNS'] = 'O'

    return df


# ══════════════════════════════════════════════════════════════════
# GR NORMALIZATION
# ══════════════════════════════════════════════════════════════════
def compute_gr_norm_params(df, p_low=3, p_high=97):
    if 'GR' not in df.columns:
        return {}
    gr_valid = df[df['GR'].notna()].copy()
    if gr_valid.empty:
        return {}
    params = {
        'global_new': {
            'p_low': float(np.nanpercentile(gr_valid['GR'], p_low)),
            'p_high': float(np.nanpercentile(gr_valid['GR'], p_high)),
            'N': int(len(gr_valid)),
        },
        'well_old': {},
    }
    for well, grp in gr_valid.groupby('WELL_NAME'):
        vals = grp['GR'].dropna()
        if len(vals) < 10:
            params['well_old'][well] = {
                'p_low': params['global_new']['p_low'],
                'p_high': params['global_new']['p_high'],
                'N': int(len(vals)),
            }
        else:
            params['well_old'][well] = {
                'p_low': float(np.nanpercentile(vals, p_low)),
                'p_high': float(np.nanpercentile(vals, p_high)),
                'N': int(len(vals)),
            }
    return params


def apply_gr_norm(df, params):
    df = df.copy()
    df['GR_NORM'] = np.nan
    if not params or 'GR' not in df.columns:
        return df
    p_low_new = params['global_new']['p_low']
    p_high_new = params['global_new']['p_high']
    well_old = params['well_old']
    for well, idx in df.groupby('WELL_NAME').groups.items():
        old = well_old.get(well)
        if old is None:
            p_low_old, p_high_old = p_low_new, p_high_new
        else:
            p_low_old, p_high_old = old['p_low'], old['p_high']
        den = p_high_old - p_low_old
        if pd.isna(den) or den == 0:
            continue
        grz = (p_high_new - p_low_new) / den
        df.loc[idx, 'GR_NORM'] = grz * \
            (df.loc[idx, 'GR'] - p_low_old) + p_low_new
    return df


# ══════════════════════════════════════════════════════════════════
# FLUID LABEL ENGINEERING
# ══════════════════════════════════════════════════════════════════
def build_fluid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom FLUID_CLASS3 (O/G/W), FLUID_HC_W (HC/W), FLUID_OG (O/G)."""
    df = df.copy()
    if 'FLUID_CODE' not in df.columns:
        df['FLUID_CODE'] = np.nan
    code = pd.to_numeric(df['FLUID_CODE'], errors='coerce')
    df['FLUID_CLASS3'] = code.map(FLUID_CODE_TO_CLASS3)
    df['FLUID_HC_W'] = df['FLUID_CLASS3'].map(CLASS3_TO_HC_W)
    df['FLUID_OG'] = df['FLUID_CLASS3'].map(CLASS3_TO_OG)
    inv = {v: k for k, v in FLUID_LETTER_TO_CODE.items()}
    df['FLUID_LETTER'] = code.map(inv)
    return df


# ══════════════════════════════════════════════════════════════════
# QC PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_qc_pipeline(df, base_logs=('GR', 'RT', 'NPHI', 'RHOB'),
                    use_iqual: bool = True):
    """QC: drop all-NaN logs, RT≤0, IQUAL>0 filter, drop NaN label."""
    log = {}
    n0 = len(df)
    df = df.copy()

    cols = [c for c in base_logs if c in df.columns]
    if cols:
        before = len(df)
        df = df.dropna(subset=cols, how='all')
        log['drop_all_nan_logs'] = int(before - len(df))
    else:
        log['drop_all_nan_logs'] = 0

    if 'RT' in df.columns:
        bad_rt = int((df['RT'].notna() & (df['RT'] <= 0)).sum())
        df.loc[df['RT'] <= 0, 'RT'] = np.nan
        log['rt_invalid_to_nan'] = bad_rt
    else:
        log['rt_invalid_to_nan'] = 0

    if use_iqual and 'IQUAL' in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df['IQUAL'], errors='coerce') > 0].copy()
        log['drop_iqual_le_0'] = int(before - len(df))
    else:
        log['drop_iqual_le_0'] = 0

    df = _coerce_fluid_label(df)
    before = len(df)
    df = df[df['FLUID_CODE'].notna()].copy()
    log['drop_fluid_label_nan'] = int(before - len(df))

    valid_codes = set(FLUID_CODE_TO_CLASS3.keys())
    before = len(df)
    df = df[df['FLUID_CODE'].astype(int).isin(valid_codes)].copy()
    log['drop_fluid_unknown_code'] = int(before - len(df))

    df = build_fluid_labels(df)
    log['total_dropped'] = int(n0 - len(df))
    log['remaining'] = int(len(df))
    return df, log


# ══════════════════════════════════════════════════════════════════
# CLASS BALANCE HANDLING
# ══════════════════════════════════════════════════════════════════
def balance_dataframe(df: pd.DataFrame, label_col: str,
                      strategy: str = 'none', random_state: int = 42) -> pd.DataFrame:
    """Resampling: none / undersample (mayoritas → minoritas) / oversample (minoritas → mayoritas)."""
    if strategy == 'none' or df.empty:
        return df.copy()
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        return df.copy()

    rng = np.random.default_rng(random_state)
    out = []
    if strategy == 'undersample':
        n_min = int(counts.min())
        for cls, _ in counts.items():
            grp = df[df[label_col] == cls]
            idx = rng.choice(grp.index, size=n_min, replace=False)
            out.append(df.loc[idx])
    elif strategy == 'oversample':
        n_max = int(counts.max())
        for cls, _ in counts.items():
            grp = df[df[label_col] == cls]
            need = n_max - len(grp)
            out.append(grp)
            if need > 0:
                idx = rng.choice(grp.index, size=need, replace=True)
                out.append(df.loc[idx])
    else:
        return df.copy()
    return pd.concat(out, ignore_index=False).sample(frac=1, random_state=random_state)


def compute_class_weights(y) -> dict:
    classes = np.unique(y)
    if len(classes) < 2:
        return {c: 1.0 for c in classes}
    w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return {c: float(wi) for c, wi in zip(classes, w)}


# ══════════════════════════════════════════════════════════════════
# MODEL BUILDER  (param translator per-algoritma)
# ══════════════════════════════════════════════════════════════════
def _normalize_params_for_model(name: str, raw: dict) -> dict:
    """
    Translate UI param generik (n_estimators, learning_rate, max_depth)
    ke kwargs spesifik per model.
    - CatBoost: pakai `iterations` & `depth`, BUKAN n_estimators/max_depth.
    - RandomForest / ExtraTrees: tidak menerima learning_rate.
    - max_depth = -1 → default model (None untuk RF/CB-depth/etc).
    """
    name = name.lower()
    n_est = raw.get('n_estimators')
    lr = raw.get('learning_rate')
    md = raw.get('max_depth')
    p = {}

    if name == 'lightgbm':
        if n_est is not None:
            p['n_estimators'] = int(n_est)
        if lr is not None:
            p['learning_rate'] = float(lr)
        if md not in (None, -1):
            p['max_depth'] = int(md)
        p.setdefault('num_leaves', 31)
        p.setdefault('subsample', 0.8)
        p.setdefault('colsample_bytree', 0.8)
        p.setdefault('random_state', 42)
        p.setdefault('verbose', -1)
        return p

    if name in ('randomforest', 'extratrees'):
        if n_est is not None:
            p['n_estimators'] = int(n_est)
        if md not in (None, -1):
            p['max_depth'] = int(md)
        # learning_rate diabaikan (model bagging)
        p.setdefault('random_state', 42)
        p.setdefault('n_jobs', -1)
        return p

    if name == 'xgboost':
        if n_est is not None:
            p['n_estimators'] = int(n_est)
        if lr is not None:
            p['learning_rate'] = float(lr)
        if md not in (None, -1):
            p['max_depth'] = int(md)
        p.setdefault('subsample', 0.8)
        p.setdefault('colsample_bytree', 0.8)
        p.setdefault('random_state', 42)
        p.setdefault('eval_metric', 'logloss')
        return p

    if name == 'catboost':
        # CatBoost: iterations (bukan n_estimators), depth (bukan max_depth)
        if n_est is not None:
            p['iterations'] = int(n_est)
        if lr is not None:
            p['learning_rate'] = float(lr)
        if md not in (None, -1):
            p['depth'] = int(md)
        p.setdefault('random_seed', 42)
        p.setdefault('verbose', 0)
        return p

    return p


def build_classifier(name: str, params: dict, class_weight=None):
    """Bangun classifier sesuai pilihan user. class_weight di-pass jika model support."""
    name = (name or 'lightgbm').lower()
    p = _normalize_params_for_model(name, params)

    if name == 'lightgbm':
        if class_weight is not None:
            p['class_weight'] = class_weight
        return lgb.LGBMClassifier(**p)
    if name == 'randomforest':
        if class_weight is not None:
            p['class_weight'] = class_weight
        return RandomForestClassifier(**p)
    if name == 'extratrees':
        if class_weight is not None:
            p['class_weight'] = class_weight
        return ExtraTreesClassifier(**p)
    if name == 'xgboost':
        if XGBClassifier is None:
            raise ImportError("XGBoost belum terinstall.")
        # scale_pos_weight di-set di luar jika perlu
        return XGBClassifier(**p)
    if name == 'catboost':
        if CatBoostClassifier is None:
            raise ImportError("CatBoost belum terinstall.")
        if class_weight is not None:
            # CatBoost: class_weights (dict) bisa langsung dipakai
            p['class_weights'] = class_weight
        return CatBoostClassifier(**p)
    raise ValueError(f"Model tidak dikenal: {name}")


def fit_binary(model, X, y, sample_weight=None):
    """Fit dengan optional sample_weight; fallback bila model tak support."""
    try:
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
    except TypeError:
        model.fit(X, y)
    return model


def extract_feature_importance(model, features):
    fi = getattr(model, 'feature_importances_', None)
    if fi is None and hasattr(model, 'get_feature_importance'):
        fi = model.get_feature_importance()
    if fi is None:
        return {f: 0.0 for f in features}
    fi = np.asarray(fi, dtype=float)
    s = float(fi.sum())
    pct = (fi / s * 100.0) if s > 0 else np.zeros_like(fi)
    return dict(sorted(zip(features, pct.tolist()), key=lambda x: -x[1]))


def compute_classification_metrics(yt, yp, labels) -> dict:
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    res = {'N': int(len(yt)), 'accuracy': float(accuracy_score(yt, yp))}
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels,
                                                      zero_division=0)
    res['per_class'] = [{
        'class': lab,
        'precision': round(float(pr[i]), 4),
        'recall': round(float(rc[i]), 4),
        'f1': round(float(f1[i]), 4),
        'support': int(sup[i]),
    } for i, lab in enumerate(labels)]
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average='weighted', zero_division=0)
    res['weighted'] = {'precision': round(float(pr_w), 4),
                       'recall': round(float(rc_w), 4),
                       'f1': round(float(f1_w), 4)}
    res['confusion_matrix'] = confusion_matrix(yt, yp, labels=labels).tolist()
    res['labels'] = list(labels)
    return res


# ══════════════════════════════════════════════════════════════════
# TRAINING — STAGE-1 (HC vs W)
# ══════════════════════════════════════════════════════════════════
def train_stage1(combined: pd.DataFrame, test_wells: list, features: list,
                 model_name: str, model_params: dict,
                 balance_strategy: str = 'none',
                 use_class_weight: bool = True) -> dict:
    """Latih classifier biner HC vs W dengan well-level test split."""
    if combined is None or combined.empty:
        st.error("❌ Stage-1: data kosong.")
        return None
    feats = [f for f in features if f in combined.columns]
    if not feats:
        st.error("❌ Stage-1: tidak ada feature valid.")
        return None
    if not test_wells:
        st.error("❌ Stage-1: pilih minimal 1 sumur sebagai Test Set.")
        return None

    df = combined.copy()
    mask_test = df['WELL_NAME'].isin(test_wells)
    df_tr = df[~mask_test].dropna(subset=feats + ['FLUID_HC_W']).copy()
    df_te = df[mask_test].copy()
    if df_tr.empty:
        st.error("❌ Stage-1: data train kosong.")
        return None
    if df_tr['FLUID_HC_W'].nunique() < 2:
        st.error("❌ Stage-1: train hanya 1 kelas (HC atau W).")
        return None

    df_tr_bal = balance_dataframe(
        df_tr, 'FLUID_HC_W', strategy=balance_strategy)
    X = df_tr_bal[feats]
    y = df_tr_bal['FLUID_HC_W']
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    cw = compute_class_weights(y) if use_class_weight else None
    cw_enc = ({int(le.transform([k])[0]): v for k, v in cw.items()}
              if cw else None)

    if model_name.lower() == 'xgboost' and use_class_weight:
        n_pos = int((y_enc == 1).sum())
        n_neg = int((y_enc == 0).sum())
        scale = (n_neg / n_pos) if n_pos > 0 else 1.0
        params = {**model_params, 'scale_pos_weight': scale}
        model = build_classifier(model_name, params, class_weight=None)
    else:
        model = build_classifier(model_name, model_params, class_weight=cw_enc)

    fit_binary(model, X, y_enc)

    # Predict on test
    df_te['HC_W_PRED'] = pd.NA
    df_te['HC_W_PROBA_HC'] = np.nan
    m_te = df_te[feats].notna().all(axis=1)
    if m_te.any():
        Xt = df_te.loc[m_te, feats]
        df_te.loc[m_te, 'HC_W_PRED'] = le.inverse_transform(model.predict(Xt))
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xt)
            hc_idx = list(le.classes_).index(
                'HC') if 'HC' in le.classes_ else 1
            df_te.loc[m_te, 'HC_W_PROBA_HC'] = proba[:, hc_idx]

    # Metrics
    metrics = None
    s1m = df_te['FLUID_HC_W'].notna() & df_te['HC_W_PRED'].notna()
    if s1m.any():
        yt = df_te.loc[s1m, 'FLUID_HC_W'].values
        yp = df_te.loc[s1m, 'HC_W_PRED'].values
        metrics = compute_classification_metrics(yt, yp, labels=['HC', 'W'])
        if df_te.loc[s1m, 'HC_W_PROBA_HC'].notna().any():
            yt_bin = (yt == 'HC').astype(int)
            yprob = df_te.loc[s1m, 'HC_W_PROBA_HC'].values
            ok = ~np.isnan(yprob)
            if ok.sum() > 5 and len(np.unique(yt_bin[ok])) == 2:
                metrics['roc_auc'] = float(
                    roc_auc_score(yt_bin[ok], yprob[ok]))
                fpr, tpr, _ = roc_curve(yt_bin[ok], yprob[ok])
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    return {
        'model': model, 'le': le, 'features': feats,
        'df_te': df_te, 'metrics': metrics,
        'feat_imp': extract_feature_importance(model, feats),
        'audit': {
            'n_train_raw': int(len(df_tr)),
            'n_train_balanced': int(len(df_tr_bal)),
            'n_HC': int((df_tr_bal['FLUID_HC_W'] == 'HC').sum()),
            'n_W': int((df_tr_bal['FLUID_HC_W'] == 'W').sum()),
            'class_weight': cw,
        },
        'model_name': model_name,
        'balance_strategy': balance_strategy,
    }


# ══════════════════════════════════════════════════════════════════
# TRAINING — STAGE-2 (O vs G, subset HC)
# ══════════════════════════════════════════════════════════════════
def train_stage2(combined: pd.DataFrame, test_wells: list, features: list,
                 model_name: str, model_params: dict,
                 balance_strategy: str = 'none',
                 use_class_weight: bool = True) -> dict:
    """Latih classifier biner O vs G HANYA pada baris HC (label aktual)."""
    if combined is None or combined.empty:
        st.error("❌ Stage-2: data kosong.")
        return None
    feats = [f for f in features if f in combined.columns]
    if not feats:
        st.error("❌ Stage-2: tidak ada feature valid.")
        return None
    if not test_wells:
        st.error("❌ Stage-2: pilih minimal 1 sumur sebagai Test Set.")
        return None

    df = combined.copy()
    mask_test = df['WELL_NAME'].isin(test_wells)
    df_tr_full = df[~mask_test]
    df_tr = df_tr_full[df_tr_full['FLUID_HC_W'] == 'HC']
    df_tr = df_tr.dropna(subset=feats + ['FLUID_OG']).copy()
    df_te = df[mask_test].copy()

    if len(df_tr) < 20:
        st.error(f"❌ Stage-2: data HC train hanya {len(df_tr)} baris (<20).")
        return None
    if df_tr['FLUID_OG'].nunique() < 2:
        st.error("❌ Stage-2: train HC hanya 1 kelas (O atau G).")
        return None

    df_tr_bal = balance_dataframe(df_tr, 'FLUID_OG', strategy=balance_strategy)
    X = df_tr_bal[feats]
    y = df_tr_bal['FLUID_OG']
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    cw = compute_class_weights(y) if use_class_weight else None
    cw_enc = ({int(le.transform([k])[0]): v for k, v in cw.items()}
              if cw else None)

    if model_name.lower() == 'xgboost' and use_class_weight:
        n_pos = int((y_enc == 1).sum())
        n_neg = int((y_enc == 0).sum())
        scale = (n_neg / n_pos) if n_pos > 0 else 1.0
        params = {**model_params, 'scale_pos_weight': scale}
        model = build_classifier(model_name, params, class_weight=None)
    else:
        model = build_classifier(model_name, model_params, class_weight=cw_enc)

    fit_binary(model, X, y_enc)

    # Predict on test (semua row, tidak filter HC dulu — biar bisa pisah evaluasi)
    df_te['OG_PRED'] = pd.NA
    df_te['OG_PROBA_O'] = np.nan
    m_te = df_te[feats].notna().all(axis=1)
    if m_te.any():
        Xt = df_te.loc[m_te, feats]
        df_te.loc[m_te, 'OG_PRED'] = le.inverse_transform(model.predict(Xt))
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xt)
            o_idx = list(le.classes_).index('O') if 'O' in le.classes_ else 1
            df_te.loc[m_te, 'OG_PROBA_O'] = proba[:, o_idx]

    # Metrics: hanya pada baris yang label aktual = HC (FLUID_OG ada)
    metrics = None
    s2m = df_te['FLUID_OG'].notna() & df_te['OG_PRED'].notna()
    if s2m.any():
        yt = df_te.loc[s2m, 'FLUID_OG'].values
        yp = df_te.loc[s2m, 'OG_PRED'].values
        metrics = compute_classification_metrics(yt, yp, labels=['O', 'G'])
        if df_te.loc[s2m, 'OG_PROBA_O'].notna().any():
            yt_bin = (yt == 'O').astype(int)
            yprob = df_te.loc[s2m, 'OG_PROBA_O'].values
            ok = ~np.isnan(yprob)
            if ok.sum() > 5 and len(np.unique(yt_bin[ok])) == 2:
                metrics['roc_auc'] = float(
                    roc_auc_score(yt_bin[ok], yprob[ok]))
                fpr, tpr, _ = roc_curve(yt_bin[ok], yprob[ok])
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    return {
        'model': model, 'le': le, 'features': feats,
        'df_te': df_te, 'metrics': metrics,
        'feat_imp': extract_feature_importance(model, feats),
        'audit': {
            'n_train_raw': int(len(df_tr)),
            'n_train_balanced': int(len(df_tr_bal)),
            'n_O': int((df_tr_bal['FLUID_OG'] == 'O').sum()),
            'n_G': int((df_tr_bal['FLUID_OG'] == 'G').sum()),
            'class_weight': cw,
        },
        'model_name': model_name,
        'balance_strategy': balance_strategy,
    }


# ══════════════════════════════════════════════════════════════════
# END-TO-END PREDICTION (chaining Stage-1 → Stage-2)
# ══════════════════════════════════════════════════════════════════
def build_end_to_end(combined, test_wells, s1_res, s2_res):
    """
    Gabungkan prediksi Stage-1 + Stage-2:
      - HC_W_PRED = output Stage-1
      - OG_PRED   = output Stage-2 hanya pada baris HC_W_PRED == 'HC'
      - FLUID_PRED_3 = O / G / W
    Return (df_pred, metrics_endtoend)
    """
    if s1_res is None:
        return None, None
    df = combined[combined['WELL_NAME'].isin(
        test_wells)].copy().reset_index(drop=True)

    feats1 = s1_res['features']
    m1_model = s1_res['model']
    m1_le = s1_res['le']

    df['HC_W_PRED'] = pd.NA
    df['HC_W_PROBA_HC'] = np.nan
    df['OG_PRED'] = pd.NA
    df['OG_PROBA_O'] = np.nan
    df['FLUID_PRED_3'] = pd.NA

    m1 = df[feats1].notna().all(axis=1)
    if m1.any():
        Xt = df.loc[m1, feats1]
        df.loc[m1, 'HC_W_PRED'] = m1_le.inverse_transform(m1_model.predict(Xt))
        if hasattr(m1_model, 'predict_proba'):
            proba = m1_model.predict_proba(Xt)
            hc_idx = list(m1_le.classes_).index(
                'HC') if 'HC' in m1_le.classes_ else 1
            df.loc[m1, 'HC_W_PROBA_HC'] = proba[:, hc_idx]

    if s2_res is not None:
        feats2 = s2_res['features']
        m2_model = s2_res['model']
        m2_le = s2_res['le']
        m2 = (df['HC_W_PRED'] == 'HC') & df[feats2].notna().all(axis=1)
        if m2.any():
            Xt = df.loc[m2, feats2]
            df.loc[m2, 'OG_PRED'] = m2_le.inverse_transform(
                m2_model.predict(Xt))
            if hasattr(m2_model, 'predict_proba'):
                proba = m2_model.predict_proba(Xt)
                o_idx = list(m2_le.classes_).index(
                    'O') if 'O' in m2_le.classes_ else 1
                df.loc[m2, 'OG_PROBA_O'] = proba[:, o_idx]

    df.loc[df['HC_W_PRED'] == 'W', 'FLUID_PRED_3'] = 'W'
    df.loc[df['OG_PRED'] == 'O', 'FLUID_PRED_3'] = 'O'
    df.loc[df['OG_PRED'] == 'G', 'FLUID_PRED_3'] = 'G'
    df.loc[(df['HC_W_PRED'] == 'HC') & df['OG_PRED'].isna(),
           'FLUID_PRED_3'] = 'HC'

    e2e_m = None
    s3 = df['FLUID_CLASS3'].notna() & df['FLUID_PRED_3'].isin(['O', 'G', 'W'])
    if s3.any():
        yt = df.loc[s3, 'FLUID_CLASS3'].values
        yp = df.loc[s3, 'FLUID_PRED_3'].values
        e2e_m = compute_classification_metrics(yt, yp, labels=['O', 'G', 'W'])
    return df, e2e_m


# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════
def _track_litho(df, depth):
    if 'LITHO_CODE' not in df.columns:
        return None
    code = pd.to_numeric(df['LITHO_CODE'], errors='coerce')
    name = code.map(LITHO_CODE_TO_NAME)
    colors = name.map(LITHO_COLOR).fillna('#1c2128')
    return go.Bar(
        x=[1] * len(df), y=depth, orientation='h', width=1.0,
        marker=dict(color=colors.tolist(), line=dict(width=0)),
        showlegend=False, hovertext=name, hoverinfo='text+y', name='LITHO',
    )


def _track_fluid(values, depth, name='FLUID', color_map=None):
    color_map = color_map or FLUID_COLOR
    s = pd.Series(values).astype(object)
    colors = s.map(color_map).fillna('#1c2128')
    return go.Bar(
        x=[1] * len(s), y=depth, orientation='h', width=1.0,
        marker=dict(color=colors.tolist(), line=dict(width=0)),
        showlegend=False, hovertext=s, hoverinfo='text+y', name=name,
    )


def plot_well_log(df: pd.DataFrame, well_name: str,
                  show_pred: bool = False) -> go.Figure:
    """
    Composite log:
      Track 1: GR / GR_NORM
      Track 2: RHOB & NPHI
      Track 3: RT (log)
      Track 4: LITHO_CODE
      Track 5: FLUID Aktual
      Track 6: FLUID Prediksi (bila show_pred)
    """
    df = df.sort_values('DEPTH').reset_index(drop=True)
    d = df['DEPTH'].values

    n_tracks = 5 + (1 if show_pred else 0)
    titles = ['GR / GR_NORM', 'RHOB / NPHI',
              'RT (log)', 'LITHO', 'FLUID Aktual']
    if show_pred:
        titles.append('FLUID Prediksi')
    widths = [1.2, 1.2, 1.0, 0.4, 0.4] + ([0.4] if show_pred else [])

    fig = make_subplots(rows=1, cols=n_tracks, shared_yaxes=True,
                        column_widths=widths, subplot_titles=titles,
                        horizontal_spacing=0.012)

    if 'GR' in df.columns:
        fig.add_trace(go.Scatter(x=df['GR'], y=d, mode='lines', name='GR',
                                 line=dict(color='#3fb950', width=1.2)),
                      row=1, col=1)
    if 'GR_NORM' in df.columns and df['GR_NORM'].notna().any():
        fig.add_trace(go.Scatter(x=df['GR_NORM'], y=d, mode='lines', name='GR_NORM',
                                 line=dict(color='#00d26a', width=1.2, dash='dot')),
                      row=1, col=1)

    if 'RHOB' in df.columns:
        fig.add_trace(go.Scatter(x=df['RHOB'], y=d, mode='lines', name='RHOB',
                                 line=dict(color='#f85149', width=1.2)),
                      row=1, col=2)
    if 'NPHI' in df.columns:
        fig.add_trace(go.Scatter(x=df['NPHI'], y=d, mode='lines', name='NPHI',
                                 line=dict(color='#58a6ff', width=1.2, dash='dot')),
                      row=1, col=2)

    if 'RT' in df.columns:
        fig.add_trace(go.Scatter(x=df['RT'].clip(lower=0.001), y=d, mode='lines',
                                 name='RT', line=dict(color='#bc8cff', width=1.2)),
                      row=1, col=3)
        fig.update_xaxes(type='log', row=1, col=3)

    tr_litho = _track_litho(df, d)
    if tr_litho is not None:
        fig.add_trace(tr_litho, row=1, col=4)
    fig.update_xaxes(showticklabels=False, range=[0, 1], row=1, col=4)

    if 'FLUID_LETTER' in df.columns:
        fig.add_trace(_track_fluid(df['FLUID_LETTER'].fillna(''), d,
                                   name='FLUID Aktual'), row=1, col=5)
    elif 'FLUID_CLASS3' in df.columns:
        fig.add_trace(_track_fluid(df['FLUID_CLASS3'].fillna(''), d,
                                   name='FLUID Aktual'), row=1, col=5)
    fig.update_xaxes(showticklabels=False, range=[0, 1], row=1, col=5)

    if show_pred and 'FLUID_PRED_3' in df.columns:
        fig.add_trace(_track_fluid(df['FLUID_PRED_3'].fillna(''), d,
                                   name='FLUID Prediksi'), row=1, col=6)
        fig.update_xaxes(showticklabels=False, range=[0, 1], row=1, col=6)

    fig.update_layout(**PLOTLY_BASE, height=820,
                      title=dict(text=f'<b>{well_name}</b>',
                                 font=dict(size=13, color='#58a6ff'), x=0.01),
                      bargap=0,
                      legend=dict(bgcolor='rgba(13,17,23,0.85)',
                                  bordercolor='#30363d', borderwidth=1,
                                  font=dict(size=9), orientation='h',
                                  yanchor='bottom', y=1.03,
                                  xanchor='left', x=0))
    fig.update_yaxes(autorange='reversed', gridcolor='#21262d',
                     zerolinecolor='#30363d', tickfont_size=9)
    fig.update_xaxes(gridcolor='#21262d',
                     zerolinecolor='#30363d', tickfont_size=9)
    return fig


def plot_class_distribution(df: pd.DataFrame, label_col: str, title: str) -> go.Figure:
    counts = df[label_col].value_counts(dropna=False).reset_index()
    counts.columns = ['class', 'n']
    counts['pct'] = (counts['n'] / counts['n'].sum() * 100).round(2)
    fig = go.Figure(go.Bar(
        x=counts['class'].astype(str), y=counts['n'],
        text=[f"{n:,}<br>({p}%)" for n, p in zip(counts['n'], counts['pct'])],
        textposition='outside',
        marker_color=[FLUID_COLOR.get(str(c), '#58a6ff')
                      for c in counts['class']],
    ))
    fig.update_layout(**PLOTLY_BASE, title=title, height=320,
                      xaxis_title='Class', yaxis_title='Count', showlegend=False)
    return fig


def plot_confusion_matrix(cm, labels, title='Confusion Matrix') -> go.Figure:
    cm = np.asarray(cm)
    z_text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, colorscale='Blues', showscale=True,
        text=z_text, texttemplate='%{text}',
        textfont=dict(size=14, color='#0d1117'),
        hovertemplate='Actual=%{y}<br>Pred=%{x}<br>N=%{z}<extra></extra>'))
    fig.update_layout(**PLOTLY_BASE, title=title, height=380,
                      xaxis_title='Predicted', yaxis_title='Actual')
    fig.update_yaxes(autorange='reversed')
    return fig


def plot_feature_importance(fi: dict, title: str) -> go.Figure:
    if not fi:
        return None
    items = list(fi.items())[::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h',
                           marker_color='#58a6ff',
                           text=[f"{v:.1f}%" for v in vals],
                           textposition='outside'))
    fig.update_layout(**PLOTLY_BASE, title=title,
                      height=max(280, 24 * len(items)),
                      xaxis_title='Importance (%)', showlegend=False)
    return fig


def plot_roc(fpr, tpr, auc, title='ROC') -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'AUC = {auc:.3f}',
                             line=dict(color='#58a6ff', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                             line=dict(color='#7d8590', width=1, dash='dash')))
    fig.update_layout(**PLOTLY_BASE, title=title, height=340,
                      xaxis_title='FPR', yaxis_title='TPR')
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR HELPER — render hyperparam UI per-stage
# ══════════════════════════════════════════════════════════════════
def render_model_block(stage_key: str, default_model='CatBoost'):
    """
    Render UI: pilih algoritma + hyperparams + balance strategy + class_weight.
    stage_key: 's1' atau 's2'  → dipakai sebagai key Streamlit unik.
    Return dict cfg.
    """
    model_opts = ['LightGBM', 'RandomForest', 'ExtraTrees']
    if XGBClassifier is not None:
        model_opts.append('XGBoost')
    if CatBoostClassifier is not None:
        model_opts.append('CatBoost')

    model_choice = st.selectbox(
        f"Algoritma ({stage_key.upper()})",
        options=model_opts,
        index=model_opts.index(
            default_model) if default_model in model_opts else 0,
        key=f'{stage_key}_model')

    name_lower = model_choice.lower()
    has_lr = name_lower in ('lightgbm', 'xgboost', 'catboost')

    n_estimators = st.number_input(
        f"n_estimators / iterations", 50, 3000, 400, 50,
        key=f'{stage_key}_n_est',
        help="LightGBM/RF/XGB pakai n_estimators · CatBoost pakai iterations (auto-translate).")

    learning_rate = None
    if has_lr:
        learning_rate = st.number_input(
            f"learning_rate", 0.005, 0.5, 0.05, 0.005,
            key=f'{stage_key}_lr',
            help="Hanya untuk boosting model (LightGBM/XGBoost/CatBoost).")

    max_depth = st.number_input(
        f"max_depth (-1 = auto)", -1, 32, -1, 1,
        key=f'{stage_key}_md',
        help="CatBoost: dipetakan ke 'depth'. Bagging (RF/ET) ignore -1 = no limit.")

    balance_strategy = st.selectbox(
        f"Resampling",
        options=['none', 'undersample', 'oversample'],
        index=0, key=f'{stage_key}_bal',
        help="undersample: kurangi mayoritas · oversample: replikasi minoritas")

    use_class_weight = st.checkbox(
        f"class_weight='balanced'", value=True, key=f'{stage_key}_cw')

    raw_params = {'n_estimators': int(n_estimators)}
    if learning_rate is not None:
        raw_params['learning_rate'] = float(learning_rate)
    if max_depth != -1:
        raw_params['max_depth'] = int(max_depth)

    return {
        'model': model_choice,
        'params': raw_params,
        'balance': balance_strategy,
        'use_class_weight': use_class_weight,
    }


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — Workflow
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo">💧 Fluid·ML</div>'
                '<div class="logo-sub">two-stage fluid classification</div>',
                unsafe_allow_html=True)

    # 01 MULTI-STRUKTUR INPUT (ZIP LAS + Zone CSV per struktur)
    st.markdown('<div class="sec">01 · Multi-Struktur Input</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="ibox">Upload <b>satu atau lebih</b> struktur. '
        'Tiap struktur = satu ZIP LAS + opsional Zone CSV + nama STRUKTUR. '
        'Nama STRUKTUR dijadikan acuan rumus DNS_CUTOFF.</div>',
        unsafe_allow_html=True)

    n_struct = st.number_input(
        "Jumlah Struktur", min_value=1, max_value=20,
        value=max(1, st.session_state.get('n_structures', 1)),
        step=1, key='n_struct_inp')
    st.session_state['n_structures'] = int(n_struct)

    # Sinkronkan list structures dengan jumlah yang di-set
    structures = st.session_state.get('structures', []) or []
    while len(structures) < n_struct:
        structures.append({'name': '', 'wells': {}, 'zone_df': None,
                           'zip_hash': None, 'zone_hash': None})
    structures = structures[:n_struct]
    st.session_state['structures'] = structures

    available_strukturs = sorted(DNS_CUTOFF_EQUATIONS.keys())

    for i in range(n_struct):
        s = structures[i]
        hdr_name = s.get('name') or f'(belum diisi)'
        n_w = len(s.get('wells') or {})
        with st.expander(f"📦 Struktur #{i+1} — {hdr_name} · wells={n_w}",
                         expanded=(i == 0 and n_w == 0)):
            opts = ['(custom)'] + available_strukturs
            cur_name = s.get('name') or ''
            cur_idx = (opts.index(cur_name) if cur_name in available_strukturs
                       else 0)
            sel = st.selectbox(
                f"STRUKTUR #{i+1}", options=opts, index=cur_idx,
                key=f'struct_sel_{i}',
                help='Pilih dari daftar (rumus DNS_CUTOFF tersedia) '
                     'atau "(custom)" untuk nama bebas.')
            if sel == '(custom)':
                name = st.text_input(
                    f"Custom STRUKTUR #{i+1}",
                    value=(cur_name if cur_name not in available_strukturs
                           else ''),
                    key=f'struct_custom_{i}')
            else:
                name = sel
            s['name'] = (name or '').strip()

            if s['name'] and s['name'] not in DNS_CUTOFF_EQUATIONS:
                st.markdown(
                    f'<div class="ibox wbox">⚠ Tidak ada rumus DNS_CUTOFF '
                    f'untuk <b>{s["name"]}</b> · DNS_CUTOFF & FLUID_DNS '
                    f'akan kosong.</div>', unsafe_allow_html=True)

            zip_up = st.file_uploader(
                f"ZIP LAS #{i+1}", type=['zip'], key=f'zip_up_{i}')
            if zip_up is not None:
                zb = zip_up.read()
                zh = hashlib.md5(zb).hexdigest()
                if s.get('zip_hash') != zh:
                    with st.spinner(f"Membaca LAS struktur "
                                    f"{s['name'] or i+1}..."):
                        wells = load_zip_cached(zb)
                    if wells:
                        s['wells'] = wells
                        s['zip_hash'] = zh
                        st.session_state['combined_df'] = None
                        st.session_state['qc_log'] = None
                        st.success(f"✅ {len(wells)} sumur: "
                                   f"{', '.join(sorted(wells.keys()))}")
                    else:
                        st.error("Tidak ada LAS valid")
                else:
                    st.success(f"✅ {len(s['wells'])} sumur (cached)")

            zone_up = st.file_uploader(
                f"Zone CSV #{i+1} (opsional)", type=['csv'],
                key=f'zone_up_{i}',
                help="Kolom: WELL, MD, ZONE/Surface")
            if zone_up is not None:
                zb = zone_up.read()
                zh = hashlib.md5(zb).hexdigest()
                if s.get('zone_hash') != zh:
                    try:
                        zdf = read_zone_csv(io.BytesIO(zb))
                        s['zone_df'] = zdf
                        s['zone_hash'] = zh
                        st.session_state['combined_df'] = None
                        st.success(f"✅ Zone: {len(zdf):,} interval")
                    except Exception as e:
                        st.error(f"❌ Gagal baca Zone CSV: {e}")
                else:
                    if s.get('zone_df') is not None:
                        st.success(
                            f"✅ Zone: {len(s['zone_df']):,} interval (cached)")

            if st.button(f"🗑 Reset Struktur #{i+1}", key=f'reset_struct_{i}',
                         use_container_width=True):
                s['wells'] = {}
                s['zone_df'] = None
                s['zip_hash'] = None
                s['zone_hash'] = None
                st.session_state['combined_df'] = None
                st.session_state['qc_log'] = None

    st.session_state['structures'] = structures

    # ── Gabung semua struktur menjadi all_wells & zone_df global ───
    all_wells_combined = {}
    for s in structures:
        for wname, wdf in (s.get('wells') or {}).items():
            all_wells_combined[wname] = wdf
    st.session_state['all_wells'] = all_wells_combined

    all_wells = all_wells_combined
    if any((s.get('wells') or {}) for s in structures):
        # build_combined_multi sudah handle merge_zone per struktur + tag STRUKTUR
        combined_raw = build_combined_multi(structures)
        # Feature engineering (LOG_RT, LOG_RGSA, RGSAA/NGSAA/DGSAA, DNS/DNSV/DNS_CUTOFF/FLUID_DNS)
        combined_raw = compute_engineered_features(combined_raw)
    else:
        combined_raw = None

    # 03 QC
    st.markdown('<div class="sec">03 · QC Pipeline</div>',
                unsafe_allow_html=True)
    use_iqual_filter = st.checkbox("Filter IQUAL > 0", value=True,
                                   help="Baris tanpa IQUAL > 0 dianggap tidak punya label fluid.")
    qc_btn = st.button("🧹 Jalankan QC",
                       disabled=combined_raw is None,
                       use_container_width=True)
    if qc_btn and combined_raw is not None:
        df_qc, ql = run_qc_pipeline(combined_raw, use_iqual=use_iqual_filter)
        st.session_state['combined_df'] = df_qc
        st.session_state['qc_log'] = ql
        st.session_state['normalized'] = False
        st.session_state['gr_norm_params'] = {}
        st.success(f"✅ QC selesai — {len(df_qc):,} baris tersisa")

    if st.session_state.get('qc_log'):
        ql = st.session_state['qc_log']
        st.markdown(
            f'<div class="ibox">'
            f'Drop all-NaN logs : <b>{ql.get("drop_all_nan_logs",0):,}</b><br>'
            f'RT invalid → NaN  : <b>{ql.get("rt_invalid_to_nan",0):,}</b><br>'
            f'Drop IQUAL ≤ 0    : <b>{ql.get("drop_iqual_le_0",0):,}</b><br>'
            f'Drop FLUID NaN    : <b>{ql.get("drop_fluid_label_nan",0):,}</b><br>'
            f'Drop kode fluid asing: <b>{ql.get("drop_fluid_unknown_code",0):,}</b><br>'
            f'<b>Total drop: {ql.get("total_dropped",0):,} · Sisa: {ql.get("remaining",0):,}</b>'
            f'</div>', unsafe_allow_html=True)

    combined = st.session_state.get('combined_df')
    if combined is None or combined.empty:
        combined = combined_raw

    # 04 GR Normalization
    st.markdown('<div class="sec">04 · Normalisasi GR</div>',
                unsafe_allow_html=True)
    p_low = st.number_input("P low", 1.0, 40.0, 3.0, key='gr_p_low')
    p_high = st.number_input("P high", 60.0, 99.0, 97.0, key='gr_p_high')
    has_gr = (combined is not None and 'GR' in combined.columns
              and combined['GR'].notna().any())
    norm_btn = st.button("📐 Hitung GR_NORM",
                         disabled=not has_gr, use_container_width=True)
    if norm_btn and has_gr:
        params = compute_gr_norm_params(combined, p_low=p_low, p_high=p_high)
        combined = apply_gr_norm(combined, params)
        st.session_state['combined_df'] = combined
        st.session_state['gr_norm_params'] = params
        st.session_state['normalized'] = True
        st.success(f"✅ GR_NORM — {combined['GR_NORM'].notna().sum():,} valid")

    # 05 Test wells (shared by both stages)
    st.markdown('<div class="sec">05 · Test Wells</div>',
                unsafe_allow_html=True)
    well_list = sorted(all_wells.keys()) if all_wells else []
    test_wells = st.multiselect("Pilih sumur Test Set", options=well_list,
                                help="Sisanya jadi Training. Berlaku untuk Stage-1 & Stage-2.",
                                key='test_wells_select')
    st.session_state['fc_test_wells'] = test_wells

    # ── Available features
    if combined is not None and not combined.empty:
        avail_feats = [c for c in ALL_FEATURE_LOGS
                       if c in combined.columns and combined[c].notna().any()]
    else:
        avail_feats = []

    # ─────────── STAGE-1 BLOCK ───────────
    st.markdown('<div class="sec">06 · STAGE-1 · HC vs W</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="stage-hdr">🟦 Stage-1 Configuration</div>',
                unsafe_allow_html=True)
    default_s1_feats = [c for c in ['GR', 'LOG_RT', 'NPHI', 'RHOB', 'VSH',
                                    'PHIE', 'SW', 'LITHO_CODE', 'RPBE', 'RGBE', 'SWGRAD', 'LOG_RGSA', 'RGSAA']
                        if c in avail_feats]
    s1_feats = st.multiselect(
        "Features Stage-1 (HC vs W)",
        options=avail_feats, default=default_s1_feats, key='s1_feats',
        help="Log predictor untuk membedakan Hidrokarbon vs Water.")
    if s1_feats:
        st.markdown('<div class="ibox">' +
                    ''.join(f'<span class="pill">{f}</span>' for f in s1_feats) +
                    '</div>', unsafe_allow_html=True)

    s1_cfg = render_model_block('s1', default_model='LightGBM')

    can_train_s1 = (combined is not None and not combined.empty
                    and len(test_wells) > 0 and len(s1_feats) > 0)
    s1_btn = st.button("🚀 Train Stage-1 (HC vs W)",
                       disabled=not can_train_s1, use_container_width=True,
                       key='train_s1_btn')
    if s1_btn and can_train_s1:
        with st.spinner("Training Stage-1..."):
            res1 = train_stage1(
                combined, test_wells, s1_feats,
                s1_cfg['model'], s1_cfg['params'],
                balance_strategy=s1_cfg['balance'],
                use_class_weight=s1_cfg['use_class_weight'])
        if res1 is not None:
            st.session_state['fc_s1_results'] = res1
            st.session_state['fc_s1_cfg'] = {**s1_cfg, 'features': s1_feats,
                                             'test_wells': test_wells}
            st.success("✅ Stage-1 selesai")

    # ─────────── STAGE-2 BLOCK ───────────
    st.markdown('<div class="sec">07 · STAGE-2 · O vs G</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="stage-hdr">🟩 Stage-2 Configuration</div>',
                unsafe_allow_html=True)
    default_s2_feats = [c for c in ['GR', 'LOG_RT', 'NPHI', 'RHOB',
                                    'PHIE', 'SW', 'VSH', 'LITHO_CODE',
                                    'NGSA', 'DGSA', 'NGSAA', 'DGSAA',
                                    'DNS', 'DNS_CUTOFF',
                                    'DNBE', 'SPBE']
                        if c in avail_feats]
    s2_feats = st.multiselect(
        "Features Stage-2 (O vs G)",
        options=avail_feats, default=default_s2_feats, key='s2_feats',
        help="Log predictor untuk membedakan Oil vs Gas (subset HC).")
    if s2_feats:
        st.markdown('<div class="ibox">' +
                    ''.join(f'<span class="pill">{f}</span>' for f in s2_feats) +
                    '</div>', unsafe_allow_html=True)

    s2_cfg = render_model_block('s2', default_model='LightGBM')

    can_train_s2 = (combined is not None and not combined.empty
                    and len(test_wells) > 0 and len(s2_feats) > 0)
    s2_btn = st.button("🚀 Train Stage-2 (O vs G)",
                       disabled=not can_train_s2, use_container_width=True,
                       key='train_s2_btn')
    if s2_btn and can_train_s2:
        with st.spinner("Training Stage-2..."):
            res2 = train_stage2(
                combined, test_wells, s2_feats,
                s2_cfg['model'], s2_cfg['params'],
                balance_strategy=s2_cfg['balance'],
                use_class_weight=s2_cfg['use_class_weight'])
        if res2 is not None:
            st.session_state['fc_s2_results'] = res2
            st.session_state['fc_s2_cfg'] = {**s2_cfg, 'features': s2_feats,
                                             'test_wells': test_wells}
            st.success("✅ Stage-2 selesai")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
st.markdown("## 💧 Two-Stage Fluid Classification")
st.markdown(
    '<div class="ibox">'
    '<b>Pipeline:</b> Stage-1 (HC vs W) &nbsp;·&nbsp; '
    'Stage-2 (O vs G, subset HC). '
    'OP→O · GP→G. Filter IQUAL > 0.<br>'
    '<b>Train per-stage</b> dengan log predictor & model masing-masing dari sidebar.'
    '</div>', unsafe_allow_html=True)

combined = st.session_state.get('combined_df')
if combined is None or combined.empty:
    combined = combined_raw

n_wells = combined['WELL_NAME'].nunique(
) if combined is not None and not combined.empty else 0
n_rows = len(combined) if combined is not None else 0
n_label = (combined['FLUID_CODE'].notna().sum()
           if combined is not None and 'FLUID_CODE' in combined.columns else 0)
n_iqual = (int((pd.to_numeric(combined.get('IQUAL'), errors='coerce') > 0).sum())
           if combined is not None and 'IQUAL' in combined.columns else 0)

s1_status = '✅' if st.session_state.get('fc_s1_results') else '—'
s2_status = '✅' if st.session_state.get('fc_s2_results') else '—'
st.markdown(
    '<div class="kpi-row">'
    f'<div class="kpi"><div class="kl">Wells</div><div class="kv">{n_wells}</div></div>'
    f'<div class="kpi"><div class="kl">Rows</div><div class="kv">{n_rows:,}</div></div>'
    f'<div class="kpi"><div class="kl">Labeled</div><div class="kv">{n_label:,}</div></div>'
    f'<div class="kpi"><div class="kl">IQUAL > 0</div><div class="kv">{n_iqual:,}</div></div>'
    f'<div class="kpi"><div class="kl">Stage-1</div><div class="kv">{s1_status}</div></div>'
    f'<div class="kpi"><div class="kl">Stage-2</div><div class="kv">{s2_status}</div></div>'
    '</div>', unsafe_allow_html=True)


tabs = st.tabs([
    "📊 Data Overview",
    "🧪 QC & Class Balance",
    "🟦 Stage-1 Result",
    "🟩 Stage-2 Result",
    "🎯 End-to-End",
    "🪵 Well Log Viewer",
    "💾 Export",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — Data Overview
# ──────────────────────────────────────────────────────────────────
with tabs[0]:
    if combined is None or combined.empty:
        st.info("Upload ZIP LAS dulu, lalu jalankan QC.")
    else:
        st.markdown("### Data per sumur")
        rows = []
        for w, grp in combined.groupby('WELL_NAME'):
            struktur = (grp['STRUKTUR'].dropna().iloc[0]
                        if 'STRUKTUR' in grp.columns and grp['STRUKTUR'].notna().any()
                        else '—')
            row = {
                'Well': w,
                'STRUKTUR': struktur,
                'Rows': len(grp),
                'Depth Range': f"{grp['DEPTH'].min():.1f} – {grp['DEPTH'].max():.1f}",
                'IQUAL>0': int((pd.to_numeric(grp.get('IQUAL'), errors='coerce') > 0).sum())
                if 'IQUAL' in grp.columns else 0,
                'FLUID labeled': int(grp.get('FLUID_CODE', pd.Series()).notna().sum()),
                'DNS_CUTOFF n': int(grp.get('DNS_CUTOFF', pd.Series()).notna().sum())
                if 'DNS_CUTOFF' in grp.columns else 0,
                'FLUID_DNS=G': int((grp.get('FLUID_DNS', pd.Series()) == 'G').sum()),
                'FLUID_DNS=O': int((grp.get('FLUID_DNS', pd.Series()) == 'O').sum()),
            }
            for cls in ['O', 'G', 'W', 'OP', 'GP']:
                row[cls] = int(
                    (grp.get('FLUID_LETTER', pd.Series()) == cls).sum())
            rows.append(row)
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

        # Ringkasan per STRUKTUR
        if 'STRUKTUR' in combined.columns:
            st.markdown("### Ringkasan per STRUKTUR")
            srows = []
            for sname, sgrp in combined.groupby('STRUKTUR'):
                has_eq = sname in DNS_CUTOFF_EQUATIONS
                srows.append({
                    'STRUKTUR': sname,
                    'Wells': sgrp['WELL_NAME'].nunique(),
                    'Rows': len(sgrp),
                    'Zones': sgrp['ZONE'].nunique() if 'ZONE' in sgrp.columns else 0,
                    'DNS_CUTOFF eq?': '✅' if has_eq else '⚠ none',
                    'DNS_CUTOFF n': int(sgrp.get('DNS_CUTOFF', pd.Series()).notna().sum())
                    if 'DNS_CUTOFF' in sgrp.columns else 0,
                })
            st.dataframe(pd.DataFrame(srows),
                         use_container_width=True, hide_index=True)

        st.markdown("### Log availability per sumur")
        avail_rows = []
        for w, grp in combined.groupby('WELL_NAME'):
            r = {'Well': w}
            for c in ALL_FEATURE_LOGS:
                if c in grp.columns:
                    n = int(grp[c].notna().sum())
                    r[c] = f"{n:,}" if n > 0 else "—"
                else:
                    r[c] = "—"
            avail_rows.append(r)
        st.dataframe(pd.DataFrame(avail_rows),
                     use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — QC & Class Balance
# ──────────────────────────────────────────────────────────────────
with tabs[1]:
    if combined is None or 'FLUID_CLASS3' not in combined.columns:
        st.info(
            "Jalankan QC dulu agar label fluid terbentuk (FLUID_CLASS3 / HC_W / OG).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_letter = plot_class_distribution(
                combined.assign(
                    FLUID_LETTER=combined['FLUID_LETTER'].fillna('NA')),
                'FLUID_LETTER',
                'Distribusi Label Asli (O/G/W/OP/GP)')
            st.plotly_chart(fig_letter, use_container_width=True)
        with c2:
            fig_hcw = plot_class_distribution(combined, 'FLUID_HC_W',
                                              'Stage-1: HC vs W')
            st.plotly_chart(fig_hcw, use_container_width=True)
        with c3:
            df_hc = combined[combined['FLUID_HC_W'] == 'HC']
            if df_hc.empty:
                st.info("Tidak ada baris HC untuk Stage-2.")
            else:
                fig_og = plot_class_distribution(df_hc, 'FLUID_OG',
                                                 'Stage-2: O vs G (subset HC)')
                st.plotly_chart(fig_og, use_container_width=True)

        st.markdown("### Imbalance Ratio (max / min)")
        for col, lbl in [('FLUID_HC_W', 'Stage-1 (HC/W)'),
                         ('FLUID_OG', 'Stage-2 (O/G — subset HC)')]:
            sub = combined if col != 'FLUID_OG' else combined[combined['FLUID_HC_W'] == 'HC']
            if sub.empty or sub[col].dropna().empty:
                continue
            counts = sub[col].value_counts()
            ratio = counts.max() / max(counts.min(), 1)
            color = 'good' if ratio < 2 else ('ok' if ratio < 5 else 'bad')
            st.markdown(
                f'<div class="ibox">{lbl}: '
                f'{dict(counts)} → ratio = '
                f'<span class="{color}"><b>{ratio:.1f}×</b></span></div>',
                unsafe_allow_html=True)

        with st.expander("📋 Mapping legend", expanded=False):
            st.markdown(
                "**FLUID_CODE → Class3:**  \n"
                "&nbsp;&nbsp;7=Gas (G) · 8=GP→G · 9=Oil (O) · 10=OP→O · 11=Water (W)\n\n"
                "**LITHO_CODE:**  \n"
                "&nbsp;&nbsp;1=Sand (kuning) · 4=Coal (hitam) · 6=Shale (abu)\n\n"
                "**Warna fluid track:** O=hijau · G=merah · W=biru · OP=cyan · GP=pink")


# ──────────────────────────────────────────────────────────────────
# Helper: render single-stage result block
# ──────────────────────────────────────────────────────────────────
def render_stage_result(res, cfg, stage_name, labels, proba_col):
    if res is None:
        st.info(
            f"Belum ada model {stage_name}. Klik **Train {stage_name}** di sidebar.")
        return
    st.markdown(
        f'<div class="ibox">'
        f'Model: <b>{cfg.get("model")}</b> · '
        f'Balance: <b>{cfg.get("balance")}</b> · '
        f'class_weight: <b>{cfg.get("use_class_weight")}</b> · '
        f'Test wells: <b>{", ".join(cfg.get("test_wells", []))}</b><br>'
        f'Features: ' +
        ''.join(f'<span class="pill">{f}</span>' for f in res['features']) +
        f'</div>', unsafe_allow_html=True)

    m = res['metrics']
    if m is None:
        st.warning(
            f"Tidak ada metric — test set tidak punya label {stage_name}.")
    else:
        cA, cB, cC, cD = st.columns(4)
        cA.metric("N", f"{m['N']:,}")
        cB.metric("Accuracy", f"{m['accuracy']:.3f}")
        cC.metric("F1 (weighted)", f"{m['weighted']['f1']:.3f}")
        cD.metric("ROC-AUC",
                  f"{m['roc_auc']:.3f}" if 'roc_auc' in m else "—")

        cL, cR = st.columns([1, 1])
        with cL:
            st.plotly_chart(plot_confusion_matrix(
                m['confusion_matrix'], m['labels'],
                title=f'{stage_name} — Confusion Matrix'),
                use_container_width=True)
            st.dataframe(pd.DataFrame(m['per_class']),
                         use_container_width=True, hide_index=True)
        with cR:
            if 'roc_curve' in m:
                rc = m['roc_curve']
                st.plotly_chart(plot_roc(rc['fpr'], rc['tpr'], m['roc_auc'],
                                         f'{stage_name} ROC'),
                                use_container_width=True)
            fi = res.get('feat_imp', {})
            if fi:
                st.plotly_chart(plot_feature_importance(
                    fi, f'{stage_name} Feature Importance'),
                    use_container_width=True)

    with st.expander(f"🔍 {stage_name} training audit", expanded=False):
        st.json(res['audit'])


# ──────────────────────────────────────────────────────────────────
# TAB 3 — Stage-1 Result
# ──────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## 🟦 Stage-1 · HC vs W")
    render_stage_result(
        st.session_state.get('fc_s1_results'),
        st.session_state.get('fc_s1_cfg', {}),
        'Stage-1', ['HC', 'W'], 'HC_W_PROBA_HC')

# ──────────────────────────────────────────────────────────────────
# TAB 4 — Stage-2 Result
# ──────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🟩 Stage-2 · O vs G (subset HC)")
    render_stage_result(
        st.session_state.get('fc_s2_results'),
        st.session_state.get('fc_s2_cfg', {}),
        'Stage-2', ['O', 'G'], 'OG_PROBA_O')

# ──────────────────────────────────────────────────────────────────
# TAB 5 — End-to-End (chaining Stage-1 → Stage-2)
# ──────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## 🎯 End-to-End · 3 kelas (O / G / W)")
    s1_res = st.session_state.get('fc_s1_results')
    s2_res = st.session_state.get('fc_s2_results')
    test_wells_e = st.session_state.get('fc_test_wells', [])
    if s1_res is None:
        st.info("Train Stage-1 dulu untuk evaluasi end-to-end.")
    elif combined is None or combined.empty:
        st.info("Data tidak tersedia.")
    elif not test_wells_e:
        st.info("Pilih Test Wells di sidebar.")
    else:
        df_pred, e2e_m = build_end_to_end(
            combined, test_wells_e, s1_res, s2_res)
        # Simpan untuk tab Well Log Viewer & Export
        st.session_state['fc_e2e_pred'] = df_pred

        if s2_res is None:
            st.warning("Stage-2 belum dilatih → end-to-end hanya bisa membedakan W vs HC "
                       "(tanpa pemecahan O/G).")

        if e2e_m is None:
            st.warning(
                "Tidak ada baris test yang punya label & prediksi lengkap.")
        else:
            cA, cB, cC = st.columns(3)
            cA.metric("N", f"{e2e_m['N']:,}")
            cB.metric("Accuracy", f"{e2e_m['accuracy']:.3f}")
            cC.metric("F1 (weighted)", f"{e2e_m['weighted']['f1']:.3f}")

            cL, cR = st.columns([1, 1])
            with cL:
                st.plotly_chart(plot_confusion_matrix(
                    e2e_m['confusion_matrix'], e2e_m['labels'],
                    title='End-to-End — Confusion Matrix (3 kelas)'),
                    use_container_width=True)
            with cR:
                st.dataframe(pd.DataFrame(e2e_m['per_class']),
                             use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────
# TAB 6 — Well Log Viewer
# ──────────────────────────────────────────────────────────────────
with tabs[5]:
    if combined is None or combined.empty:
        st.info("Belum ada data.")
    else:
        wells = sorted(combined['WELL_NAME'].unique().tolist())
        sel_well = st.selectbox("Pilih sumur", options=wells)
        # Pakai end-to-end pred bila tersedia & sumur termasuk test
        df_pred = st.session_state.get('fc_e2e_pred')
        df_well = None
        show_pred = False
        if (df_pred is not None and not df_pred.empty
                and sel_well in df_pred['WELL_NAME'].unique()):
            df_well = df_pred[df_pred['WELL_NAME'] == sel_well].copy()
            show_pred = True
        if df_well is None:
            df_well = combined[combined['WELL_NAME'] == sel_well].copy()

        if 'ZONE' in df_well.columns and df_well['ZONE'].nunique() > 1:
            zones = sorted([z for z in df_well['ZONE'].dropna().unique()
                            if str(z).upper() not in ('UNKNOWN', 'NAN', '')])
            sel_zones = st.multiselect(
                "Filter zona (kosong = semua)", options=zones)
            if sel_zones:
                df_well = df_well[df_well['ZONE'].isin(sel_zones)].copy()

        if df_well.empty:
            st.warning("Tidak ada data setelah filter.")
        else:
            fig = plot_well_log(df_well, sel_well, show_pred=show_pred)
            st.plotly_chart(fig, use_container_width=True)

            cnt = df_well.get('FLUID_LETTER', pd.Series()).value_counts()
            if not cnt.empty:
                st.markdown("**Distribusi FLUID Aktual di tampilan:** " +
                            " · ".join([f"{k}={v}" for k, v in cnt.items()]))
            if show_pred:
                cnt_p = df_well.get('FLUID_PRED_3', pd.Series()).value_counts()
                if not cnt_p.empty:
                    st.markdown("**Distribusi FLUID Prediksi di tampilan:** " +
                                " · ".join([f"{k}={v}" for k, v in cnt_p.items()]))

# ──────────────────────────────────────────────────────────────────
# TAB 7 — Export
# ──────────────────────────────────────────────────────────────────
with tabs[6]:
    s1_res = st.session_state.get('fc_s1_results')
    s2_res = st.session_state.get('fc_s2_results')

    st.markdown("### 💾 Export hasil")
    if s1_res is None and s2_res is None:
        st.info("Belum ada model. Train minimal 1 stage dari sidebar.")
    else:
        # End-to-end CSV (jika ada)
        df_pred = st.session_state.get('fc_e2e_pred')
        if df_pred is not None and not df_pred.empty:
            export_cols = ['WELL_NAME', 'DEPTH', 'ZONE',
                           'FLUID_CODE', 'FLUID_CLASS3',
                           'FLUID_HC_W', 'FLUID_OG',
                           'HC_W_PRED', 'HC_W_PROBA_HC',
                           'OG_PRED', 'OG_PROBA_O',
                           'FLUID_PRED_3']
            export_cols = [c for c in export_cols if c in df_pred.columns]
            csv_bytes = df_pred[export_cols].to_csv(
                index=False).encode('utf-8')
            st.download_button(
                "⬇ End-to-End Predictions (CSV)",
                data=csv_bytes,
                file_name="fluid_predictions_endtoend.csv",
                mime="text/csv",
                use_container_width=True)

        # Per-stage models pkl
        pkg = {
            'stage1': {
                'model': s1_res['model'] if s1_res else None,
                'label_encoder': s1_res['le'] if s1_res else None,
                'features': s1_res['features'] if s1_res else None,
                'config': st.session_state.get('fc_s1_cfg', {}),
            } if s1_res else None,
            'stage2': {
                'model': s2_res['model'] if s2_res else None,
                'label_encoder': s2_res['le'] if s2_res else None,
                'features': s2_res['features'] if s2_res else None,
                'config': st.session_state.get('fc_s2_cfg', {}),
            } if s2_res else None,
            'gr_norm_params': st.session_state.get('gr_norm_params', {}),
            'fluid_code_to_class3': FLUID_CODE_TO_CLASS3,
            'fluid_letter_to_code': FLUID_LETTER_TO_CODE,
        }
        pkl_bytes = pickle.dumps(pkg)
        st.download_button(
            "⬇ Model Package — Stage-1 & Stage-2 (.pkl)",
            data=pkl_bytes,
            file_name="fluid_two_stage_model.pkl",
            mime="application/octet-stream",
            use_container_width=True)

        # Metrics JSON
        export_metrics = {
            'stage1': s1_res['metrics'] if s1_res else None,
            'stage2': s2_res['metrics'] if s2_res else None,
        }
        m_export = json.loads(json.dumps(
            export_metrics, default=lambda o: None))
        json_bytes = json.dumps(m_export, indent=2).encode('utf-8')
        st.download_button(
            "⬇ Metrics (JSON)",
            data=json_bytes,
            file_name="fluid_metrics.json",
            mime="application/json",
            use_container_width=True)

        st.markdown(
            '<div class="ibox">Package berisi model Stage-1 & Stage-2 '
            '(masing-masing dengan feature & LabelEncoder sendiri), '
            'plus parameter GR_NORM untuk inference.</div>',
            unsafe_allow_html=True)
