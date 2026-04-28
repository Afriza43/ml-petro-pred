"""
Petro·ML  —  Petrophysics Machine Learning Dashboard
Upload 1 ZIP → pilih sumur test → merge zone/marker CSV → train → predict
"""

import lightgbm as lgb
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
import lasio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import zipfile
import json
import pickle
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Petro·ML",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;
  --border:#30363d;--border2:#21262d;
  --accent:#f0a500;--a2:#58a6ff;--a3:#3fb950;
  --danger:#f85149;--warn:#e3b341;
  --text:#e6edf3;--muted:#7d8590;--muted2:#484f58;
  --r:6px;
}
html,body,[class*="css"]{
  font-family:'IBM Plex Sans',sans-serif;
  background:var(--bg)!important;color:var(--text)!important;
}
[data-testid="stSidebar"]{
  background:var(--bg2)!important;
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] label{font-size:0.79rem!important;}
.main .block-container{padding:1.4rem 2rem;max-width:100%;}
.logo{font-family:'IBM Plex Mono',monospace;font-size:1.85rem;
  font-weight:600;color:var(--accent);letter-spacing:-1.5px;}
.logo-sub{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
  color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;}
.sec{font-family:'IBM Plex Mono',monospace;font-size:0.64rem;
  color:var(--muted2);text-transform:uppercase;letter-spacing:2px;
  padding:.65rem 0 .25rem;border-top:1px solid var(--border2);margin-top:.5rem;}
.kpi-row{display:flex;gap:10px;flex-wrap:wrap;margin:.7rem 0;}
.kpi{background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--r);padding:11px 16px;flex:1;min-width:120px;}
.kl{font-family:'IBM Plex Mono',monospace;font-size:.63rem;
  color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;}
.kv{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;}
.ks{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:var(--muted);margin-top:1px;}
.good{color:var(--a3)!important;}.ok{color:var(--accent)!important;}
.bad{color:var(--danger)!important;}.na{color:var(--muted)!important;}
.pill{display:inline-block;background:var(--bg3);border:1px solid var(--border);
  border-radius:20px;padding:2px 9px;font-family:'IBM Plex Mono',monospace;
  font-size:.67rem;color:var(--a2);margin:2px;}
.ibox{background:var(--bg2);border:1px solid var(--border);
  border-left:3px solid var(--a2);border-radius:var(--r);
  padding:8px 13px;font-size:.78rem;color:var(--muted);
  font-family:'IBM Plex Mono',monospace;margin:.35rem 0;line-height:1.6;}
.wbox{border-left-color:var(--warn)!important;}
.ebox{border-left-color:var(--danger)!important;}
.stButton>button{
  background:var(--accent)!important;color:#000!important;
  border:none!important;font-family:'IBM Plex Mono',monospace!important;
  font-weight:600!important;font-size:.82rem!important;
  border-radius:var(--r)!important;padding:.47rem 1.3rem!important;
}
.stButton>button:hover{background:#ffc233!important;transform:translateY(-1px);}
.stButton>button:disabled{
  background:var(--muted2)!important;color:var(--muted)!important;}
[data-testid="stTabs"] [role="tab"]{
  font-family:'IBM Plex Mono',monospace;font-size:.78rem;
  color:var(--muted)!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  color:var(--accent)!important;
  border-bottom:2px solid var(--accent)!important;}
[data-testid="stExpander"]{
  background:var(--bg2)!important;
  border:1px solid var(--border)!important;border-radius:var(--r)!important;}
hr{border-color:var(--border2)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
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

ALL_LOGS = ['GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT', 'VSH_LINEAR']
ALL_TARGETS = ['VSH', 'PHIE', 'SW']
TARGET_BOUNDS = {
    'VSH': (0.0, 1.0),
    'PHIE': (0.0, 0.3),
    'SW': (0.0, 1.0),
}
TARGET_COLORS = {
    'VSH': ('#f0a500', '#ffd166'),
    'PHIE': ('#58a6ff', '#a5d8ff'),
    'SW': ('#3fb950', '#8fffb0'),
}
LOG_COLORS = {'GR': '#3fb950', 'GR_NORM': '#00d26a', 'NPHI': '#58a6ff',
              'RHOB': '#f85149', 'RT': '#bc8cff'}
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
    'marker_df': None,
    'combined_df': None,
    'results': None,
    'trained': False,
    'cfg': {},
    'normalized': False,
    'gr_norm_params': {},
    'qc_log': None,
    'zip_hash': None,
    'zone_hash': None,
    'marker_hash': None,
    'test_results': None,
    'test_metrics': None,
    # Multi-structure state
    'app_page': 'Single Structure',
    # {name: {'wells': {}, 'zone_df': df, 'zip_hash': str, 'zone_hash': str}}
    'structures': {},
    'ms_combined_df': None,    # combined df from all structures
    'ms_qc_log': None,
    'ms_normalized': False,
    'ms_gr_norm_params': {},
    'ms_results': None,
    'ms_trained': False,
    'ms_cfg': {},
    'ms_test_results': None,
    'ms_test_metrics': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
def _normalize_well_name(name: str) -> str:
    """
    Normalisasi nama sumur ke format PREFIX-NNN[SUFFIX].
    BN-63    → BN-063
    BN-63TW  → BN-063TW
    BN-011TW → BN-011TW  (sudah 3 digit, tidak berubah)
    MJ-17    → MJ-017
    """
    import re
    # Cari pola: huruf(+tanda hubung)+angka(+suffix opsional)
    m = re.match(r'^([A-Za-z]+[-_])(\d+)([A-Za-z]*)$', name.strip())
    if m:
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        # Pad angka ke 3 digit, suffix (TW, A, B, dll) dibiarkan
        return f"{prefix.upper()}{digits.zfill(3)}{suffix.upper()}"
    return name.upper()


def _auto_convert_rhob_to_gcc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-convert RHOB dari kg/m3 ke g/cc per file LAS.

    Rule:
    - jika median RHOB valid > 100 DAN mayoritas nilai valid > 100,
      anggap RHOB masih kg/m3 lalu bagi 1000
    - jika tidak, anggap RHOB sudah g/cc
    """
    df = df.copy()

    if 'RHOB' not in df.columns:
        return df

    rhob_num = pd.to_numeric(df['RHOB'], errors='coerce')
    valid = rhob_num.dropna()
    valid = valid[valid > 0]

    if valid.empty:
        return df

    med = float(valid.median())

    # Convert hanya jika benar-benar dominan skala ribuan
    if med > 100:
        df['RHOB'] = rhob_num / 1000.0

    return df


def read_las_bytes(content: bytes, well_name: str):
    """
    Baca LAS dari bytes. Fallback otomatis jika mode cepat gagal.
    """
    def _parse(las_str: str):
        """Coba baca dengan mode cepat, fallback ke mode standar."""
        try:
            # Mode cepat: skip validasi baris per baris
            return lasio.read(
                io.StringIO(las_str),
                ignore_header_errors=True,
                read_policy='quick',
                null_policy='none',      # jangan replace null di lasio, kita handle manual
            )
        except Exception:
            # Fallback: mode standar lasio tanpa opsi tambahan
            return lasio.read(
                io.StringIO(las_str),
                ignore_header_errors=True,
            )

    # --- Decode bytes → string, coba UTF-8 dulu lalu latin-1 ---
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
    df.rename(columns={k: v for k, v in MNEMONIC_MAP.items()
                       if k in df.columns}, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    if 'VSH' not in df.columns and 'VCL' in df.columns:
        df['VSH'] = df['VCL']
    if 'GR_NORM' not in df.columns and 'GRN' in df.columns:
        df['GR_NORM'] = df['GRN']

    if 'DEPTH' not in df.columns:
        df.rename(columns={df.columns[0]: 'DEPTH'}, inplace=True)

    # Null value dari header LAS
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

    df['WELL_NAME'] = _normalize_well_name(well_name)

    if 'ZONE' not in df.columns:
        df['ZONE'] = 'UNKNOWN'
    if 'MARKER' not in df.columns:
        df['MARKER'] = 'UNKNOWN'

    return df.sort_values('DEPTH').reset_index(drop=True)


def load_zip(uploaded_zip) -> dict:
    wells = {}
    data = uploaded_zip.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
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

# ── Tambahkan fungsi cached ini di bagian atas, setelah fungsi load_zip ──


@st.cache_data(show_spinner=False)
def load_zip_cached(file_bytes: bytes) -> dict:
    """
    Load ZIP sekali saja. Di-cache berdasarkan isi file (bytes).
    Tidak akan re-run selama file yang sama tidak berubah.
    """
    import io
    import zipfile
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


def merge_zone_marker(df, zone_df, marker_df):
    """
    Gabungkan zone dan marker ke df sumur.
    zone_df : output dari read_zone_csv → kolom WELL_NAME, DEPTH_TOP, DEPTH_BOT, ZONE
    marker_df: kolom WELL_NAME, DEPTH/MD, MARKER
    """
    df = df.copy()
    wname = _normalize_well_name(df['WELL_NAME'].iloc[0])

    # ── Zone (format interval TOP/BOT) ──
    if zone_df is not None and len(zone_df) > 0:
        z = zone_df.copy()
        z.columns = [c.strip().upper() for c in z.columns]

        # Support kedua format: interval (TOP+BOT) dan marker-style
        wc = next((c for c in z.columns if 'WELL' in c), None)
        tc = next((c for c in z.columns
                   if c in ('DEPTH_TOP', 'TOP', 'FROM', 'MD_TOP')), None)
        bc = next((c for c in z.columns
                   if c in ('DEPTH_BOT', 'BOT', 'BASE', 'TO', 'MD_BOT')), None)
        znc = next((c for c in z.columns
                    if c in ('ZONE', 'FORMATION', 'FORM', 'UNIT', 'SURFACE')
                    or 'ZONE' in c), None)

        if all([wc, tc, bc, znc]):
            zw = z[z[wc].astype(str).str.strip().str.upper() == wname]
            for _, row in zw.iterrows():
                try:
                    top = float(row[tc])
                    bot = float(row[bc])
                    zone_val = str(row[znc]).strip()
                    if zone_val and zone_val.upper() not in ('NAN', 'NONE', ''):
                        m = (df['DEPTH'] >= top) & (df['DEPTH'] < bot)
                        df.loc[m, 'ZONE'] = zone_val
                except (ValueError, TypeError):
                    continue

    # ── Marker ──
    if marker_df is not None and len(marker_df) > 0:
        m = marker_df.copy()
        m.columns = [c.strip().upper() for c in m.columns]
        wc = next((c for c in m.columns if 'WELL' in c), None)
        dc = next((c for c in m.columns
                   if c in ('DEPTH', 'MD', 'DEPTH_MD') or 'DEPTH' in c), None)
        mc = next((c for c in m.columns
                   if c in ('MARKER', 'NAME', 'LABEL', 'SURFACE')
                   or 'MARKER' in c), None)
        if all([wc, dc, mc]):
            mw = m[m[wc].astype(str).str.strip().str.upper() == wname]
            mw = mw.copy()
            mw[dc] = pd.to_numeric(
                mw[dc].astype(str).str.replace(',', '.', regex=False),
                errors='coerce')
            mw = mw.dropna(subset=[dc]).sort_values(dc)
            for _, row in mw.iterrows():
                val = str(row[mc]).strip()
                if val and val.upper() not in ('NAN', 'NONE', ''):
                    df.loc[df['DEPTH'] >= row[dc], 'MARKER'] = val
    return df


def build_combined(wells, zone_df, marker_df):
    dfs = [merge_zone_marker(df, zone_df, marker_df) for df in wells.values()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════
# ZONE CSV — smart reader untuk berbagai format
# ══════════════════════════════════════════════════════════════════


def read_zone_csv(file_obj) -> pd.DataFrame:
    """
    Baca Zone CSV dengan auto-detect:
    - Separator: koma atau titik koma
    - Desimal: titik atau koma Eropa
    - Format: marker-style (WELL, MD, ZONE_NAME)
      → dikonversi ke (WELL_NAME, DEPTH_TOP, DEPTH_BOT, ZONE)

    Contoh input yang didukung:
      Well identifier,MD,Surface
      MJ-017TW,300,PLB
      MJ-017TW,520,         ← baris kosong = end marker

      Well identifier;MD;Surface
      TMB-005;372,24;ABF    ← desimal koma (European)
    """
    raw = file_obj.read() if hasattr(file_obj, 'read') else open(file_obj, 'rb').read()
    text = raw.decode('utf-8', errors='replace')

    # Auto-detect separator
    sep = ';' if text.count(';') > text.count(',') / 2 else ','

    # Jika separator ';' dan ada desimal koma → ganti desimal koma → titik
    # (hanya pada kolom numerik, bukan nama zona)
    if sep == ';':
        lines = []
        for line in text.splitlines():
            parts = line.split(';')
            fixed = []
            for i, p in enumerate(parts):
                p = p.strip()
                # Coba convert: jika ada koma dan bisa jadi float → ganti
                if ',' in p:
                    try:
                        float(p.replace(',', '.'))
                        p = p.replace(',', '.')
                    except ValueError:
                        pass  # bukan angka, biarkan
                fixed.append(p)
            lines.append(sep.join(fixed))
        text = '\n'.join(lines)

    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]

    # Auto-detect kolom
    well_col = next((c for c in df.columns
                     if 'WELL' in c or 'UWI' in c), None)
    depth_col = next((c for c in df.columns
                      if c in ('MD', 'DEPTH', 'DEPTH_MD', 'TVD')
                      or 'DEPTH' in c or c == 'MD'), None)
    zone_col = next((c for c in df.columns
                     if c in ('SURFACE', 'ZONE', 'FORMATION', 'FORM',
                              'UNIT', 'LAYER', 'NAME', 'MARKER')
                     or 'ZONE' in c or 'FORM' in c or 'SURF' in c), None)

    if not all([well_col, depth_col, zone_col]):
        # Fallback: kolom 0=well, 1=depth, 2=zone
        cols = df.columns.tolist()
        if len(cols) >= 3:
            well_col, depth_col, zone_col = cols[0], cols[1], cols[2]
        else:
            raise ValueError(
                f"Tidak bisa deteksi kolom Zone CSV. "
                f"Kolom ditemukan: {df.columns.tolist()}")

    df = df[[well_col, depth_col, zone_col]].copy()
    df.columns = ['WELL_NAME', 'MD', 'ZONE_NAME']
    df['WELL_NAME'] = df['WELL_NAME'].str.strip().apply(_normalize_well_name)
    df['MD'] = pd.to_numeric(df['MD'].str.strip()
                             .str.replace(',', '.', regex=False),
                             errors='coerce')
    df['ZONE_NAME'] = df['ZONE_NAME'].str.strip()

    return _convert_marker_to_intervals(df)


def _convert_marker_to_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi format marker (WELL, MD_top, ZONE_NAME) ke interval
    (WELL_NAME, DEPTH_TOP, DEPTH_BOT, ZONE).

    Logic:
    - Tiap baris = top dari zona baru
    - BOT dari zona N = TOP dari zona N+1
    - BOT dari zona terakhir = 99999 (tak terbatas ke bawah)
    - Baris dengan ZONE_NAME kosong/NaN = END marker, tidak jadi zona
    """
    records = []
    for well, grp in df.groupby('WELL_NAME'):
        grp = grp.sort_values('MD').reset_index(drop=True)
        # Filter baris yang punya nama zona
        valid = grp[grp['ZONE_NAME'].notna() &
                    (grp['ZONE_NAME'] != '') &
                    (grp['ZONE_NAME'].str.upper() != 'NAN')]
        valid = valid.reset_index(drop=True)

        for i, row in valid.iterrows():
            top = row['MD']
            # BOT = MD baris berikutnya di grp asli (bukan hanya valid)
            # → cari entry berikutnya setelah top di grp asli
            next_entries = grp[grp['MD'] > top]['MD']
            bot = float(next_entries.iloc[0]) if len(
                next_entries) > 0 else 99999.0

            records.append({
                'WELL_NAME': well,
                'DEPTH_TOP': float(top),
                'DEPTH_BOT': bot,
                'ZONE': row['ZONE_NAME'],
            })

    return pd.DataFrame(records)
# ══════════════════════════════════════════════════════════════════
# GR NORMALIZATION  (P3/P97 per zona — metode sama seperti notebook)
# ══════════════════════════════════════════════════════════════════


def compute_gr_norm_params(df: pd.DataFrame,
                           pct_low_old: float = 3,
                           pct_high_old: float = 97,
                           pct_low_new: float = 3,
                           pct_high_new: float = 97) -> dict:
    """
    Parameter normalisasi GR dengan konsep OLD -> NEW

    OLD:
      percentile per WELL_NAME dari data GR pada well tersebut
    NEW:
      percentile gabungan semua sumur dalam ZIP (1 struktur)

    Rumus final akan dipakai di apply_gr_norm():
      grz = (pct_high_new - pct_low_new) / (pct_high_old - pct_low_old)
      grn = grz * (gr - pct_low_old) + pct_low_new

    Catatan:
    - 'old' = nilai GR percentile masing-masing well
    - 'new' = nilai GR percentile keseluruhan struktur (semua well di zip)
    """
    if 'GR' not in df.columns or 'WELL_NAME' not in df.columns:
        return {}

    df = df.copy()
    df['WELL_NAME'] = df['WELL_NAME'].astype(str).str.strip()

    gr_valid = df[df['GR'].notna()].copy()
    if gr_valid.empty:
        return {}

    params = {
        'global_new': {},
        'well_old': {},
        'pct_config': {
            'pct_low_old': float(pct_low_old),
            'pct_high_old': float(pct_high_old),
            'pct_low_new': float(pct_low_new),
            'pct_high_new': float(pct_high_new),
        }
    }

    # NEW = percentile gabungan semua sumur dalam struktur
    global_low_new = float(np.nanpercentile(gr_valid['GR'], pct_low_new))
    global_high_new = float(np.nanpercentile(gr_valid['GR'], pct_high_new))
    params['global_new'] = {
        'p_low': global_low_new,
        'p_high': global_high_new,
        'N': int(len(gr_valid)),
        'source': 'all_wells_in_zip'
    }

    # OLD = percentile masing-masing well
    for well, grp in gr_valid.groupby('WELL_NAME'):
        vals = grp['GR'].dropna()
        if len(vals) < 10:
            old_low = global_low_new
            old_high = global_high_new
            src = 'global_fallback'
        else:
            old_low = float(np.nanpercentile(vals, pct_low_old))
            old_high = float(np.nanpercentile(vals, pct_high_old))
            src = 'per_well'

        params['well_old'][well] = {
            'p_low': old_low,
            'p_high': old_high,
            'N': int(len(vals)),
            'source': src
        }

    return params


def apply_gr_norm(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Terapkan normalisasi GR dengan rumus remapping:

        grz = (pct_high_new - pct_low_new) / (pct_high_old - pct_low_old)
        grn = grz * (gr - pct_low_old) + pct_low_new

    Di sini:
    - pct_low_old / pct_high_old = nilai percentile GR masing-masing well
    - pct_low_new / pct_high_new = nilai percentile GR gabungan semua well dalam zip
    """
    df = df.copy()
    df['GR_NORM'] = np.nan

    if not params or 'GR' not in df.columns or 'WELL_NAME' not in df.columns:
        return df

    df['WELL_NAME'] = df['WELL_NAME'].astype(str).str.strip()

    global_new = params.get('global_new', {})
    well_old = params.get('well_old', {})

    pct_low_new = global_new.get('p_low', np.nan)
    pct_high_new = global_new.get('p_high', np.nan)

    if pd.isna(pct_low_new) or pd.isna(pct_high_new):
        return df

    for well, idx in df.groupby('WELL_NAME').groups.items():
        old_p = well_old.get(well, None)
        if old_p is None:
            pct_low_old = pct_low_new
            pct_high_old = pct_high_new
        else:
            pct_low_old = old_p['p_low']
            pct_high_old = old_p['p_high']

        den = pct_high_old - pct_low_old
        if pd.isna(den) or den == 0:
            df.loc[idx, 'GR_NORM'] = np.nan
            continue

        grz = (pct_high_new - pct_low_new) / den
        df.loc[idx, 'GR_NORM'] = grz * \
            (df.loc[idx, 'GR'] - pct_low_old) + pct_low_new

    return df

# ══════════════════════════════════════════════════════════════════
# QC HELPERS
# ══════════════════════════════════════════════════════════════════


def apply_zscore_filter(df: pd.DataFrame, cols: list[str], threshold: float = 3.0) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    log = {'zscore_rows_dropped': 0}
    cols = [c for c in cols if c in df.columns]
    if not cols or df.empty:
        return df, log

    bad_mask = pd.Series(False, index=df.index)
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        valid = s.notna()
        if valid.sum() < 10:
            continue
        mu = s[valid].mean()
        sigma = s[valid].std(ddof=0)
        if sigma is None or np.isnan(sigma) or sigma == 0:
            continue
        z = (s - mu) / sigma
        bad_mask = bad_mask | (z.abs() > threshold)

    log['zscore_rows_dropped'] = int(bad_mask.sum())
    if bad_mask.any():
        df = df.loc[~bad_mask].copy()
    return df, log


def build_vsh_linear_feature(df: pd.DataFrame,
                             grn_col: str = 'GR_NORM',
                             zone_col: str = 'ZONE',
                             p_low: float = 3,
                             p_high: float = 97) -> pd.DataFrame:
    """
    Hitung VSH_LINEAR dari GR_NORM:
        VSH_LINEAR = (GR_NORM - GR_MA) / (GR_SH - GR_MA)

    Sementara:
    - GR_MA dan GR_SH dihitung dari distribusi GR_NORM seluruh data valid
    - hasil di-clip ke [0,1]
    """
    df = df.copy()
    df['VSH_LINEAR'] = np.nan

    if grn_col not in df.columns or not df[grn_col].notna().any():
        return df

    vals = df[grn_col].dropna()
    if len(vals) == 0:
        return df

    gr_ma = float(np.nanpercentile(vals, p_low))
    gr_sh = float(np.nanpercentile(vals, p_high))

    rng = gr_sh - gr_ma
    if pd.isna(rng) or rng == 0:
        return df

    mask = df[grn_col].notna()
    df.loc[mask, 'VSH_LINEAR'] = (
        (df.loc[mask, grn_col] - gr_ma) / rng).clip(0, 1)
    return df

# ══════════════════════════════════════════════════════════════════
# TARGET POLICY
# ══════════════════════════════════════════════════════════════════


def apply_target_training_policy(df: pd.DataFrame, target: str,
                                 rules=None):
    """
    Policy khusus per target untuk training/evaluasi.
    Rules (semua default True = aktif):
    - rule_vsh_drop_zero: buang VSH = 0 dari training (coal marker)
    - rule_sw_drop_one:   buang SW  = 1 dari training
    - rule_phie_drop_zero: buang PHIE = 0 dari training
    Jika rules dict None, default lama dipakai (drop VSH=0 saja).
    """
    df = df.copy()
    info = {
        'drop_vsh_zero_for_training': 0,
        'drop_sw_one_for_training': 0,
        'drop_phie_zero_for_training': 0,
        'keep_sw_eq_1': 0,
    }
    if rules is None:
        rules = {
            'rule_vsh_drop_zero': True,
            'rule_sw_drop_one': False,
            'rule_phie_drop_zero': False,
        }

    if target == 'VSH' and 'VSH' in df.columns:
        if rules.get('rule_vsh_drop_zero', True):
            mask_zero = df['VSH'].notna() & (df['VSH'] == 0)
            info['drop_vsh_zero_for_training'] = int(mask_zero.sum())
            df = df.loc[~mask_zero].copy()

    if target == 'SW' and 'SW' in df.columns:
        info['keep_sw_eq_1'] = int((df['SW'] == 1).sum())
        if rules.get('rule_sw_drop_one', False):
            mask_one = df['SW'].notna() & (df['SW'] == 1)
            info['drop_sw_one_for_training'] = int(mask_one.sum())
            df = df.loc[~mask_one].copy()

    if target == 'PHIE' and 'PHIE' in df.columns:
        if rules.get('rule_phie_drop_zero', False):
            mask_zero = df['PHIE'].notna() & (df['PHIE'] == 0)
            info['drop_phie_zero_for_training'] = int(mask_zero.sum())
            df = df.loc[~mask_zero].copy()

    return df, info

# ══════════════════════════════════════════════════════════════════
# QC PIPELINE  (dari notebook A6)
# ══════════════════════════════════════════════════════════════════


def run_qc_pipeline(df: pd.DataFrame, use_zscore: bool = False, zscore_threshold: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """
    QC utama:
      1. Hapus baris di mana GR, RT, NPHI, RHOB semuanya NaN.
      2. RT ≤ 0 → NaN.
      3. Target bounds:
         - VSH:  0–1
         - PHIE: 0–0.5
         - SW:   0–1
      4. Label kosong TIDAK dibuang di tahap QC global.
         Missing label akan ditangani per target saat training.
      5. SW = 1 dipertahankan; hanya dicatat untuk audit.

    Return: (df_clean, qc_log_dict)
    """
    log = {}
    n0 = len(df)
    df = df.copy()
    df = build_vsh_linear_feature(df)

    if use_zscore:
        zdf, zlog = apply_zscore_filter(
            df, ['GR', 'NPHI', 'RHOB', 'RT'], threshold=zscore_threshold)
        df = zdf
        log.update(zlog)
    else:
        log['zscore_rows_dropped'] = 0

    log_cols = [c for c in ['GR', 'NPHI', 'RHOB', 'RT'] if c in df.columns]
    if log_cols:
        before = len(df)
        df = df.dropna(subset=log_cols, how='all')
        log['drop_all_nan_logs'] = int(before - len(df))
    else:
        log['drop_all_nan_logs'] = 0

    if 'RT' in df.columns:
        n_rt = int((df['RT'].notna() & (df['RT'] <= 0)).sum())
        df.loc[df['RT'] <= 0, 'RT'] = np.nan
        log['rt_invalid_to_nan'] = n_rt
    else:
        log['rt_invalid_to_nan'] = 0

    for tgt, (lo, hi) in TARGET_BOUNDS.items():
        key = f'drop_{tgt.lower()}_out_of_range'
        if tgt in df.columns:
            bad = df[tgt].notna() & ((df[tgt] < lo) | (df[tgt] > hi))
            log[key] = int(bad.sum())
            if bad.any():
                df = df.loc[~bad].copy()
        else:
            log[key] = 0

    if 'SW' in df.columns:
        log['sw_eq_1_kept'] = int((df['SW'] == 1).sum())
    else:
        log['sw_eq_1_kept'] = 0

    if 'VSH' in df.columns:
        log['vsh_eq_0_kept_for_rule'] = int((df['VSH'] == 0).sum())
    else:
        log['vsh_eq_0_kept_for_rule'] = 0

    log['drop_empty_labels'] = 0
    log['total_dropped'] = int(n0 - len(df))
    log['remaining'] = int(len(df))
    return df, log


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
def compute_features(df, le_zone, opts):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df = build_vsh_linear_feature(df)

    if 'RT' in df.columns:
        df['LOG_RT'] = np.where(df['RT'] > 0, np.log10(df['RT']), np.nan)

    if 'RHOB' in df.columns and 'NPHI' in df.columns:
        RHO_MA = 2.65
        RHO_FL = 1.00
        phid = (RHO_MA - df['RHOB']) / (RHO_MA - RHO_FL)
        if opts.get('use_dn_sep'):
            df['DN_SEP'] = df['NPHI'] - phid
        if opts.get('use_crossover'):
            df['NPHI_RHOB_CROSS'] = phid - df['NPHI']
            df['CROSS_POS'] = df['NPHI_RHOB_CROSS'].clip(lower=0)

    zone_clean = df['ZONE'].fillna('UNKNOWN').astype(str).str.strip(
    ) if 'ZONE' in df.columns else pd.Series('UNKNOWN', index=df.index)
    df['ZONE_IS_VALID'] = ~zone_clean.str.upper().isin(
        ['UNKNOWN', 'NAN', '', 'NONE'])
    if opts.get('use_zone') and le_zone is not None:
        known = set(le_zone.classes_)
        zone_known = zone_clean.apply(lambda x: x if x in known else np.nan)
        df['ZONE_ENC'] = np.nan
        mask_zone = zone_known.notna()
        if mask_zone.any():
            df.loc[mask_zone, 'ZONE_ENC'] = le_zone.transform(
                zone_known.loc[mask_zone])

    for col in ['VSH_PRED', 'PHIE_PRED', 'SW_PRED']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def resolve_feats(logs, opts, extra=None):
    """Legacy — masih dipakai untuk feat_preview di sidebar."""
    feats = ['LOG_RT' if lg == 'RT' else lg for lg in logs]
    if opts.get('use_dn_sep'):
        feats.append('DN_SEP')
    if opts.get('use_crossover'):
        feats += ['NPHI_RHOB_CROSS', 'CROSS_POS']
    if opts.get('use_zone'):
        feats.append('ZONE_ENC')
    if extra:
        feats += extra
    return list(dict.fromkeys(feats))


def resolve_feats_for_target(chosen: list) -> list:
    """
    Konversi pilihan user → nama kolom aktual di DataFrame.
    RT → LOG_RT
    Semua lain langsung dipakai as-is (sudah di-compute di compute_features)
    """
    feats = []
    for f in chosen:
        if f == 'RT':
            feats.append('LOG_RT')
        else:
            feats.append(f)   # NPHI_RHOB_CROSS, CROSS_POS, DN_SEP — langsung
    return list(dict.fromkeys(feats))


def build_model(model_name: str, params: dict):
    model_name = (model_name or 'lightgbm').lower()
    if model_name == 'ann':
        ann_params = params.copy()
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('mlp', MLPRegressor(**ann_params))
        ])
    if model_name == 'randomforest':
        rf_params = params.copy()
        rf_params.pop('verbosity', None)
        return RandomForestRegressor(**rf_params)
    if model_name == 'extratrees':
        et_params = params.copy()
        et_params.pop('verbosity', None)
        return ExtraTreesRegressor(**et_params)
    if model_name == 'catboost':
        if CatBoostRegressor is None:
            raise ImportError("CatBoost belum terinstall di environment ini.")
        return CatBoostRegressor(**params)
    if model_name == 'xgboost':
        if XGBRegressor is None:
            raise ImportError("XGBoost belum terinstall. pip install xgboost")
        xgb_p = params.copy()
        return XGBRegressor(**xgb_p)
    if model_name == 'stacking':
        sp = params.copy()
        cv_folds = int(sp.pop('cv', 5))
        nj = int(sp.pop('n_jobs', -1))
        lgb_ne = int(sp.get('lgb_n_estimators', 500))
        lgb_lr = float(sp.get('lgb_learning_rate', 0.03))
        rf_ne = int(sp.get('rf_n_estimators', 300))
        ridge_a = float(sp.get('ridge_alpha', 1.0))
        base = [
            ('lgb', lgb.LGBMRegressor(
                n_estimators=lgb_ne, learning_rate=lgb_lr,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=1, verbose=-1)),
            ('rf', RandomForestRegressor(
                n_estimators=rf_ne, max_depth=14,
                random_state=42, n_jobs=1)),
        ]
        if CatBoostRegressor is not None:
            base.append(('cb', CatBoostRegressor(
                iterations=300, learning_rate=0.05, depth=6,
                random_seed=42, verbose=0)))
        return StackingRegressor(
            estimators=base, final_estimator=Ridge(alpha=ridge_a),
            cv=cv_folds, n_jobs=nj, passthrough=False)
    # Default: LightGBM
    return lgb.LGBMRegressor(**params)


def clip_target_predictions(pred, target):
    lo, hi = TARGET_BOUNDS.get(target, (None, None))
    pred = np.asarray(pred, dtype=float)
    if lo is not None and hi is not None:
        pred = np.clip(pred, lo, hi)
    return pred


def _fit_model(model, X, y, sw=None):
    """Fit model with optional sample_weight (skip if model doesn't support it)."""
    if sw is not None and len(sw) == len(X):
        try:
            model.fit(X, y, sample_weight=sw)
        except TypeError:
            model.fit(X, y)
    else:
        model.fit(X, y)


def make_oof_predictions(clean_df: pd.DataFrame, feature_cols: list, target_col: str,
                         model_name: str, model_params: dict,
                         sample_weight: pd.Series = None) -> pd.Series:
    preds = pd.Series(np.nan, index=clean_df.index, dtype=float)
    if clean_df.empty:
        return preds

    wells = clean_df['WELL_NAME'].dropna().astype(str).unique().tolist()
    if len(wells) >= 2:
        for holdout in wells:
            tr = clean_df[clean_df['WELL_NAME'] != holdout]
            va = clean_df[clean_df['WELL_NAME'] == holdout]
            tr = tr.dropna(subset=feature_cols + [target_col])
            va = va.dropna(subset=feature_cols)
            if len(tr) < 20 or len(va) == 0:
                continue
            model = build_model(model_name, model_params)
            sw = sample_weight.loc[tr.index] if sample_weight is not None else None
            _fit_model(model, tr[feature_cols], tr[target_col], sw)
            preds.loc[va.index] = clip_target_predictions(
                model.predict(va[feature_cols]), target_col)
        return preds

    n_splits = min(5, max(2, len(clean_df) // 50))
    if len(clean_df) < n_splits:
        n_splits = max(2, min(3, len(clean_df)))
    if len(clean_df) < 2:
        return preds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(clean_df):
        tr = clean_df.iloc[tr_idx].dropna(subset=feature_cols + [target_col])
        va = clean_df.iloc[va_idx].dropna(subset=feature_cols)
        if len(tr) < 20 or len(va) == 0:
            continue
        model = build_model(model_name, model_params)
        sw = sample_weight.loc[tr.index] if sample_weight is not None else None
        _fit_model(model, tr[feature_cols], tr[target_col], sw)
        preds.loc[va.index] = clip_target_predictions(
            model.predict(va[feature_cols]), target_col)
    return preds


# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════


def safe_m(yt, yp):
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    ok = ~(np.isnan(yt) | np.isnan(yp))
    n = int(ok.sum())
    if n < 5:
        return {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'N': n}
    return {'R2': round(float(r2_score(yt[ok], yp[ok])), 4),
            'RMSE': round(float(np.sqrt(mean_squared_error(yt[ok], yp[ok]))), 5),
            'MAE': round(float(mean_absolute_error(yt[ok], yp[ok])), 5), 'N': n}


def r2c(r2):
    if np.isnan(r2):
        return 'na'
    return 'good' if r2 >= 0.75 else ('ok' if r2 >= 0.50 else 'bad')


def compute_zone_metrics(df_te, targets, zone_col='ZONE'):
    """
    Hitung metrics per zona dari df_te.
    Return: {target: DataFrame(ZONE, R2, RMSE, MAE, N)}
    """
    results = {}
    if df_te is None or len(df_te) == 0 or zone_col not in df_te.columns:
        return results
    for tgt in targets:
        if tgt not in df_te.columns or f'{tgt}_PRED' not in df_te.columns:
            continue
        rows = []
        for z, grp in df_te.groupby(zone_col):
            if str(z).upper() in ('UNKNOWN', 'NAN', '', 'NONE'):
                continue
            m = safe_m(grp[tgt], grp[f'{tgt}_PRED'])
            if m['N'] >= 5:
                rows.append({'ZONE': z, **m})
        if rows:
            results[tgt] = pd.DataFrame(
                rows).sort_values('R2', ascending=False)
    return results


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════
def run_training(combined, test_wells, target_feats, opts, params, model_name='lightgbm', training_mode='standard', sample_weight=None, target_rules=None):
    """
    Training per target dengan propagated feature non-leaky.
    - Preprocessing params (LabelEncoder, VSH_LINEAR) dihitung dari TRAIN only.
    - VSH/PHIE/SW di train-set dipropagasikan memakai OOF prediction.
    - sample_weight: optional array aligned with combined index.
    """
    if combined is None or len(combined) == 0:
        st.error("❌ Data belum tersedia — jalankan QC Pipeline dulu.")
        return None

    if not test_wells:
        st.error("❌ Pilih minimal 1 sumur sebagai Test Set.")
        return None

    mask_test = combined['WELL_NAME'].isin(test_wells)
    if (~mask_test).sum() == 0:
        st.error("❌ Semua sumur dijadikan test — tidak ada data training.")
        return None

    # ── Reset prediksi lama sebelum training baru ──
    for pc in ['VSH_PRED', 'PHIE_PRED', 'SW_PRED']:
        if pc in combined.columns:
            combined[pc] = np.nan

    # ── VSH_LINEAR: hitung dari TRAIN only, apply ke semua ──
    train_only = combined[~mask_test]
    if 'GR_NORM' in train_only.columns and train_only['GR_NORM'].notna().any():
        grn_valid = train_only[train_only['GR_NORM'].notna()]
        zone_s = train_only['ZONE'].fillna('UNKNOWN').astype(str).str.strip(
        ) if 'ZONE' in train_only.columns else pd.Series('UNKNOWN', index=train_only.index)

        g_ma = float(np.nanpercentile(grn_valid['GR_NORM'], 3))
        g_sh = float(np.nanpercentile(grn_valid['GR_NORM'], 97))
        if g_sh <= g_ma:
            g_ma, g_sh = 0.0, 1.0

        vsh_lin_params = {}
        for z, grp in grn_valid.groupby(zone_s.loc[grn_valid.index]):
            if str(z).upper() in ('UNKNOWN', 'NAN', '', 'NONE'):
                continue
            vals = grp['GR_NORM'].dropna()
            if len(vals) < 20:
                vsh_lin_params[z] = (g_ma, g_sh)
            else:
                z_ma = float(np.nanpercentile(vals, 3))
                z_sh = float(np.nanpercentile(vals, 97))
                if z_sh <= z_ma:
                    z_ma, z_sh = g_ma, g_sh
                vsh_lin_params[z] = (z_ma, z_sh)

        # Apply train-derived params to ALL data
        combined = combined.copy()
        combined['VSH_LINEAR'] = np.nan
        zone_all = combined['ZONE'].fillna('UNKNOWN').astype(str).str.strip(
        ) if 'ZONE' in combined.columns else pd.Series('UNKNOWN', index=combined.index)

        for z, (z_ma, z_sh) in vsh_lin_params.items():
            m = (zone_all == z) & combined['GR_NORM'].notna()
            if m.any():
                rng = max(z_sh - z_ma, 1e-9)
                combined.loc[m, 'VSH_LINEAR'] = (
                    (combined.loc[m, 'GR_NORM'] - z_ma) / rng).clip(0, 1)

        m_global = combined['VSH_LINEAR'].isna() & combined['GR_NORM'].notna()
        if m_global.any():
            rng = max(g_sh - g_ma, 1e-9)
            combined.loc[m_global, 'VSH_LINEAR'] = (
                (combined.loc[m_global, 'GR_NORM'] - g_ma) / rng).clip(0, 1)
    elif 'VSH_LINEAR' not in combined.columns or combined['VSH_LINEAR'].isna().all():
        combined = build_vsh_linear_feature(
            combined, grn_col='GR_NORM', zone_col='ZONE')

    # ── LabelEncoder: fit on TRAIN only ──
    le_zone = None
    if opts.get('use_zone') and 'ZONE' in combined.columns:
        zone_series = combined.loc[~mask_test, 'ZONE'].fillna(
            'UNKNOWN').astype(str).str.strip()
        zones = sorted([z for z in zone_series.unique().tolist()
                        if z.upper() not in ('UNKNOWN', 'NAN', '', 'NONE')])
        if zones:
            le_zone = LabelEncoder()
            le_zone.fit(zones)

    df_tr = compute_features(combined[~mask_test].copy(), le_zone, opts)
    df_te = compute_features(combined[mask_test].copy(), le_zone, opts)

    if df_tr is None or len(df_tr) == 0:
        st.error("❌ Data training kosong setelah feature engineering.")
        return None

    if df_te is None or len(df_te) == 0:
        st.warning("⚠ Data test kosong — metrics tidak akan dihitung.")
        df_te = pd.DataFrame()

    if 'ZONE_IS_VALID' in df_tr.columns:
        before = len(df_tr)
        df_tr = df_tr[df_tr['ZONE_IS_VALID']].copy()
        n_drop_unknown = before - len(df_tr)
        if n_drop_unknown > 0:
            st.info(
                f"ℹ Training: {n_drop_unknown:,} baris ZONE=UNKNOWN tidak dipakai untuk training.")

    # ── Compute sample weights if provided ──
    sw_series = None
    if sample_weight is not None and 'STRUCTURE' in df_tr.columns:
        struct_counts = df_tr['STRUCTURE'].value_counts()
        if len(struct_counts) > 1:
            max_count = struct_counts.max()
            sw_series = df_tr['STRUCTURE'].map(
                lambda s: max_count / struct_counts.get(s, max_count))
            st.info(f"ℹ Pembobotan per struktur aktif: {dict(struct_counts)}")

    models = {}
    feat_imp = {}
    metrics = {}
    train_audit = {}
    ordered_targets = [t for t in ['VSH', 'PHIE', 'SW'] if t in target_feats]

    for tgt in ordered_targets:
        chosen = target_feats.get(tgt, [])
        if not chosen:
            st.warning(f"⚠ {tgt}: tidak ada feature dipilih — skip")
            continue
        if tgt not in df_tr.columns:
            st.warning(f"⚠ {tgt}: kolom tidak ada di data — skip")
            continue

        feat_cols = resolve_feats_for_target(chosen)
        feat_tr = [f for f in feat_cols if f in df_tr.columns]
        feat_te = [f for f in feat_cols if f in df_te.columns]

        missing_tr = [f for f in feat_cols if f not in df_tr.columns]
        if missing_tr:
            st.warning(
                f"⚠ {tgt}: kolom training tidak ada: {missing_tr} — diabaikan")

        if not feat_tr:
            st.error(f"❌ {tgt}: tidak ada feature valid — skip")
            continue

        clean = df_tr.dropna(subset=feat_tr + [tgt]).copy()

        # Policy khusus per target (rules bisa ditoggle user)
        clean, tgt_policy = apply_target_training_policy(
            clean, tgt, rules=target_rules)

        if tgt == 'VSH' and tgt_policy.get('drop_vsh_zero_for_training', 0) > 0:
            st.info(
                f"ℹ VSH rule aktif: {tgt_policy['drop_vsh_zero_for_training']:,} baris dengan VSH = 0 "
                f"tidak dipakai training (coal marker)."
            )
        if tgt == 'SW' and tgt_policy.get('drop_sw_one_for_training', 0) > 0:
            st.info(
                f"ℹ SW rule aktif: {tgt_policy['drop_sw_one_for_training']:,} baris dengan SW = 1 "
                f"tidak dipakai training."
            )
        if tgt == 'PHIE' and tgt_policy.get('drop_phie_zero_for_training', 0) > 0:
            st.info(
                f"ℹ PHIE rule aktif: {tgt_policy['drop_phie_zero_for_training']:,} baris dengan PHIE = 0 "
                f"tidak dipakai training."
            )

        use_hybrid_vsh = (
            training_mode == 'hybrid_vsh_linear_residual' and tgt == 'VSH')
        target_train_col = tgt
        if use_hybrid_vsh:
            if 'VSH_LINEAR' not in clean.columns:
                st.warning(
                    '⚠ VSH_LINEAR tidak tersedia — hybrid residual VSH dilewati, fallback ke standard.')
                use_hybrid_vsh = False
            else:
                clean = clean.dropna(subset=['VSH_LINEAR']).copy()
                clean['_RESIDUAL_TARGET_'] = clean['VSH'] - clean['VSH_LINEAR']
                target_train_col = '_RESIDUAL_TARGET_'

        if len(clean) < 20:
            st.warning(f"⚠ {tgt}: hanya {len(clean)} baris training — skip")
            continue

        _sw_clean = sw_series.loc[clean.index] if sw_series is not None else None
        clean['_OOF_PRED_'] = make_oof_predictions(
            clean, feat_tr, target_train_col, model_name, params,
            sample_weight=_sw_clean)

        if use_hybrid_vsh:
            df_tr.loc[clean.index, f'{tgt}_PRED'] = clip_target_predictions(
                clean['VSH_LINEAR'] + clean['_OOF_PRED_'], tgt)
        else:
            df_tr.loc[clean.index, f'{tgt}_PRED'] = clean['_OOF_PRED_']

        model = build_model(model_name, params)
        _fit_model(model, clean[feat_tr], clean[target_train_col], _sw_clean)
        models[tgt] = {
            'model': model,
            'ft_tr': feat_tr,
            'ft_te': feat_te,
            'ft_chosen': chosen,
        }

        fi = getattr(model, 'feature_importances_', None)
        # Stacking: ambil dari base estimator pertama yang punya feature_importances_
        if fi is None and hasattr(model, 'estimators_'):
            for _bname, _best in model.estimators_:
                _bfi = getattr(_best, 'feature_importances_', None)
                if _bfi is not None:
                    fi = _bfi
                    break
        if fi is None and hasattr(model, 'get_feature_importance'):
            fi = model.get_feature_importance()
        if fi is not None:
            fi = np.asarray(fi, dtype=float)
            fi_sum = float(fi.sum())
            pct = (fi / fi_sum *
                   100.0) if fi_sum > 0 else np.zeros_like(fi, dtype=float)
            feat_imp[tgt] = dict(
                sorted(zip(feat_tr, pct.tolist()), key=lambda x: -x[1]))
        else:
            feat_imp[tgt] = {f: 0.0 for f in feat_tr}

        if len(df_te) > 0 and feat_te:
            missing_te = [f for f in feat_cols if f not in df_te.columns]
            if missing_te:
                st.warning(
                    f"⚠ {tgt} test: kolom tidak ada: {missing_te} — diabaikan")
            pred_mask = df_te[feat_te].notna().all(axis=1)
            if 'ZONE_IS_VALID' in df_te.columns:
                pred_mask = pred_mask & df_te['ZONE_IS_VALID']
            if pred_mask.sum() > 0 and len(feat_te) == len(feat_tr):
                raw_pred = model.predict(df_te.loc[pred_mask, feat_te])
                if training_mode == 'hybrid_vsh_linear_residual' and tgt == 'VSH' and 'VSH_LINEAR' in df_te.columns:
                    df_te.loc[pred_mask, f'{tgt}_PRED'] = clip_target_predictions(
                        df_te.loc[pred_mask, 'VSH_LINEAR'] + raw_pred, tgt)
                else:
                    df_te.loc[pred_mask, f'{tgt}_PRED'] = clip_target_predictions(
                        raw_pred, tgt)
            elif pred_mask.sum() > 0:
                st.warning(
                    f"⚠ {tgt}: jumlah feature train ({len(feat_tr)}) ≠ test ({len(feat_te)}) — prediksi test dilewati")

            if tgt in df_te.columns and f'{tgt}_PRED' in df_te.columns:
                for w in test_wells:
                    wm = df_te['WELL_NAME'] == w

                    # Policy evaluasi
                    if tgt == 'VSH':
                        wm = wm & (df_te[tgt] != 0)

                    metrics.setdefault(tgt, {})[w] = safe_m(
                        df_te.loc[wm, tgt], df_te.loc[wm, f'{tgt}_PRED'])

        train_audit[tgt] = {
            'n_train_rows': int(len(clean)),
            'n_oof_rows': int(clean['_OOF_PRED_'].notna().sum()),
            'n_train_wells': int(clean['WELL_NAME'].nunique()),
            'drop_vsh_zero_for_training': int(tgt_policy.get('drop_vsh_zero_for_training', 0)),
            'keep_sw_eq_1': int(tgt_policy.get('keep_sw_eq_1', 0)),
            'hybrid_vsh_linear_residual': bool(use_hybrid_vsh),
        }

    return dict(df_tr=df_tr, df_te=df_te, models=models,
                le_zone=le_zone, feat_imp=feat_imp,
                metrics=metrics, train_audit=train_audit)

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════


def plot_log(df, targets, well_name, zone_filter=None):
    """zone_filter: None atau list nama zona untuk dibatasi. Rows di luar
    zona yang dipilih akan di-NaN-kan (tetap jaga sumbu depth)."""
    df = df.copy()
    if zone_filter is not None and len(zone_filter) > 0 and 'ZONE' in df.columns:
        zcol = df['ZONE'].fillna('UNKNOWN').astype(str)
        keep = zcol.isin([str(z) for z in zone_filter])
        # NaN-kan kolom log & target di luar zona agar sumbu depth tetap utuh
        cols_null = [c for c in ['GR', 'GR_NORM', 'RHOB', 'NPHI', 'RT']
                     if c in df.columns]
        for t in targets:
            cols_null += [c for c in [t, f'{t}_PRED'] if c in df.columns]
        for c in cols_null:
            df.loc[~keep, c] = np.nan
    pred_tgts = [t for t in targets if f'{t}_PRED' in df.columns]
    nc = 3+len(pred_tgts)
    widths = [1, 1.1, 0.85]+[1.2]*len(pred_tgts)
    titles = ['GR', 'RHOB / NPHI', 'RT (log)']+pred_tgts
    fig = make_subplots(rows=1, cols=nc, shared_yaxes=True,
                        column_widths=widths, subplot_titles=titles,
                        horizontal_spacing=0.012)
    d = df['DEPTH'].values

    # GR + GR_NORM overlay
    if 'GR' in df:
        fig.add_trace(go.Scatter(x=df['GR'], y=d, mode='lines', name='GR',
                                 line=dict(color=LOG_COLORS['GR'], width=1.2)), row=1, col=1)
    if 'GR_NORM' in df and df['GR_NORM'].notna().any():
        # Second x-axis for GR_NORM (0-1 scale)
        fig.add_trace(go.Scatter(x=df['GR_NORM'], y=d, mode='lines', name='GR_NORM',
                                 line=dict(color=LOG_COLORS['GR_NORM'], width=1.2,
                                           dash='dot')), row=1, col=1)

    # RHOB + NPHI + crossover
    if 'RHOB' in df:
        fig.add_trace(go.Scatter(x=df['RHOB'], y=d, mode='lines', name='RHOB',
                                 line=dict(color=LOG_COLORS['RHOB'], width=1.2)), row=1, col=2)
    if 'NPHI' in df:
        fig.add_trace(go.Scatter(x=df['NPHI'], y=d, mode='lines', name='NPHI',
                                 line=dict(color=LOG_COLORS['NPHI'], width=1.2, dash='dot')), row=1, col=2)
        if 'RHOB' in df:
            phid = ((2.65-df['RHOB'])/1.65).values
            nphi = df['NPHI'].values
            fig.add_trace(go.Scatter(
                x=np.where(phid > nphi, phid, nphi), y=d,
                mode='lines', fill=None, line=dict(width=0),
                showlegend=False, hoverinfo='skip'), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=nphi, y=d, mode='lines', name='Crossover',
                fill='tonextx', fillcolor='rgba(240,165,0,0.2)',
                line=dict(width=0), showlegend=True), row=1, col=2)

    # RT
    if 'RT' in df:
        fig.add_trace(go.Scatter(
            x=df['RT'].clip(lower=0.001), y=d, mode='lines',
            name='RT', line=dict(color=LOG_COLORS['RT'], width=1.2)), row=1, col=3)
        fig.update_xaxes(type='log', row=1, col=3)

    # Target tracks
    for ti, tgt in enumerate(pred_tgts, start=4):
        ac, pc = TARGET_COLORS[tgt]
        has_act = tgt in df.columns and df[tgt].notna().any()
        has_pred = f'{tgt}_PRED' in df.columns and df[f'{tgt}_PRED'].notna(
        ).any()

        if has_act:
            fig.add_trace(go.Scatter(x=df[tgt], y=d, mode='lines',
                                     name=f'{tgt} Aktual', line=dict(color=ac, width=1.5)), row=1, col=ti)
        if has_pred:
            fig.add_trace(go.Scatter(x=df[f'{tgt}_PRED'], y=d, mode='lines',
                                     name=f'{tgt} Prediksi',
                                     line=dict(color=pc, width=1.5, dash='dash')), row=1, col=ti)

        if has_act and has_pred:
            r, g, b = tuple(int(pc.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            a_fill = df[tgt].fillna(method='ffill').fillna(0)
            p_fill = df[f'{tgt}_PRED'].fillna(method='ffill').fillna(0)
            fig.add_trace(go.Scatter(x=a_fill, y=d, mode='lines', fill=None,
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=ti)
            fig.add_trace(go.Scatter(x=p_fill, y=d, mode='lines',
                                     fill='tonextx', fillcolor=f'rgba({r},{g},{b},0.18)',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=ti)

    fig.update_layout(**PLOTLY_BASE, height=800,
                      title=dict(text=f'<b>{well_name}</b>',
                                 font=dict(size=13, color='#f0a500'), x=0.01),
                      legend=dict(bgcolor='rgba(13,17,23,0.85)', bordercolor='#30363d',
                                  borderwidth=1, font=dict(size=9),
                                  orientation='h', yanchor='bottom', y=1.03, xanchor='left', x=0))
    fig.update_yaxes(autorange='reversed', gridcolor='#21262d',
                     zerolinecolor='#30363d', tickfont_size=9)
    fig.update_xaxes(gridcolor='#21262d',
                     zerolinecolor='#30363d', tickfont_size=9)
    return fig


def plot_scatter(df, targets, zone_filter=None):
    """zone_filter: None atau list nama zona untuk dibatasi."""
    if zone_filter is not None and len(zone_filter) > 0 and 'ZONE' in df.columns:
        zcol = df['ZONE'].fillna('UNKNOWN').astype(str)
        df = df[zcol.isin([str(z) for z in zone_filter])].copy()
    tgts = [
        t for t in targets if t in df.columns and f'{t}_PRED' in df.columns]
    if not tgts:
        return None
    fig = make_subplots(rows=1, cols=len(tgts),
                        subplot_titles=tgts, horizontal_spacing=0.09)
    for ti, tgt in enumerate(tgts, 1):
        ac, _ = TARGET_COLORS[tgt]
        msk = df[tgt].notna() & df[f'{tgt}_PRED'].notna()

        # Policy visualisasi
        if tgt == 'VSH':
            msk = msk & (df[tgt] != 0)

        if msk.sum() < 5:
            continue

        yt = df.loc[msk, tgt].values
        yp = df.loc[msk, f'{tgt}_PRED'].values
        r2 = r2_score(yt, yp)
        fig.add_trace(go.Scatter(x=yt, y=yp, mode='markers',
                                 marker=dict(color=ac, size=3, opacity=0.45),
                                 name=tgt, showlegend=False), row=1, col=ti)
        lim = [min(yt.min(), yp.min())-.02, max(yt.max(), yp.max())+.02]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
                                 line=dict(color='#7d8590',
                                           dash='dash', width=1),
                                 showlegend=False), row=1, col=ti)
        fig.add_annotation(
            xref='x domain' if ti == 1 else f'x{ti} domain',
            yref='y domain' if ti == 1 else f'y{ti} domain',
            x=0.05, y=0.95,
            showarrow=False, text=f'R²={r2:.4f}',
            font=dict(family='IBM Plex Mono', size=11, color='#f0a500'),
            bgcolor='rgba(13,17,23,0.8)', bordercolor='#30363d')
    fig.update_layout(**PLOTLY_BASE, height=380,
                      title=dict(text='<b>Aktual vs Prediksi</b>',
                                 font=dict(color='#f0a500', size=13), x=0.01))
    fig.update_yaxes(gridcolor='#21262d')
    fig.update_xaxes(gridcolor='#21262d')
    return fig


def plot_fi(feat_imp):
    if not feat_imp:
        return None
    tgts = list(feat_imp.keys())
    fig = make_subplots(rows=1, cols=len(tgts), subplot_titles=tgts,
                        horizontal_spacing=0.1)
    pal = ['#f0a500', '#58a6ff', '#3fb950', '#bc8cff', '#f85149', '#ffd166']
    for ti, tgt in enumerate(tgts, 1):
        fi = feat_imp[tgt]
        ks = list(fi.keys())
        vs = [round(v, 1) for v in fi.values()]
        fig.add_trace(go.Bar(y=ks, x=vs, orientation='h',
                             marker_color=[pal[i % len(pal)]
                                           for i in range(len(ks))],
                             text=[f'{v:.1f}%' for v in vs], textposition='outside',
                             textfont=dict(size=9, family='IBM Plex Mono'),
                             showlegend=False), row=1, col=ti)
    ht = max(260, 50*max(len(v) for v in feat_imp.values())+80)
    fig.update_layout(**PLOTLY_BASE, height=ht,
                      title=dict(text='<b>Feature Importance</b>',
                                 font=dict(color='#f0a500', size=13), x=0.01))
    fig.update_xaxes(ticksuffix='%', gridcolor='#21262d')
    fig.update_yaxes(gridcolor='#21262d')
    return fig


# ══════════════════════════════════════════════════════════════════
# MODEL EXPORT / IMPORT
# ══════════════════════════════════════════════════════════════════

def export_model_package(res, cfg, gr_norm_params) -> bytes:
    """
    Bundle semua model + config ke satu file .pkl yang bisa di-download.
    Isi paket:
      - models: dict {target: model_object}
      - config: feature info, model params, training mode, dll.
      - gr_norm_params: parameter normalisasi GR per zona
      - le_zone: LabelEncoder zona (jika ada)
      - mnemonic_map: MNEMONIC_MAP untuk standarisasi kolom
    """
    package = {
        'version': '1.0',
        'models': {tgt: info['model'] for tgt, info in res['models'].items()},
        'feature_cols': {tgt: info['ft_tr'] for tgt, info in res['models'].items()},
        'feature_chosen': {tgt: info['ft_chosen'] for tgt, info in res['models'].items()},
        'config': {
            'targets': cfg.get('targets', []),
            'model_name': cfg.get('model_name', 'LightGBM'),
            'model_params': cfg.get('model_params', {}),
            'training_mode': cfg.get('training_mode', 'Standard ML'),
            'opts': cfg.get('opts', {}),
        },
        'gr_norm_params': gr_norm_params,
        'le_zone': res.get('le_zone'),
        'target_bounds': TARGET_BOUNDS,
        'mnemonic_map': MNEMONIC_MAP,
    }
    buf = io.BytesIO()
    pickle.dump(package, buf)
    return buf.getvalue()


def generate_model_doc(cfg, res) -> str:
    """Generate dokumentasi penggunaan model dalam format Markdown."""
    doc = """# Petro·ML — Model Documentation

## Overview
Model petrophysics yang di-train menggunakan Petro·ML Dashboard.

## Model Info
"""
    doc += f"- **Algoritma**: {cfg.get('model_name', 'LightGBM')}\n"
    doc += f"- **Mode Training**: {cfg.get('training_mode', 'Standard ML')}\n"
    doc += f"- **Target**: {', '.join(cfg.get('targets', []))}\n\n"

    doc += "## Features per Target\n"
    tf = cfg.get('target_feats', {})
    for tgt, feats in tf.items():
        doc += f"\n### {tgt}\n"
        doc += f"Features: `{'`, `'.join(feats)}`\n"

    doc += """
## Cara Penggunaan

### 1. Load Model
```python
import pickle

with open('petro_ml_model.pkl', 'rb') as f:
    package = pickle.load(f)

models = package['models']           # dict {target: sklearn/lgb model}
feature_cols = package['feature_cols']  # dict {target: [col names]}
gr_norm_params = package['gr_norm_params']
le_zone = package['le_zone']         # LabelEncoder (bisa None)
config = package['config']
```

### 2. Prepare Data
```python
import lasio
import numpy as np
import pandas as pd

# Baca LAS
las = lasio.read('well_test.las')
df = las.df().reset_index()
df.columns = [c.strip().upper() for c in df.columns]

# Rename kolom sesuai standar (gunakan package['mnemonic_map'])
mnemonic_map = package['mnemonic_map']
df.rename(columns={k: v for k, v in mnemonic_map.items() if k in df.columns}, inplace=True)

# Feature engineering
if 'RT' in df.columns:
    df['LOG_RT'] = np.where(df['RT'] > 0, np.log10(df['RT']), np.nan)

if 'RHOB' in df.columns and 'NPHI' in df.columns:
    RHO_MA, RHO_FL = 2.65, 1.00
    phid = (RHO_MA - df['RHOB']) / (RHO_MA - RHO_FL)
    df['DN_SEP'] = df['NPHI'] - phid
    df['NPHI_RHOB_CROSS'] = phid - df['NPHI']
    df['CROSS_POS'] = df['NPHI_RHOB_CROSS'].clip(lower=0)
```

### 3. GR Normalization
```python
# Gunakan gr_norm_params dari package
for zone, p in gr_norm_params.items():
    rng = p['p_high'] - p['p_low']
    if rng > 0:
        mask = df['ZONE'] == zone  # atau semua baris jika UNKNOWN
        df.loc[mask, 'GR_NORM'] = ((df.loc[mask, 'GR'] - p['p_low']) / rng).clip(0, 1)
```

### 4. Predict (urutan: VSH → PHIE → SW)
```python
for target in ['VSH', 'PHIE', 'SW']:
    if target not in models:
        continue
    model = models[target]
    feats = feature_cols[target]

    # Pastikan semua feature tersedia
    mask = df[feats].notna().all(axis=1)
    if mask.sum() > 0:
        pred = model.predict(df.loc[mask, feats])
        bounds = package['target_bounds'][target]
        pred = np.clip(pred, bounds[0], bounds[1])
        df.loc[mask, f'{target}_PRED'] = pred

        # Penting: VSH_PRED/PHIE_PRED dipakai sebagai feature target berikutnya
```

### 5. Hybrid VSH Residual (jika digunakan)
```python
if config['training_mode'] == 'Hybrid Residual VSH (Linear)':
    # Model memprediksi residual, bukan nilai absolut
    # VSH_PRED = VSH_LINEAR + model.predict(X)
    raw_pred = models['VSH'].predict(df.loc[mask, feats])
    df.loc[mask, 'VSH_PRED'] = np.clip(df.loc[mask, 'VSH_LINEAR'] + raw_pred, 0, 1)
```

## Notes
- Urutan prediksi penting: VSH → PHIE → SW (cascading features)
- VSH_PRED dipakai sebagai input untuk PHIE, PHIE_PRED untuk SW
- NULL values: -9999.25, -999.25, dll harus di-replace ke NaN sebelum prediksi
- RT <= 0 harus di-set ke NaN

## Multi-Structure: Kombinasi ZONE
Jika model di-train dengan multi-structure, ZONE dikombinasikan dengan
nama struktur agar zona dengan nama sama di struktur berbeda diperlakukan
terpisah oleh model.

Format: `ZONE_STRUCTURE` — contoh: `Upper TAF_BN`, `BRF_Tanjung Tiga Barat`

Saat prediksi well baru, pastikan kolom ZONE dikombinasikan:
```python
# Jika punya kolom STRUCTURE dan ZONE:
df['ZONE'] = df['ZONE'] + '_' + df['STRUCTURE']
# Contoh: "Upper TAF" + "BN" → "Upper TAF_BN"
```
"""
    return doc


def export_single_target_package(target, res, cfg, gr_norm_params) -> bytes:
    """
    Export model untuk satu target saja (VSH, PHIE, atau SW).
    Termasuk info cascading dependency.
    """
    info = res['models'][target]

    # Tentukan cascading dependency
    cascade_order = ['VSH', 'PHIE', 'SW']
    tgt_idx = cascade_order.index(target) if target in cascade_order else 0
    depends_on = [t for t in cascade_order[:tgt_idx] if t in res['models']]

    package = {
        'version': '1.0',
        'target': target,
        'model': info['model'],
        'feature_cols': info['ft_tr'],
        'feature_chosen': info['ft_chosen'],
        'cascading_dependency': depends_on,
        'config': {
            'target': target,
            'model_name': cfg.get('model_name', 'LightGBM'),
            'model_params': cfg.get('model_params', {}),
            'training_mode': cfg.get('training_mode', 'Standard ML'),
            'opts': cfg.get('opts', {}),
        },
        'gr_norm_params': gr_norm_params,
        'le_zone': res.get('le_zone'),
        'target_bounds': {target: TARGET_BOUNDS.get(target, (0, 1))},
        'mnemonic_map': MNEMONIC_MAP,
    }
    buf = io.BytesIO()
    pickle.dump(package, buf)
    return buf.getvalue()


def generate_single_target_doc(target, res, cfg) -> str:
    """
    Generate dokumentasi untuk model satu target.
    Berisi: input log yang diperlukan, feature engineering, cara prediksi.
    """
    info = res['models'][target]
    feats = info['ft_tr']

    cascade_order = ['VSH', 'PHIE', 'SW']
    tgt_idx = cascade_order.index(target) if target in cascade_order else 0
    depends_on = [t for t in cascade_order[:tgt_idx] if t in res['models']]

    # Identifikasi raw logs yang dibutuhkan dari features
    raw_logs_needed = set()
    derived_feats = []
    for f in feats:
        fu = f.upper()
        if fu in ('GR', 'GR_NORM'):
            raw_logs_needed.add('GR')
        elif fu in ('NPHI',):
            raw_logs_needed.add('NPHI')
        elif fu in ('RHOB',):
            raw_logs_needed.add('RHOB')
        elif fu in ('RT', 'LOG_RT'):
            raw_logs_needed.add('RT')
        elif fu in ('VSH_LINEAR',):
            raw_logs_needed.add('GR')
        elif fu in ('DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS'):
            raw_logs_needed.add('NPHI')
            raw_logs_needed.add('RHOB')
            derived_feats.append(f)
        elif fu in ('ZONE_ENC',):
            raw_logs_needed.add('ZONE (encoded)')
        elif fu in ('VSH', 'VSH_PRED'):
            if 'VSH' in depends_on:
                raw_logs_needed.add('VSH_PRED (dari model VSH)')
            else:
                raw_logs_needed.add('VSH')
        elif fu in ('PHIE', 'PHIE_PRED'):
            if 'PHIE' in depends_on:
                raw_logs_needed.add('PHIE_PRED (dari model PHIE)')
            else:
                raw_logs_needed.add('PHIE')
        else:
            raw_logs_needed.add(f)

    doc = f"""# Petro·ML — Model {target} Documentation

## Target
**{target}** — {_target_desc(target)}

## Model Info
- **Algoritma**: {cfg.get('model_name', 'LightGBM')}
- **Mode Training**: {cfg.get('training_mode', 'Standard ML')}
- **Jumlah Feature**: {len(feats)}
- **Bounds Output**: {TARGET_BOUNDS.get(target, (0, 1))}

## Input Log yang Diperlukan (Raw)
Berikut kurva log mentah yang **harus tersedia** di file LAS:

| Log | Deskripsi |
|-----|-----------|
"""
    for log in sorted(raw_logs_needed):
        doc += f"| `{log}` | {_log_desc(log)} |\n"

    if depends_on:
        doc += f"""
## Cascading Dependency
Model ini **membutuhkan hasil prediksi** dari model sebelumnya:

"""
        for dep in depends_on:
            doc += f"- **{dep}_PRED** harus dihitung terlebih dahulu (jalankan model {dep})\n"
        doc += f"""
**Urutan prediksi wajib**: {' → '.join(depends_on + [target])}
"""

    doc += f"""
## Features yang Digunakan Model
```
{chr(10).join(feats)}
```

## Feature Engineering yang Diperlukan
```python
import numpy as np

# 1. Standarisasi nama kolom (rename sesuai MNEMONIC_MAP)
# Contoh: 'ILD' → 'RT', 'CNPHI' → 'NPHI', dll.

# 2. LOG_RT (jika dibutuhkan)
"""
    if 'LOG_RT' in feats:
        doc += "df['LOG_RT'] = np.where(df['RT'] > 0, np.log10(df['RT']), np.nan)\n"
    else:
        doc += "# LOG_RT tidak dibutuhkan untuk model ini\n"

    doc += "\n# 3. DN_SEP, NPHI_RHOB_CROSS, CROSS_POS (jika dibutuhkan)\n"
    if any(f in feats for f in ['DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS']):
        doc += """RHO_MA, RHO_FL = 2.65, 1.00
phid = (RHO_MA - df['RHOB']) / (RHO_MA - RHO_FL)
df['DN_SEP'] = df['NPHI'] - phid
df['NPHI_RHOB_CROSS'] = phid - df['NPHI']
df['CROSS_POS'] = df['NPHI_RHOB_CROSS'].clip(lower=0)
"""
    else:
        doc += "# Tidak dibutuhkan untuk model ini\n"

    doc += "\n# 4. VSH_LINEAR (jika dibutuhkan)\n"
    if 'VSH_LINEAR' in feats:
        doc += """gr_min = df['GR'].quantile(0.05)
gr_max = df['GR'].quantile(0.95)
df['VSH_LINEAR'] = ((df['GR'] - gr_min) / (gr_max - gr_min)).clip(0, 1)
"""
    else:
        doc += "# Tidak dibutuhkan untuk model ini\n"

    doc += "\n# 5. GR_NORM (jika dibutuhkan)\n"
    if 'GR_NORM' in feats:
        doc += """# Gunakan gr_norm_params dari package
for zone, p in gr_norm_params.items():
    rng = p['p_high'] - p['p_low']
    if rng > 0:
        mask = True  # atau filter by zone
        df.loc[mask, 'GR_NORM'] = ((df.loc[mask, 'GR'] - p['p_low']) / rng).clip(0, 1)
"""
    else:
        doc += "# Tidak dibutuhkan untuk model ini\n"

    doc += f"""```

## Cara Prediksi
```python
import pickle
import numpy as np

with open('petro_ml_{target.lower()}_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

model = pkg['model']
feature_cols = pkg['feature_cols']
bounds = pkg['target_bounds']['{target}']

# Pastikan semua feature tersedia
mask = df[feature_cols].notna().all(axis=1)
pred = model.predict(df.loc[mask, feature_cols])
df.loc[mask, '{target}_PRED'] = np.clip(pred, bounds[0], bounds[1])
```
"""

    if cfg.get('training_mode', '') == 'Hybrid Residual VSH (Linear)' and target == 'VSH':
        doc += """
## Note: Hybrid Residual Mode
Model ini memprediksi **residual** (selisih dari VSH_LINEAR), bukan nilai absolut.
```python
raw_pred = model.predict(df.loc[mask, feature_cols])
df.loc[mask, 'VSH_PRED'] = np.clip(df.loc[mask, 'VSH_LINEAR'] + raw_pred, 0, 1)
```
"""

    doc += f"""
## Checklist Migrasi
- [ ] File LAS tersedia dengan kurva: {', '.join(sorted(raw_logs_needed))}
- [ ] Null values (-999.25 dll) sudah di-replace ke NaN
- [ ] RT <= 0 sudah di-set ke NaN
- [ ] Feature engineering sudah dilakukan (lihat di atas)
"""
    if 'GR_NORM' in feats:
        doc += "- [ ] GR normalisasi sudah dilakukan (gunakan gr_norm_params)\n"
    if depends_on:
        doc += f"- [ ] Model {', '.join(depends_on)} sudah dijalankan terlebih dahulu\n"
    if 'ZONE_ENC' in feats:
        doc += "- [ ] Kolom ZONE tersedia dan di-encode dengan LabelEncoder dari package\n"

    return doc


def _target_desc(target):
    descs = {
        'VSH': 'Volume Shale (fraksi, 0-1)',
        'PHIE': 'Effective Porosity (fraksi, 0-0.3)',
        'SW': 'Water Saturation (fraksi, 0-1)',
    }
    return descs.get(target, target)


def _log_desc(log):
    descs = {
        'GR': 'Gamma Ray',
        'NPHI': 'Neutron Porosity',
        'RHOB': 'Bulk Density',
        'RT': 'Deep Resistivity',
        'VSH': 'Volume Shale (dari log/interpretasi)',
        'PHIE': 'Effective Porosity (dari log/interpretasi)',
        'VSH_PRED (dari model VSH)': 'Output model VSH — harus prediksi dulu',
        'PHIE_PRED (dari model PHIE)': 'Output model PHIE — harus prediksi dulu',
        'ZONE (encoded)': 'Zone/Formation name (LabelEncoded)',
    }
    return descs.get(log, log)


def predict_with_package(package, df_input, zone_df=None, marker_df=None):
    """
    Jalankan prediksi menggunakan model package pada DataFrame input.
    Return: DataFrame dengan kolom _PRED ditambahkan.
    """
    models = package['models']
    feature_cols = package['feature_cols']
    gr_params = package.get('gr_norm_params', {})
    le_zone = package.get('le_zone')
    config = package.get('config', {})
    opts = config.get('opts', {})
    training_mode = config.get('training_mode', 'Standard ML')

    df = df_input.copy()

    # Merge zone/marker if provided
    if zone_df is not None or marker_df is not None:
        df = merge_zone_marker(df, zone_df, marker_df)

    # Kombinasi ZONE + STRUCTURE (jika multi-structure)
    # Format sama seperti training: "Upper TAF_BN"
    if 'STRUCTURE' in df.columns and 'ZONE' in df.columns:
        df['ZONE_ORIGINAL'] = df['ZONE']
        df['ZONE'] = df.apply(
            lambda r: (f"{r['ZONE']}_{r['STRUCTURE']}"
                       if (pd.notna(r['ZONE'])
                           and str(r['ZONE']).upper()
                           not in ('UNKNOWN', 'NAN', '', 'NONE')
                           and pd.notna(r['STRUCTURE']))
                       else r['ZONE']),
            axis=1)

    # GR Normalization — skip jika GR_NORM sudah ada dan valid
    _has_gr_norm = ('GR_NORM' in df.columns and df['GR_NORM'].notna().any())
    if gr_params and 'GR' in df.columns and not _has_gr_norm:
        # Multi-structure: params bisa berupa dict per struktur
        if isinstance(gr_params, dict) and 'STRUCTURE' in df.columns:
            # Cek apakah params per-structure (key = nama struktur, value = dict)
            first_val = next(iter(gr_params.values()), None)
            if isinstance(first_val, dict) and 'global_new' in first_val:
                # Per-structure params
                dfs_n = []
                for sn, sg in df.groupby('STRUCTURE'):
                    sp = gr_params.get(sn, gr_params.get(str(sn).upper(), {}))
                    if sp:
                        dfs_n.append(apply_gr_norm(sg, sp))
                    else:
                        dfs_n.append(sg)
                df = pd.concat(dfs_n, ignore_index=True)
            else:
                df = apply_gr_norm(df, gr_params)
        else:
            df = apply_gr_norm(df, gr_params)

    # Feature engineering
    df = build_vsh_linear_feature(df)

    if 'RT' in df.columns:
        df['LOG_RT'] = np.where(df['RT'] > 0, np.log10(df['RT']), np.nan)

    if 'RHOB' in df.columns and 'NPHI' in df.columns:
        RHO_MA, RHO_FL = 2.65, 1.00
        phid = (RHO_MA - df['RHOB']) / (RHO_MA - RHO_FL)
        if opts.get('use_dn_sep'):
            df['DN_SEP'] = df['NPHI'] - phid
        if opts.get('use_crossover'):
            df['NPHI_RHOB_CROSS'] = phid - df['NPHI']
            df['CROSS_POS'] = df['NPHI_RHOB_CROSS'].clip(lower=0)

    # Zone encoding
    if opts.get('use_zone') and le_zone is not None and 'ZONE' in df.columns:
        zone_clean = df['ZONE'].fillna('UNKNOWN').astype(str).str.strip()
        known = set(le_zone.classes_)
        zone_known = zone_clean.apply(lambda x: x if x in known else np.nan)
        df['ZONE_ENC'] = np.nan
        mask_zone = zone_known.notna()
        if mask_zone.any():
            df.loc[mask_zone, 'ZONE_ENC'] = le_zone.transform(
                zone_known.loc[mask_zone])

    # Init prediction columns
    for col in ['VSH_PRED', 'PHIE_PRED', 'SW_PRED']:
        if col not in df.columns:
            df[col] = np.nan

    # Predict in order: VSH → PHIE → SW (cascading)
    ordered = [t for t in ['VSH', 'PHIE', 'SW'] if t in models]
    for tgt in ordered:
        model = models[tgt]
        feats = feature_cols[tgt]
        missing = [f for f in feats if f not in df.columns]
        available = [f for f in feats if f in df.columns]
        if not available or len(missing) > 0:
            continue

        mask = df[feats].notna().all(axis=1)
        if 'ZONE_IS_VALID' not in df.columns:
            zone_s = df['ZONE'].fillna('UNKNOWN').astype(str).str.strip(
            ) if 'ZONE' in df.columns else pd.Series('UNKNOWN', index=df.index)
            df['ZONE_IS_VALID'] = ~zone_s.str.upper().isin(
                ['UNKNOWN', 'NAN', '', 'NONE'])
        if 'ZONE_ENC' in feats:
            mask = mask & df['ZONE_IS_VALID']

        if mask.sum() > 0:
            raw_pred = model.predict(df.loc[mask, feats])
            if training_mode == 'Hybrid Residual VSH (Linear)' and tgt == 'VSH' and 'VSH_LINEAR' in df.columns:
                df.loc[mask, 'VSH_PRED'] = clip_target_predictions(
                    df.loc[mask, 'VSH_LINEAR'].values + raw_pred, tgt)
            else:
                df.loc[mask, f'{tgt}_PRED'] = clip_target_predictions(
                    raw_pred, tgt)

    return df


# ══════════════════════════════════════════════════════════════════
# ML WORKFLOW VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def render_ml_workflow(cfg=None, res=None, mode='single'):
    """Tampilkan flow diagram & ringkasan konfigurasi training yang telah dilakukan."""

    # ── CSS tambahan ──
    st.markdown("""
    <style>
    .wf-section{font-family:'IBM Plex Mono',monospace;font-size:.72rem;
      color:#7d8590;letter-spacing:.08em;text-transform:uppercase;
      margin:1.4rem 0 .45rem;}
    .wf-card{background:#161b22;border:1px solid #30363d;border-radius:8px;
      padding:14px 18px;margin-bottom:10px;}
    .wf-card b{color:#e6edf3;}
    .wf-step{display:flex;align-items:flex-start;gap:14px;
      padding:10px 0;border-bottom:1px solid #21262d;}
    .wf-step:last-child{border-bottom:none;}
    .wf-num{background:#0d1117;border:1px solid #30363d;border-radius:50%;
      width:26px;height:26px;display:flex;align-items:center;justify-content:center;
      font-size:.75rem;color:#58a6ff;flex-shrink:0;font-weight:700;}
    .wf-body{flex:1;}
    .wf-title{font-size:.9rem;font-weight:700;color:#e6edf3;margin-bottom:3px;}
    .wf-desc{font-size:.78rem;color:#8b949e;line-height:1.55;}
    .wf-badge{display:inline-block;background:#21262d;border:1px solid #30363d;
      border-radius:4px;padding:1px 7px;font-size:.7rem;color:#58a6ff;
      margin:2px 2px 0 0;font-family:'IBM Plex Mono',monospace;}
    .wf-badge.ok{border-color:#238636;color:#3fb950;}
    .wf-badge.warn{border-color:#9e6a03;color:#e3b341;}
    .wf-arrow{text-align:center;color:#30363d;font-size:1.3rem;
      line-height:1;margin:2px 0;}
    .wf-cascade{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;}
    .wf-cas-item{background:#0d1117;border:1px solid #30363d;border-radius:6px;
      padding:8px 14px;text-align:center;flex:1;min-width:80px;}
    .wf-cas-label{font-size:.7rem;color:#7d8590;font-family:'IBM Plex Mono',monospace;}
    .wf-cas-val{font-size:1rem;font-weight:700;color:#f0a500;}
    </style>""", unsafe_allow_html=True)

    is_trained = cfg is not None and res is not None

    st.markdown(
        f'<div class="wf-section">Mode: '
        f'{"Multi-Structure" if mode == "multi" else "Single Structure"}</div>',
        unsafe_allow_html=True)

    # ── BLOK STEPS ──
    tgt_list = cfg.get('targets', ['VSH', 'PHIE', 'SW']) if cfg else [
        'VSH', 'PHIE', 'SW']
    mdl_name = cfg.get('model_name', '—') if cfg else '—'
    trn_mode = cfg.get('training_mode', '—') if cfg else '—'
    opts = cfg.get('opts', {}) if cfg else {}
    n_tw = len(cfg.get('test_wells', [])) if cfg else 0
    structures = cfg.get('structures', []) if cfg else []

    def _badges(items, cls=''):
        return ''.join(f'<span class="wf-badge {cls}">{i}</span>' for i in items)

    steps = [
        {
            'icon': '📁', 'title': '01 · Input Data',
            'desc': (
                f'Upload file <b>.las</b> (format LAS 2.0/3.0) per sumur. '
                + (f'Struktur: {_badges(structures, "ok")}' if structures else
                   'Satu struktur (ZIP berisi semua LAS).')
                + '<br>Mnemonic distandarisasi otomatis (GR, NPHI, RHOB, RT, VSH, PHIE, SW, dll).'
            )
        },
        {
            'icon': '🗺️', 'title': '02 · Zone & Marker Merge',
            'desc': (
                'Zone CSV di-merge berdasarkan kedalaman (MD) ke setiap baris LAS. '
                'Hasilnya kolom <code>ZONE</code> untuk setiap sampel kedalaman. '
                + (f'Zona dikombinasi: <b>ZONE_STRUCTURE</b> (misal <code>Upper TAF_BN</code>) '
                   'agar zona yang sama di struktur berbeda diperlakukan terpisah.'
                   if mode == 'multi' else '')
            )
        },
        {
            'icon': '🧹', 'title': '03 · QC Pipeline',
            'desc': (
                'Baris dengan semua log NaN dihapus. RT ≤ 0 → NaN. '
                'Target di luar batas fisik dihapus '
                '(VSH: 0–1, PHIE: 0–0.5, SW: 0–1). '
                'Opsional: Z-Score outlier filter pada GR/NPHI/RHOB/RT.'
            )
        },
        {
            'icon': '📐', 'title': '04 · Normalisasi GR',
            'desc': (
                'GR per sumur dinormalisasi ke referensi struktur '
                'menggunakan remapping percentile OLD→NEW:<br>'
                '<code>GR_NORM = ((P_HIGH_NEW − P_LOW_NEW) / (P_HIGH_OLD − P_LOW_OLD)) '
                '× (GR − P_LOW_OLD) + P_LOW_NEW</code><br>'
                + ('GR_NORM dihitung <b>per struktur</b> secara terpisah.' if mode == 'multi'
                   else 'GR_NORM dihitung dari semua sumur dalam ZIP.')
                + ' Di-skip jika kolom GR_NORM/GRN sudah ada di LAS.'
            )
        },
        {
            'icon': '⚗️', 'title': '05 · Feature Engineering',
            'desc': (
                'Fitur turunan dihitung dari log dasar:<br>'
                '<b>VSH_LINEAR</b> = (GR_NORM − GR_MA) / (GR_SH − GR_MA) — baseline petrophysics VSH<br>'
                '<b>LOG_RT</b> = log₁₀(RT)<br>'
                '<b>DN_SEP</b> = NPHI − PHID &nbsp;·&nbsp; '
                '<b>NPHI_RHOB_CROSS</b> = PHID − NPHI &nbsp;·&nbsp; '
                '<b>CROSS_POS</b> = max(0, PHID − NPHI)<br>'
                '<b>ZONE_ENC</b> = LabelEncoder zona (fit dari data train only)'
            )
        },
        {
            'icon': '✂️', 'title': '06 · Split Train / Test',
            'desc': (
                f'Sumur test ({_badges(cfg["test_wells"], "warn") if (cfg and n_tw <= 8) else f"<b>{n_tw} sumur</b>"}) '
                f'dipisahkan dari data training. '
                f'Semua parameter preprocessing (GR_MA, GR_SH, LabelEncoder) '
                f'dihitung dari data <b>training only</b> (no leakage).'
            ) if cfg else 'Sumur test ditentukan user; preprocessing fit dari train only.'
        },
        {
            'icon': '🔁', 'title': '07 · Out-Of-Fold (OOF) Predictions',
            'desc': (
                'Prediksi training set menggunakan strategi '
                '<b>Leave-One-Well-Out (LOWO)</b>: '
                'model dilatih tanpa 1 sumur, prediksi dibuat untuk sumur tersebut. '
                'Diulang untuk semua sumur training. '
                'Hasil OOF digunakan sebagai <code>VSH_PRED</code> / <code>PHIE_PRED</code> '
                'untuk feature cascading pada target berikutnya.'
            )
        },
        {
            'icon': '🔗', 'title': '08 · Cascading Prediction',
            'desc': (
                'Target diprediksi secara berurutan sehingga prediksi sebelumnya '
                'menjadi feature untuk target berikutnya:'
            )
        },
        {
            'icon': '🤖', 'title': f'09 · Training Model — {mdl_name}',
            'desc': (
                f'Mode training: <b>{trn_mode}</b>.<br>'
                + ('Hybrid Residual: model memprediksi <b>residual</b> = VSH − VSH_LINEAR, '
                   'lalu dijumlahkan kembali ke VSH_LINEAR untuk hasil akhir.<br>'
                   if 'Hybrid' in trn_mode else '')
                + 'Features per target: '
                + ' &nbsp;|&nbsp; '.join(
                    f'<b>{t}</b>: {_badges(cfg["target_feats"].get(t,[]), "")}'
                    for t in tgt_list if cfg and cfg.get("target_feats")
                )
                if cfg and cfg.get("target_feats") else
                'Features dipilih user per target (GR, GR_NORM, NPHI, RHOB, LOG_RT, ZONE_ENC, dll).'
            )
        },
        {
            'icon': '📊', 'title': '10 · Evaluasi Model',
            'desc': (
                'Metrics dihitung pada sumur test (prediksi vs aktual): '
                '<b>R²</b>, <b>RMSE</b>, <b>MAE</b>. '
                'Tersedia breakdown per sumur dan per zona. '
                'Feature importance dari model ditampilkan untuk interpretasi.'
            )
        },
        {
            'icon': '💾', 'title': '11 · Export & Deployment',
            'desc': (
                'Model di-bundle ke file <code>.pkl</code> berisi: '
                'model object, feature list, config, GR norm params, LabelEncoder zona. '
                'Tab <b>Model Testing</b>: upload LAS baru → prediksi langsung '
                'dengan model yang sudah di-train atau dari file .pkl.'
            )
        },
    ]

    for i, s in enumerate(steps):
        is_cascade = (i == 7)  # step 08
        st.markdown(
            f'<div class="wf-card">'
            f'<div class="wf-step">'
            f'<div class="wf-num">{s["icon"]}</div>'
            f'<div class="wf-body">'
            f'<div class="wf-title">{s["title"]}</div>'
            f'<div class="wf-desc">{s["desc"]}</div>'
            + ('''
            <div class="wf-cascade" style="margin-top:10px;">
              <div class="wf-cas-item">
                <div class="wf-cas-label">Step 1</div>
                <div class="wf-cas-val">VSH</div>
                <div class="wf-cas-label" style="margin-top:4px;">GR · GR_NORM<br>NPHI · RHOB<br>VSH_LINEAR</div>
              </div>
              <div style="display:flex;align-items:center;color:#30363d;font-size:1.4rem;">→</div>
              <div class="wf-cas-item">
                <div class="wf-cas-label">Step 2</div>
                <div class="wf-cas-val">PHIE</div>
                <div class="wf-cas-label" style="margin-top:4px;">GR · NPHI · RHOB<br>+ <span style="color:#f0a500">VSH_PRED</span></div>
              </div>
              <div style="display:flex;align-items:center;color:#30363d;font-size:1.4rem;">→</div>
              <div class="wf-cas-item">
                <div class="wf-cas-label">Step 3</div>
                <div class="wf-cas-val">SW</div>
                <div class="wf-cas-label" style="margin-top:4px;">RT · NPHI · RHOB<br>+ <span style="color:#f0a500">VSH_PRED</span><br>+ <span style="color:#f0a500">PHIE_PRED</span></div>
              </div>
            </div>
            ''' if is_cascade else '')
            + '</div></div></div>',
            unsafe_allow_html=True)

    # ── Status aktual training ──
    if is_trained and res.get('metrics'):
        st.markdown('<div class="wf-section">Hasil Training Aktual</div>',
                    unsafe_allow_html=True)
        mets = res['metrics']
        html = '<div class="kpi-row">'
        for tgt, wm in mets.items():
            r2s = [m['R2'] for m in wm.values()
                   if not np.isnan(m.get('R2', float('nan')))]
            avg_r2 = np.mean(r2s) if r2s else float('nan')
            cls = r2c(avg_r2)
            r2_str = f"{avg_r2:.4f}" if not np.isnan(avg_r2) else "N/A"
            html += (f'<div class="kpi">'
                     f'<div class="kl">{tgt}</div>'
                     f'<div class="kv {cls}">{r2_str}</div>'
                     f'<div class="ks">avg R² · {len(wm)} sumur test</div></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MULTI-STRUCTURE PAGE (self-contained function)
# ══════════════════════════════════════════════════════════════════

def _run_multi_structure_page():
    """Halaman terpisah untuk training dari gabungan beberapa struktur."""
    import hashlib

    SS = st.session_state
    structs = SS['structures']  # {name: {wells, zone_df, zip_hash, zone_hash}}

    # ── Folder-based loader helper ──────────────────────────────────────────
    _DATA_STRUCTS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data_structures')

    def _scan_structure_folders():
        """Kembalikan dict {nama_folder: {zip_path, zone_path, marker_path}}."""
        found = {}
        if not os.path.isdir(_DATA_STRUCTS_DIR):
            return found
        for entry in sorted(os.scandir(_DATA_STRUCTS_DIR), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            name = entry.name
            zips = sorted([f.path for f in os.scandir(entry.path)
                           if f.is_file() and f.name.lower().endswith('.zip')])
            csvs = sorted([f.path for f in os.scandir(entry.path)
                           if f.is_file() and f.name.lower().endswith('.csv')
                           and 'marker' not in f.name.lower()])
            markers = sorted([f.path for f in os.scandir(entry.path)
                              if f.is_file() and f.name.lower().endswith('.csv')
                              and 'marker' in f.name.lower()])
            if zips:
                found[name] = {
                    'zip_path': zips[0],
                    'zone_path': csvs[0] if csvs else None,
                    'marker_path': markers[0] if markers else None,
                }
        return found

    def _load_structure_from_folder(folder_entry):
        """Load satu struktur dari dict {zip_path, zone_path, marker_path}."""
        import hashlib
        zp = folder_entry['zip_path']
        with open(zp, 'rb') as f:
            zb = f.read()
        zhash = hashlib.md5(zb).hexdigest()
        wells = load_zip_cached(zb)
        zone_df = None
        zone_hash = None
        marker_df = None
        if folder_entry.get('zone_path'):
            try:
                with open(folder_entry['zone_path'], 'rb') as f:
                    zone_bytes = f.read()
                zone_df = read_zone_csv(io.BytesIO(zone_bytes))
                zone_hash = hashlib.md5(zone_bytes).hexdigest()
            except Exception:
                pass
        if folder_entry.get('marker_path'):
            try:
                marker_df = pd.read_csv(folder_entry['marker_path'])
            except Exception:
                pass
        return zb, zhash, wells, zone_df, zone_hash, marker_df

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown('<div class="sec">01 · Tambah Struktur</div>',
                    unsafe_allow_html=True)

        # ── Mode pilih: Upload atau Load dari Folder ──
        add_mode = st.radio(
            "Sumber Data",
            ["📤 Upload ZIP", "📁 Load dari Folder"],
            horizontal=True, key='ms_add_mode',
            help="'Load dari Folder' membaca dari folder data_structures/ "
                 "di direktori yang sama dengan app ini.")

        if add_mode == "📁 Load dari Folder":
            avail_folders = _scan_structure_folders()
            if not avail_folders:
                st.info(
                    f"Belum ada subfolder di `data_structures/`. "
                    f"Buat subfolder per struktur dan masukkan ZIP + CSV ke dalamnya.")
            else:
                st.markdown(f"**{len(avail_folders)} struktur tersedia:**")
                checked = {}
                for fname in avail_folders:
                    fe = avail_folders[fname]
                    has_zone = "✓ Zone" if fe['zone_path'] else "—"
                    already = fname.upper() in structs
                    label = f"{fname} {'*(loaded)*' if already else ''}"
                    checked[fname] = st.checkbox(
                        label,
                        value=already,
                        key=f'ms_folder_ck_{fname}',
                        help=f"ZIP: {os.path.basename(fe['zip_path'])} | {has_zone}")

                folder_load_btn = st.button(
                    "📥 Load Struktur Terpilih",
                    use_container_width=True, key='ms_folder_load_btn',
                    type='primary')

                if folder_load_btn:
                    selected = [fn for fn, ck in checked.items() if ck]
                    # Hapus struktur yang di-uncheck
                    to_remove = [
                        fn.upper() for fn in avail_folders
                        if not checked.get(fn, False) and fn.upper() in structs]
                    for rm in to_remove:
                        del structs[rm]

                    n_loaded = 0
                    for fname in selected:
                        sname = fname.upper()
                        fe = avail_folders[fname]
                        with st.spinner(f"Membaca {sname}..."):
                            try:
                                import hashlib
                                zb, zhash, wells, zone_df, zone_hash, marker_df = \
                                    _load_structure_from_folder(fe)
                            except Exception as ex:
                                st.error(f"❌ Gagal load {sname}: {ex}")
                                continue
                        if not wells:
                            st.error(f"❌ Tidak ada LAS valid di ZIP {sname}")
                            continue
                        # Skip jika hash sama (tidak berubah)
                        existing = structs.get(sname, {})
                        if existing.get('zip_hash') == zhash and \
                                existing.get('zone_hash') == zone_hash:
                            n_loaded += 1
                            continue
                        tagged_wells = {}
                        for wn, df in wells.items():
                            df = df.copy()
                            df['STRUCTURE'] = sname
                            tagged_wells[f"{sname}::{wn}"] = df
                        structs[sname] = {
                            'wells': tagged_wells,
                            'zone_df': zone_df,
                            'marker_df': marker_df,
                            'zip_hash': zhash,
                            'zone_hash': zone_hash,
                            'n_wells': len(wells),
                            'well_names': sorted(wells.keys()),
                            'source': 'folder',
                            'folder_name': fname,
                        }
                        n_loaded += 1

                    SS['structures'] = structs
                    SS['ms_combined_df'] = None
                    SS['ms_qc_log'] = None
                    SS['ms_normalized'] = False
                    SS['ms_gr_norm_params'] = {}
                    SS['ms_results'] = None
                    SS['ms_trained'] = False
                    st.success(
                        f"✅ {n_loaded} struktur dimuat "
                        f"({len(to_remove)} dihapus)" if to_remove
                        else f"✅ {n_loaded} struktur dimuat")
                    st.rerun()

        else:
            # ── Mode Upload (existing behavior) ──
            struct_name = st.text_input(
                "Nama Struktur", value="", key='ms_struct_name',
                help="Contoh: BN, MJ, TMB — nama field/area")

            ms_zip = st.file_uploader("ZIP berisi LAS", type=['zip'],
                                      key='ms_zip_up')
            ms_zone = st.file_uploader("Zone CSV (opsional)", type=['csv'],
                                       key='ms_zone_up')
            ms_marker = st.file_uploader("Marker CSV (opsional)", type=['csv'],
                                         key='ms_marker_up')

            add_btn = st.button("➕  Tambah Struktur",
                                disabled=(not struct_name or not ms_zip),
                                use_container_width=True, key='ms_add_btn')

            if add_btn and struct_name and ms_zip:
                sname = struct_name.strip().upper()
                zb = ms_zip.read()
                zhash = hashlib.md5(zb).hexdigest()

                existing = structs.get(sname, {})
                if existing.get('zip_hash') == zhash:
                    st.info(f"ℹ Struktur {sname} sudah ada dengan ZIP yang sama.")
                else:
                    with st.spinner(f"Membaca LAS dari {sname}..."):
                        wells = load_zip_cached(zb)
                    if not wells:
                        st.error(f"❌ Tidak ada LAS valid di ZIP {sname}")
                    else:
                        tagged_wells = {}
                        for wn, df in wells.items():
                            df = df.copy()
                            df['STRUCTURE'] = sname
                            tagged_wells[f"{sname}::{wn}"] = df
                        entry = {
                            'wells': tagged_wells,
                            'zone_df': None,
                            'marker_df': None,
                            'zip_hash': zhash,
                            'zone_hash': None,
                            'n_wells': len(wells),
                            'well_names': sorted(wells.keys()),
                            'source': 'upload',
                        }
                        if ms_zone is not None:
                            try:
                                zone_bytes = ms_zone.read()
                                entry['zone_df'] = read_zone_csv(
                                    io.BytesIO(zone_bytes))
                                entry['zone_hash'] = hashlib.md5(
                                    zone_bytes).hexdigest()
                            except Exception as e:
                                st.warning(f"⚠ Zone CSV error untuk {sname}: {e}")
                        if ms_marker is not None:
                            try:
                                marker_bytes = ms_marker.read()
                                entry['marker_df'] = pd.read_csv(
                                    io.BytesIO(marker_bytes))
                            except Exception as e:
                                st.warning(f"⚠ Marker CSV error untuk {sname}: {e}")

                        structs[sname] = entry
                        SS['structures'] = structs
                        SS['ms_combined_df'] = None
                        SS['ms_qc_log'] = None
                        SS['ms_normalized'] = False
                        SS['ms_gr_norm_params'] = {}
                        SS['ms_results'] = None
                        SS['ms_trained'] = False
                        st.success(f"✅ {sname}: {len(wells)} sumur ditambahkan")

        # ── Daftar struktur ──
        if structs:
            st.markdown('<div class="sec">Struktur Aktif</div>',
                        unsafe_allow_html=True)

            # Tombol refresh untuk struktur dari folder (reload jika file berubah)
            folder_structs = [s for s, d in structs.items()
                              if d.get('source') == 'folder']
            if folder_structs:
                if st.button("🔄 Refresh dari Folder", use_container_width=True,
                             key='ms_refresh_folder',
                             help="Reload ulang struktur dari folder jika ada file yang diubah"):
                    avail_now = _scan_structure_folders()
                    for sname in list(structs.keys()):
                        if structs[sname].get('source') != 'folder':
                            continue
                        fname = structs[sname].get('folder_name', sname)
                        fe = avail_now.get(fname)
                        if fe is None:
                            continue
                        import hashlib as _hl
                        with open(fe['zip_path'], 'rb') as f:
                            new_zb = f.read()
                        new_zhash = _hl.md5(new_zb).hexdigest()
                        if new_zhash == structs[sname].get('zip_hash'):
                            continue
                        wells = load_zip_cached(new_zb)
                        if not wells:
                            continue
                        tagged = {}
                        for wn, df in wells.items():
                            df = df.copy()
                            df['STRUCTURE'] = sname
                            tagged[f"{sname}::{wn}"] = df
                        zone_df, zone_hash, marker_df = None, None, None
                        if fe.get('zone_path'):
                            try:
                                with open(fe['zone_path'], 'rb') as f:
                                    zb2 = f.read()
                                zone_df = read_zone_csv(io.BytesIO(zb2))
                                zone_hash = _hl.md5(zb2).hexdigest()
                            except Exception:
                                pass
                        if fe.get('marker_path'):
                            try:
                                marker_df = pd.read_csv(fe['marker_path'])
                            except Exception:
                                pass
                        structs[sname].update({
                            'wells': tagged, 'zone_df': zone_df,
                            'marker_df': marker_df, 'zip_hash': new_zhash,
                            'zone_hash': zone_hash,
                            'n_wells': len(wells),
                            'well_names': sorted(wells.keys()),
                        })
                    SS['structures'] = structs
                    SS['ms_combined_df'] = None
                    SS['ms_trained'] = False
                    st.success("✅ Folder struktur di-refresh")
                    st.rerun()

            for sname, sdata in structs.items():
                c1, c2 = st.columns([4, 1])
                with c1:
                    wells_str = ', '.join(sdata.get('well_names', [])[:5])
                    if len(sdata.get('well_names', [])) > 5:
                        wells_str += f' +{len(sdata["well_names"])-5}'
                    zone_tag = '✓ Zone' if sdata.get(
                        'zone_df') is not None else '—'
                    src_icon = '📁' if sdata.get('source') == 'folder' else '📤'
                    st.markdown(
                        f'<div class="ibox" style="padding:5px 10px;">'
                        f'{src_icon} <b>{sname}</b> · {sdata["n_wells"]} sumur · {zone_tag}<br>'
                        f'<span style="font-size:.65rem;color:var(--muted2);">{wells_str}</span>'
                        f'</div>', unsafe_allow_html=True)
                with c2:
                    if st.button("🗑", key=f'ms_del_{sname}',
                                 help=f"Hapus {sname}"):
                        del structs[sname]
                        SS['structures'] = structs
                        SS['ms_combined_df'] = None
                        SS['ms_trained'] = False
                        st.rerun()

        # ── Total summary ──
        all_ms_wells = {}
        for sdata in structs.values():
            all_ms_wells.update(sdata.get('wells', {}))

        n_total_wells = len(all_ms_wells)
        n_structs = len(structs)

        if n_total_wells > 0:
            st.markdown(
                f'<div class="ibox" style="border-left-color:var(--a3);">'
                f'<b>{n_structs}</b> struktur · <b>{n_total_wells}</b> sumur total'
                f'</div>', unsafe_allow_html=True)

        # ── Build combined ──
        # PENTING: ZONE dikombinasi dengan STRUCTURE agar zona dengan nama
        # yang sama di struktur berbeda (misal "Upper TAF" di BN vs MJ)
        # diperlakukan sebagai zona terpisah oleh model.
        # Format: "BRF_Tanjung Tiga Barat", "Upper TAF_BN"
        # Saat prediksi, user cukup punya kolom STRUCTURE + ZONE lalu
        # dikombinasi menjadi ZONE_STRUCT = ZONE + "_" + STRUCTURE.
        ms_combined_raw = None
        if all_ms_wells:
            dfs = []
            for sname, sdata in structs.items():
                zone_df = sdata.get('zone_df')
                marker_df = sdata.get('marker_df')
                for wkey, wdf in sdata.get('wells', {}).items():
                    merged = merge_zone_marker(wdf, zone_df, marker_df)
                    # Kombinasi: "Upper TAF" + "BN" → "Upper TAF_BN"
                    if 'ZONE' in merged.columns:
                        merged['ZONE_ORIGINAL'] = merged['ZONE']

                        def _combine_zone_structure(z, s=sname):
                            if pd.isna(z):
                                return z

                            z_str = str(z).strip()
                            s_str = str(s).strip()

                            if z_str.upper() in ('UNKNOWN', 'NAN', '', 'NONE'):
                                return z

                            if z_str.upper().endswith(f"_{s_str.upper()}"):
                                return z_str

                            return f"{z_str}_{s_str}"

                        merged['ZONE'] = merged['ZONE'].apply(
                            _combine_zone_structure)
                    dfs.append(merged)
            if dfs:
                ms_combined_raw = pd.concat(dfs, ignore_index=True)

        # ── 02 QC ──
        st.markdown('<div class="sec">02 · QC Data</div>',
                    unsafe_allow_html=True)
        can_qc = ms_combined_raw is not None and len(ms_combined_raw) > 0
        ms_zscore = st.checkbox('Z-Score Outlier Filter', value=False,
                                key='ms_zscore')
        ms_zthr = st.select_slider('Z-Score threshold',
                                   options=[2.5, 3.0, 3.5], value=3.0,
                                   key='ms_zthr')
        ms_qc_btn = st.button("🧹  Jalankan QC",
                              disabled=not can_qc,
                              use_container_width=True, key='ms_qc_btn')
        if ms_qc_btn and can_qc:
            df_qc, qc_log = run_qc_pipeline(
                ms_combined_raw.copy(),
                use_zscore=ms_zscore, zscore_threshold=ms_zthr)
            SS['ms_combined_df'] = df_qc
            SS['ms_qc_log'] = qc_log
            SS['ms_normalized'] = False
            SS['ms_gr_norm_params'] = {}
            st.success(f"✅ QC selesai — {len(df_qc):,} baris tersisa")

        if SS.get('ms_qc_log'):
            ql = SS['ms_qc_log']
            st.markdown(
                f'<div class="ibox">'
                f'Total drop: <b>{ql.get("total_dropped",0):,}</b> · '
                f'Sisa: <b>{ql.get("remaining",0):,}</b>'
                f'</div>', unsafe_allow_html=True)

        # Active combined
        _mcd = SS.get('ms_combined_df')
        ms_combined = _mcd if (
            _mcd is not None and not _mcd.empty) else ms_combined_raw

        # ── 03 GR Norm ──
        st.markdown('<div class="sec">03 · Normalisasi GR</div>',
                    unsafe_allow_html=True)
        ms_has_gr = (ms_combined is not None and 'GR' in ms_combined.columns
                     and ms_combined['GR'].notna().any())

        # Cek apakah GR_NORM sudah ada di data LAS
        ms_has_gr_norm_direct = (
            ms_combined is not None and 'GR_NORM' in ms_combined.columns
            and ms_combined['GR_NORM'].notna().any())

        if ms_has_gr_norm_direct and not SS.get('ms_normalized'):
            _n_valid = ms_combined['GR_NORM'].notna().sum()
            st.info(
                f"ℹ GR_NORM sudah tersedia di data ({_n_valid:,} nilai valid). "
                f"Normalisasi di-skip. Klik tombol di bawah jika ingin hitung ulang.")

        mc1, mc2 = st.columns(2)
        with mc1:
            ms_plow = st.number_input("P low", 1, 20, 3, key='ms_plow')
        with mc2:
            ms_phigh = st.number_input("P high", 80, 99, 97, key='ms_phigh')
        ms_norm_btn = st.button("📐  Hitung GR_NORM",
                                disabled=not ms_has_gr,
                                use_container_width=True, key='ms_norm_btn')
        if ms_norm_btn and ms_has_gr and ms_combined is not None:
            # Per-structure normalization: hitung params per struktur
            all_params = {}
            if 'STRUCTURE' in ms_combined.columns:
                for s_name, s_grp in ms_combined.groupby('STRUCTURE'):
                    s_params = compute_gr_norm_params(
                        s_grp, pct_low_old=ms_plow, pct_high_old=97,
                        pct_low_new=ms_plow, pct_high_new=ms_phigh)
                    if s_params:
                        all_params[s_name] = s_params
                # Apply per structure
                dfs_normed = []
                for s_name, s_grp in ms_combined.groupby('STRUCTURE'):
                    if s_name in all_params:
                        dfs_normed.append(
                            apply_gr_norm(s_grp, all_params[s_name]))
                    else:
                        dfs_normed.append(s_grp)
                ms_combined = pd.concat(dfs_normed, ignore_index=True)
            else:
                # Fallback: single compute
                all_params['_ALL'] = compute_gr_norm_params(
                    ms_combined, pct_low_old=ms_plow, pct_high_old=97,
                    pct_low_new=ms_plow, pct_high_new=ms_phigh)
                ms_combined = apply_gr_norm(
                    ms_combined, all_params['_ALL'])

            SS['ms_combined_df'] = ms_combined
            SS['ms_gr_norm_params'] = all_params
            SS['ms_normalized'] = True
            st.success(f"✅ GR_NORM selesai (per struktur) — "
                       f"{ms_combined['GR_NORM'].notna().sum():,} nilai valid")

        ms_has_grn = (ms_combined is not None and 'GR_NORM' in ms_combined.columns
                      and ms_combined['GR_NORM'].notna().any())
        if SS['ms_normalized'] or ms_has_grn:
            _ms_prm = SS.get('ms_gr_norm_params', {})
            if _ms_prm:
                _struct_names = [k for k in _ms_prm if k != '_ALL']
                st.markdown(
                    f'<div class="ibox">GR_NORM aktif (per struktur) · '
                    f'{len(_struct_names)} struktur: '
                    f'{", ".join(_struct_names) if _struct_names else "ALL"}'
                    f'</div>',
                    unsafe_allow_html=True)

        # ── 04 Sumur Test ──
        st.markdown('<div class="sec">04 · Sumur Test</div>',
                    unsafe_allow_html=True)
        ms_well_list = sorted(all_ms_wells.keys()) if all_ms_wells else []
        # Tampilkan nama pendek (tanpa prefix struktur) di UI
        ms_display_names = {wk: wk.split('::', 1)[-1] for wk in ms_well_list}
        ms_test_wells = st.multiselect(
            "Pilih sumur Test",
            options=ms_well_list,
            format_func=lambda x: f"{x.split('::')[0]} / {x.split('::')[-1]}",
            key='ms_test_wells',
            help="Format: STRUKTUR / SUMUR")
        if ms_test_wells:
            n_tr = len([w for w in ms_well_list if w not in ms_test_wells])
            st.markdown(
                f'<div class="ibox">🏋 Train: <b>{n_tr}</b> · '
                f'🔬 Test: <b>{len(ms_test_wells)}</b></div>',
                unsafe_allow_html=True)

        # ── 05 Target ──
        st.markdown('<div class="sec">05 · Target Prediksi</div>',
                    unsafe_allow_html=True)
        ms_avail_tgts = ([c for c in ALL_TARGETS
                          if ms_combined is not None and c in ms_combined.columns
                          and ms_combined[c].notna().any()]
                         if ms_combined is not None else ALL_TARGETS)
        ms_sel_tgts = st.multiselect("Target", options=ms_avail_tgts,
                                     default=ms_avail_tgts, key='ms_sel_tgts')

        # ── 06 Feature per Target ──
        st.markdown('<div class="sec">06 · Feature per Target</div>',
                    unsafe_allow_html=True)
        ms_avail_base = [c for c in ALL_LOGS
                         if ms_combined is not None and len(ms_combined) > 0
                         and c in ms_combined.columns
                         and ms_combined[c].notna().any()]
        if not ms_avail_base:
            ms_avail_base = [c for c in ['GR', 'NPHI', 'RHOB', 'RT']
                             if ms_combined is not None
                             and c in (ms_combined.columns
                                       if ms_combined is not None else [])]

        ms_has_nphi_rhob = (
            'NPHI' in ms_avail_base and 'RHOB' in ms_avail_base)
        ms_derived = []
        if ms_has_nphi_rhob:
            ms_derived += ['DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS']
        ms_derived += ['ZONE_ENC']
        ms_propagated = ['VSH_PRED', 'PHIE_PRED']

        ms_gr_feat = 'GR_NORM' if 'GR_NORM' in ms_avail_base else 'GR'
        _ms_defaults = {
            'VSH': [f for f in [ms_gr_feat, 'VSH_LINEAR', 'RHOB', 'DN_SEP',
                                'NPHI_RHOB_CROSS', 'ZONE_ENC']
                    if f in ms_avail_base + ms_derived],
            'PHIE': [f for f in ['NPHI', 'RHOB', 'DN_SEP', 'NPHI_RHOB_CROSS',
                                 'CROSS_POS', 'VSH_PRED', 'ZONE_ENC']
                     if f in ms_avail_base + ms_derived + ms_propagated],
            'SW': [f for f in ['RT', 'NPHI_RHOB_CROSS', 'CROSS_POS',
                               'VSH_PRED', 'PHIE_PRED', 'ZONE_ENC']
                   if f in ms_avail_base + ms_derived + ms_propagated],
        }

        ms_target_feats = {}
        for tgt in (ms_sel_tgts or ALL_TARGETS):
            with st.expander(f"📐 {tgt} — feature", expanded=False):
                all_o = list(dict.fromkeys(
                    ms_avail_base + ms_derived + ms_propagated))
                defs = [f for f in _ms_defaults.get(tgt, ms_avail_base)
                        if f in all_o]
                chosen = st.multiselect(
                    f"Feature {tgt}", options=all_o, default=defs,
                    key=f'ms_feat_{tgt}')
                ms_target_feats[tgt] = chosen
                pills = ''.join(
                    f'<span class="pill">{f}</span>' for f in chosen)
                st.markdown(
                    f'<div class="ibox" style="margin-top:4px;">'
                    f'{pills if pills else "—"}</div>',
                    unsafe_allow_html=True)

        ms_all_chosen = [f for fs in ms_target_feats.values() for f in fs]
        ms_opts = {
            'use_dn_sep': 'DN_SEP' in ms_all_chosen,
            'use_crossover': ('NPHI_RHOB_CROSS' in ms_all_chosen
                              or 'CROSS_POS' in ms_all_chosen),
            'use_zone': 'ZONE_ENC' in ms_all_chosen,
        }

        # ── 07 Model Params ──
        st.markdown('<div class="sec">07 · Model Parameters</div>',
                    unsafe_allow_html=True)
        ms_model_choice = st.selectbox(
            "Algoritma",
            ['LightGBM', 'RandomForest', 'CatBoost', 'ANN',
             'XGBoost', 'ExtraTrees', 'Stacking Ensemble'],
            index=0, key='ms_model_choice')

        ms_training_mode = st.selectbox(
            'Mode Training',
            ['Standard ML', 'Hybrid Residual VSH (Linear)'],
            index=0, key='ms_train_mode')

        if ms_model_choice == 'LightGBM':
            with st.expander("⚙ LightGBM"):
                _ne = st.slider("n_estimators", 100, 2000, 800, 100,
                                key='ms_ne')
                _lr = st.select_slider(
                    "learning_rate",
                    [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                    value=0.02, key='ms_lr')
                _nl = st.slider("num_leaves", 15, 255, 127, 16, key='ms_nl')
                _mc = st.slider("min_child_samples", 5, 100, 10, 5,
                                key='ms_mc')
                _ss = st.slider("subsample", 0.4, 1.0, 0.7, 0.1,
                                key='ms_ss')
                _cb = st.slider("colsample_bytree", 0.4, 1.0, 0.7, 0.1,
                                key='ms_cb')
                _ra = st.number_input("reg_alpha", 0.0, 10.0, 0.05, 0.01,
                                      key='ms_ra')
                _rl = st.number_input("reg_lambda", 0.0, 10.0, 0.50, 0.10,
                                      key='ms_rl')
            ms_model_params = dict(
                n_estimators=_ne, learning_rate=_lr, num_leaves=_nl,
                min_child_samples=_mc, subsample=_ss, colsample_bytree=_cb,
                reg_alpha=_ra, reg_lambda=_rl, random_state=42, n_jobs=-1,
                verbose=-1)
            ms_model_name = 'lightgbm'
        elif ms_model_choice == 'RandomForest':
            with st.expander("⚙ RandomForest"):
                _ne = st.slider("n_estimators", 100, 2000, 600, 100,
                                key='ms_rf_ne')
                _md = st.slider("max_depth", 3, 40, 14, 1, key='ms_rf_md')
                _msl = st.slider("min_samples_leaf", 1, 20, 2, 1,
                                 key='ms_rf_msl')
                _mss = st.slider("min_samples_split", 2, 20, 4, 1,
                                 key='ms_rf_mss')
                _mf = st.selectbox("max_features", ['sqrt', 'log2', None],
                                   index=0, key='ms_rf_mf')
            ms_model_params = dict(
                n_estimators=_ne, max_depth=_md, min_samples_leaf=_msl,
                min_samples_split=_mss, max_features=_mf,
                random_state=42, n_jobs=-1)
            ms_model_name = 'randomforest'
        elif ms_model_choice == 'ANN':
            with st.expander("⚙ ANN / MLP"):
                _hidden = st.selectbox(
                    'hidden_layer_sizes',
                    ['64', '128', '64-32', '128-64', '128-64-32'],
                    index=3, key='ms_ann_h')
                _alpha = st.number_input(
                    'alpha', 0.00001, 0.1, 0.0005, 0.0001,
                    format='%.5f', key='ms_ann_a')
                _lri = st.select_slider(
                    'learning_rate_init',
                    [0.0005, 0.001, 0.003, 0.005, 0.01],
                    value=0.001, key='ms_ann_lr')
                _act = st.selectbox('activation', ['relu', 'tanh'],
                                    index=0, key='ms_ann_act')
                _mi = st.slider('max_iter', 200, 3000, 1200, 100,
                                key='ms_ann_mi')
            _hmap = {'64': (64,), '128': (128,), '64-32': (64, 32),
                     '128-64': (128, 64), '128-64-32': (128, 64, 32)}
            ms_model_params = dict(
                hidden_layer_sizes=_hmap[_hidden], activation=_act,
                solver='adam', alpha=float(_alpha),
                learning_rate_init=float(_lri), max_iter=int(_mi),
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=42)
            ms_model_name = 'ann'
        elif ms_model_choice == 'CatBoost':
            with st.expander("⚙ CatBoost"):
                _ne = st.slider("iterations", 100, 2000, 800, 100,
                                key='ms_cb_it')
                _lr = st.select_slider(
                    "learning_rate",
                    [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                    value=0.03, key='ms_cb_lr')
                _dep = st.slider("depth", 3, 10, 6, 1, key='ms_cb_dep')
                _l2 = st.number_input("l2_leaf_reg", 1.0, 20.0, 3.0, 0.5,
                                      key='ms_cb_l2')
            ms_model_params = dict(
                iterations=_ne, learning_rate=_lr, depth=_dep,
                l2_leaf_reg=_l2, loss_function='RMSE',
                eval_metric='RMSE', random_seed=42, verbose=0)
            ms_model_name = 'catboost'
        elif ms_model_choice == 'XGBoost':
            with st.expander("⚙ XGBoost"):
                _ne = st.slider("n_estimators", 100, 2000, 800, 100,
                                key='ms_xgb_ne')
                _lr = st.select_slider("learning_rate",
                                       [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                                       value=0.03, key='ms_xgb_lr')
                _dep = st.slider("max_depth", 3, 10, 6, 1, key='ms_xgb_dep')
                _mcw = st.slider("min_child_weight", 1, 20, 5, 1,
                                 key='ms_xgb_mcw')
                _ss = st.slider("subsample", 0.4, 1.0, 0.8, 0.1,
                                key='ms_xgb_ss')
                _cb = st.slider("colsample_bytree", 0.4, 1.0, 0.8, 0.1,
                                key='ms_xgb_cb')
                _ra = st.number_input("reg_alpha", 0.0, 10.0, 0.1, 0.05,
                                      key='ms_xgb_ra')
                _rl = st.number_input("reg_lambda", 0.0, 10.0, 1.0, 0.1,
                                      key='ms_xgb_rl')
            ms_model_params = dict(
                n_estimators=_ne, learning_rate=_lr, max_depth=_dep,
                min_child_weight=_mcw, subsample=_ss, colsample_bytree=_cb,
                reg_alpha=_ra, reg_lambda=_rl,
                tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
            ms_model_name = 'xgboost'
        elif ms_model_choice == 'ExtraTrees':
            with st.expander("⚙ ExtraTrees"):
                _ne = st.slider("n_estimators", 100, 2000, 600, 100,
                                key='ms_et_ne')
                _md = st.slider("max_depth", 3, 40, 14, 1, key='ms_et_md')
                _msl = st.slider("min_samples_leaf", 1, 20, 2, 1,
                                 key='ms_et_msl')
                _mss = st.slider("min_samples_split", 2, 20, 4, 1,
                                 key='ms_et_mss')
                _mf = st.selectbox("max_features", ['sqrt', 'log2', None],
                                   index=0, key='ms_et_mf')
            ms_model_params = dict(
                n_estimators=_ne, max_depth=_md, min_samples_leaf=_msl,
                min_samples_split=_mss, max_features=_mf,
                random_state=42, n_jobs=-1)
            ms_model_name = 'extratrees'
        else:  # Stacking Ensemble
            with st.expander("⚙ Stacking Ensemble"):
                st.caption(
                    "Menggabungkan LightGBM + RF (+ CatBoost jika tersedia) "
                    "dengan Ridge meta-learner. Lebih lambat tapi akurasi lebih baik.")
                _st_lgb_ne = st.slider("LightGBM n_estimators", 200, 1000,
                                       500, 100, key='ms_st_lgb_ne')
                _st_lgb_lr = st.select_slider("LightGBM learning_rate",
                                              [0.01, 0.02, 0.03, 0.05, 0.08],
                                              value=0.03, key='ms_st_lgb_lr')
                _st_rf_ne = st.slider("RF n_estimators", 100, 800, 300, 100,
                                      key='ms_st_rf_ne')
                _st_ridge_a = st.number_input("Ridge alpha", 0.01, 10.0,
                                              1.0, 0.1, key='ms_st_ridge_a')
                _st_cv = st.slider("CV folds (internal)", 3, 7, 5, 1,
                                   key='ms_st_cv')
            ms_model_params = dict(
                lgb_n_estimators=_st_lgb_ne, lgb_learning_rate=_st_lgb_lr,
                rf_n_estimators=_st_rf_ne, ridge_alpha=_st_ridge_a,
                cv=_st_cv, n_jobs=-1)
            ms_model_name = 'stacking'

        # ── Per-Structure Weighting ──
        ms_use_weight = st.checkbox(
            "Pembobotan per Struktur",
            value=False, key='ms_use_weight',
            help="Berikan bobot lebih tinggi pada struktur dengan data lebih sedikit "
                 "agar kontribusi setiap struktur seimbang saat training.")

        # ── Target Rules ──
        with st.expander("📋 Target Rules (Training)", expanded=False):
            st.caption(
                "Rule membatasi data training per target. "
                "Nonaktifkan jika ingin model tetap belajar dari nilai ekstrem.")
            ms_rule_vsh = st.checkbox(
                "Buang VSH = 0 (coal marker)",
                value=True, key='ms_rule_vsh_drop_zero')
            ms_rule_sw = st.checkbox(
                "Buang SW = 1",
                value=False, key='ms_rule_sw_drop_one')
            ms_rule_phie = st.checkbox(
                "Buang PHIE = 0",
                value=False, key='ms_rule_phie_drop_zero')
        ms_target_rules = {
            'rule_vsh_drop_zero': ms_rule_vsh,
            'rule_sw_drop_one': ms_rule_sw,
            'rule_phie_drop_zero': ms_rule_phie,
        }

        # ── Train Button ──
        st.markdown("---")
        ms_can_train = (ms_combined is not None and len(ms_combined) > 0
                        and len(ms_test_wells) > 0
                        and len(ms_sel_tgts) > 0
                        and any(len(v) > 0 for v in ms_target_feats.values())
                        and not (ms_model_choice == 'XGBoost'
                                 and XGBRegressor is None))
        ms_train_btn = st.button("▶  Jalankan Training (Multi)",
                                 disabled=not ms_can_train,
                                 use_container_width=True,
                                 key='ms_train_btn')
        if not ms_can_train:
            miss = []
            if not structs:
                miss.append("tambah struktur")
            if not ms_test_wells:
                miss.append("pilih test well")
            if not ms_sel_tgts:
                miss.append("pilih target")
            if miss:
                st.markdown(
                    f'<div class="ibox wbox">Perlu: {" · ".join(miss)}</div>',
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TRAINING TRIGGER (Multi-Structure)
    # ══════════════════════════════════════════════════════════════
    if ms_train_btn and ms_can_train:
        _train_df = ms_combined.copy()
        if 'STRUCTURE' in _train_df.columns:
            _train_df['_ORIG_WELL'] = _train_df['WELL_NAME']
            _train_df['WELL_NAME'] = (
                _train_df['STRUCTURE'].astype(str) + '::' +
                _train_df['WELL_NAME'].astype(str)
            )

        _ms_sw = 'structure' if ms_use_weight else None
        with st.spinner("Melatih model multi-struktur..."):
            ms_res = run_training(
                _train_df, ms_test_wells, ms_target_feats, ms_opts,
                ms_model_params, model_name=ms_model_name,
                training_mode=(
                    'hybrid_vsh_linear_residual'
                    if ms_training_mode == 'Hybrid Residual VSH (Linear)'
                    else 'standard'),
                sample_weight=_ms_sw,
                target_rules=ms_target_rules)

        if ms_res is not None:
            SS.update(
                ms_results=ms_res, ms_trained=True,
                ms_cfg=dict(
                    test_wells=ms_test_wells,
                    targets=ms_sel_tgts,
                    target_feats=ms_target_feats,
                    opts=ms_opts,
                    model_name=ms_model_choice,
                    model_params=ms_model_params,
                    training_mode=ms_training_mode,
                    structures=list(structs.keys()),
                ))
            st.success(
                f"✅ Training multi-struktur selesai! "
                f"({n_structs} struktur, {n_total_wells} sumur)")
        else:
            SS['ms_trained'] = False

    # ══════════════════════════════════════════════════════════════
    # HEADER (Multi-Structure)
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="logo">Petro·ML</div>'
        '<div class="logo-sub" style="margin-bottom:.55rem;">'
        'Multi-Structure Training Dashboard</div><hr>',
        unsafe_allow_html=True)

    # ── Data Summary ──
    if ms_combined is not None and len(ms_combined) > 0:
        html = '<div class="kpi-row">'
        html += (f'<div class="kpi"><div class="kl">Struktur</div>'
                 f'<div class="kv ok">{n_structs}</div>'
                 f'<div class="ks">{", ".join(structs.keys())}</div></div>')
        html += (f'<div class="kpi"><div class="kl">Sumur</div>'
                 f'<div class="kv ok">{n_total_wells}</div>'
                 f'<div class="ks">total</div></div>')
        html += (f'<div class="kpi"><div class="kl">Baris</div>'
                 f'<div class="kv">{len(ms_combined):,}</div>'
                 f'<div class="ks">setelah QC</div></div>')
        qc_ok = bool(SS.get('ms_qc_log'))
        html += (f'<div class="kpi"><div class="kl">QC</div>'
                 f'<div class="kv {"good" if qc_ok else "na"}">'
                 f'{"✓" if qc_ok else "—"}</div></div>')
        norm_ok = SS.get('ms_normalized', False) or (
            'GR_NORM' in ms_combined.columns
            and ms_combined['GR_NORM'].notna().any())
        html += (f'<div class="kpi"><div class="kl">GR_NORM</div>'
                 f'<div class="kv {"good" if norm_ok else "na"}">'
                 f'{"✓" if norm_ok else "—"}</div></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        with st.expander("📋 Ringkasan per Sumur"):
            _sum_df = ms_combined.copy()
            if 'STRUCTURE' in _sum_df.columns:
                _sum_df['_DISPLAY'] = (_sum_df['STRUCTURE'].astype(str)
                                       + ' / ' + _sum_df['WELL_NAME'].astype(str))
            else:
                _sum_df['_DISPLAY'] = _sum_df['WELL_NAME']
            agg = {'N': ('DEPTH', 'count'),
                   'Depth_Min': ('DEPTH', 'min'),
                   'Depth_Max': ('DEPTH', 'max')}
            for c in ['GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT'] + ALL_TARGETS:
                if c in _sum_df.columns:
                    agg[c] = (c, lambda x, c=c: f"{x.notna().mean()*100:.0f}%")
            summ = _sum_df.groupby('_DISPLAY').agg(**agg).reset_index()
            summ.rename(columns={'_DISPLAY': 'Struktur / Well'}, inplace=True)
            st.dataframe(summ, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # RESULT TABS (Multi-Structure)
    # ══════════════════════════════════════════════════════════════
    if SS['ms_trained'] and SS['ms_results']:
        ms_res = SS['ms_results']
        ms_cfg = SS['ms_cfg']

        if 'df_te' not in ms_res or 'df_tr' not in ms_res:
            st.warning("⚠ Hasil training tidak lengkap.")
            return

        df_te = ms_res['df_te'] if ms_res['df_te'] is not None else pd.DataFrame()
        df_tr = ms_res['df_tr']

        mt1, mt2, mt3, mt4, mt5, mt6, mt7 = st.tabs([
            "📊 Metrics & Importance",
            "🪵 Log Plot",
            "🎯 Scatter Plot",
            "💾 Export",
            "📦 Download Model",
            "🧪 Model Testing",
            "📋 ML Workflow",
        ])

        # ── Metrics ──
        with mt1:
            st.markdown("### Model Performance (Multi-Structure)")
            mets = ms_res['metrics']
            if mets:
                audit = ms_res.get('train_audit', {})
                for tgt, wm in mets.items():
                    st.markdown(f"**{tgt}**")
                    html = '<div class="kpi-row">'
                    for w, m in wm.items():
                        r2 = m['R2']
                        cls = r2c(r2)
                        # Display name
                        dname = w.split('::')[-1] if '::' in w else w
                        sname = w.split('::')[0] if '::' in w else ''
                        r2s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                        html += (
                            f'<div class="kpi">'
                            f'<div class="kl">{sname}/{dname}</div>'
                            f'<div class="kv {cls}">{r2s}</div>'
                            f'<div class="ks">RMSE={m["RMSE"]:.4f} · '
                            f'MAE={m["MAE"]:.4f} · N={m["N"]:,}</div></div>')
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
                    if tgt in audit:
                        st.caption(
                            f"Train: {audit[tgt]['n_train_rows']:,} rows · "
                            f"OOF: {audit[tgt]['n_oof_rows']:,} · "
                            f"Wells: {audit[tgt]['n_train_wells']}")
            else:
                st.markdown(
                    '<div class="ibox">Label aktual tidak tersedia — '
                    'metrics tidak dihitung.</div>',
                    unsafe_allow_html=True)

            st.markdown("### Feature Importance")
            fig_fi = plot_fi(ms_res['feat_imp'])
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)

            # ── Metrics per Zone × Structure (grouped per structure) ──
            st.markdown("### Performance per Zone × Structure")
            if len(df_te) > 0 and 'ZONE' in df_te.columns:
                # Extract structure from ZONE (format: ZONE_STRUCTURE)
                # or from WELL_NAME (format: STRUCTURE::WELL)
                _te_structs = []
                if 'ZONE_ORIGINAL' in df_te.columns:
                    # Use WELL_NAME to get structure
                    _te_structs = sorted(set(
                        w.split('::')[0] for w in df_te['WELL_NAME'].unique()
                        if '::' in w))

                if _te_structs:
                    # Filter by structure
                    _sel_struct = st.selectbox(
                        "Filter Struktur",
                        ["Semua"] + _te_structs,
                        key='ms_zone_struct_filter')

                    for s_name in (_te_structs if _sel_struct == "Semua"
                                   else [_sel_struct]):
                        st.markdown(f"#### Struktur: {s_name}")
                        # Filter df_te for this structure
                        s_mask = df_te['WELL_NAME'].str.startswith(
                            f"{s_name}::")
                        df_s = df_te[s_mask]
                        if len(df_s) == 0:
                            st.caption(
                                "Tidak ada data test untuk struktur ini.")
                            continue

                        zm = compute_zone_metrics(
                            df_s, ms_cfg['targets'], zone_col='ZONE')
                        if zm:
                            for tgt, zdf in zm.items():
                                st.markdown(f"**{tgt}**")
                                html = '<div class="kpi-row">'
                                for _, row in zdf.iterrows():
                                    r2 = row['R2']
                                    cls = r2c(r2)
                                    r2s = (f"{r2:.4f}"
                                           if not np.isnan(r2) else "N/A")
                                    # Display zone original (strip structure suffix)
                                    zone_disp = str(row['ZONE'])
                                    if zone_disp.endswith(f"_{s_name}"):
                                        zone_disp = zone_disp[
                                            :-(len(s_name) + 1)]
                                    html += (
                                        f'<div class="kpi">'
                                        f'<div class="kl">'
                                        f'{zone_disp} — {s_name}</div>'
                                        f'<div class="kv {cls}">{r2s}</div>'
                                        f'<div class="ks">'
                                        f'RMSE={row["RMSE"]:.4f} · '
                                        f'MAE={row["MAE"]:.4f} · '
                                        f'N={int(row["N"]):,}</div></div>')
                                html += '</div>'
                                st.markdown(html, unsafe_allow_html=True)
                        else:
                            st.caption("Tidak cukup data per zona.")

                    with st.expander("📋 Tabel Detail per Zone (semua)"):
                        zm_all = compute_zone_metrics(
                            df_te, ms_cfg['targets'], zone_col='ZONE')
                        if zm_all:
                            for tgt, zdf in zm_all.items():
                                st.markdown(f"**{tgt}**")
                                st.dataframe(
                                    zdf.style.format(
                                        {'R2': '{:.4f}', 'RMSE': '{:.5f}',
                                         'MAE': '{:.5f}', 'N': '{:,.0f}'},
                                        na_rep='—'),
                                    use_container_width=True, hide_index=True)
                else:
                    # Fallback: no structure info, show flat zone metrics
                    zm = compute_zone_metrics(
                        df_te, ms_cfg['targets'], zone_col='ZONE')
                    if zm:
                        for tgt, zdf in zm.items():
                            st.markdown(f"**{tgt}**")
                            html = '<div class="kpi-row">'
                            for _, row in zdf.iterrows():
                                r2 = row['R2']
                                cls = r2c(r2)
                                r2s = (f"{r2:.4f}"
                                       if not np.isnan(r2) else "N/A")
                                html += (
                                    f'<div class="kpi">'
                                    f'<div class="kl">{row["ZONE"]}</div>'
                                    f'<div class="kv {cls}">{r2s}</div>'
                                    f'<div class="ks">'
                                    f'RMSE={row["RMSE"]:.4f} · '
                                    f'MAE={row["MAE"]:.4f} · '
                                    f'N={int(row["N"]):,}</div></div>')
                            html += '</div>'
                            st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div class="ibox">Tidak cukup data per zona.</div>',
                            unsafe_allow_html=True)

        # ── Log Plot ──
        with mt2:
            st.markdown("### Log Plot — Aktual vs Prediksi")
            plot_wells = (sorted(df_te['WELL_NAME'].unique())
                          if len(df_te) > 0
                          else sorted(df_tr['WELL_NAME'].unique()))
            sel_w = st.selectbox(
                "Pilih Sumur", plot_wells,
                format_func=lambda x: (
                    f"{x.split('::')[0]} / {x.split('::')[-1]}"
                    if '::' in x else x),
                key='ms_pw')
            df_src = (df_te if (len(df_te) > 0
                                and sel_w in df_te['WELL_NAME'].values)
                      else df_tr)
            df_w = df_src[df_src['WELL_NAME'] == sel_w].copy()
            if len(df_w) > 0:
                dmin = float(df_w['DEPTH'].min())
                dmax = float(df_w['DEPTH'].max())
                c1, c2 = st.columns(2)
                with c1:
                    d0 = st.number_input("Dari (m)", dmin, dmax, dmin,
                                         key='ms_d0')
                with c2:
                    d1 = st.number_input("Sampai (m)", dmin, dmax, dmax,
                                         key='ms_d1')
                df_w = df_w[(df_w['DEPTH'] >= d0) & (df_w['DEPTH'] <= d1)]
                display_name = (f"{sel_w.split('::')[0]} / "
                                f"{sel_w.split('::')[-1]}"
                                if '::' in sel_w else sel_w)

                # Zone filter
                zf_ms = None
                if 'ZONE' in df_w.columns:
                    _zs = sorted([z for z in df_w['ZONE'].dropna().unique()
                                  if str(z).upper()
                                  not in ('UNKNOWN', 'NAN', '', 'NONE')])
                    if _zs:
                        sel_zs = st.multiselect(
                            "Filter Zone (kosong = semua)",
                            _zs, default=[], key='ms_log_zone_filter')
                        if sel_zs:
                            zf_ms = sel_zs
                st.plotly_chart(
                    plot_log(df_w, ms_cfg['targets'], display_name,
                             zone_filter=zf_ms),
                    use_container_width=True)

        # ── Scatter ──
        with mt3:
            st.markdown("### Scatter — Aktual vs Prediksi")
            sc_wells = (sorted(df_te['WELL_NAME'].unique())
                        if len(df_te) > 0 else [])
            if not sc_wells:
                st.markdown(
                    '<div class="ibox wbox">Tidak ada data test.</div>',
                    unsafe_allow_html=True)
            else:
                sel_sw = st.multiselect(
                    "Filter Sumur", sc_wells, default=sc_wells,
                    format_func=lambda x: (
                        f"{x.split('::')[0]}/{x.split('::')[-1]}"
                        if '::' in x else x),
                    key='ms_sc_wells')
                df_sc = df_te[df_te['WELL_NAME'].isin(sel_sw)]

                # Zone filter
                zf_ms_sc = None
                if 'ZONE' in df_sc.columns:
                    _zs_sc = sorted([z for z in df_sc['ZONE'].dropna().unique()
                                     if str(z).upper()
                                     not in ('UNKNOWN', 'NAN', '', 'NONE')])
                    if _zs_sc:
                        sel_zs_sc = st.multiselect(
                            "Filter Zone (kosong = semua)",
                            _zs_sc, default=[], key='ms_sc_zone_filter')
                        if sel_zs_sc:
                            zf_ms_sc = sel_zs_sc
                fig_sc = plot_scatter(df_sc, ms_cfg['targets'],
                                      zone_filter=zf_ms_sc)
                if fig_sc:
                    st.plotly_chart(fig_sc, use_container_width=True)

        # ── Export CSV ──
        with mt4:
            st.markdown("### Download Hasil")
            choice = st.radio("Ekspor dari",
                              ["Test", "Training", "Semua"],
                              horizontal=True, key='ms_exp_choice')
            df_exp = (df_te if choice == "Test" else
                      df_tr if choice == "Training" else
                      pd.concat([df_tr, df_te], ignore_index=True)).copy()
            base_c = ['WELL_NAME', 'DEPTH', 'ZONE', 'ZONE_ORIGINAL',
                      'MARKER', 'GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT']
            if 'STRUCTURE' in df_exp.columns:
                base_c = ['STRUCTURE'] + base_c
            act_c = [t for t in ms_cfg['targets'] if t in df_exp.columns]
            pred_c = [f'{t}_PRED' for t in ms_cfg['targets']
                      if f'{t}_PRED' in df_exp.columns]
            drv_c = [c for c in ['LOG_RT', 'DN_SEP', 'NPHI_RHOB_CROSS',
                                 'CROSS_POS', 'VSH_LINEAR']
                     if c in df_exp.columns]
            all_c = [c for c in base_c + act_c + pred_c + drv_c
                     if c in df_exp.columns]
            df_out = df_exp[all_c].reset_index(drop=True)
            st.dataframe(df_out.head(100), use_container_width=True,
                         height=300)
            st.caption(f"Preview 100 / {len(df_out):,} baris")
            st.download_button(
                "⬇  Download CSV",
                data=df_out.to_csv(index=False).encode('utf-8'),
                file_name="petro_ml_multi_predictions.csv",
                mime="text/csv", use_container_width=True,
                key='ms_dl_csv')

        # ── Download Model ──
        with mt5:
            st.markdown("### Download Model Package (Multi-Structure)")
            if ms_res.get('models'):
                for tgt, info in ms_res['models'].items():
                    mtype = type(info['model']).__name__
                    nf = len(info['ft_tr'])
                    st.markdown(
                        f'<div class="kpi" style="display:inline-block;'
                        f'margin:4px;">'
                        f'<div class="kl">{tgt}</div>'
                        f'<div class="kv ok" style="font-size:1rem;">'
                        f'{mtype}</div>'
                        f'<div class="ks">{nf} features</div></div>',
                        unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    pkg = export_model_package(
                        ms_res, ms_cfg,
                        SS.get('ms_gr_norm_params', {}))
                    st.download_button(
                        "⬇  Download Model (.pkl)",
                        data=pkg, file_name="petro_ml_multi_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True, key='ms_dl_pkl')
                with c2:
                    doc = generate_model_doc(ms_cfg, ms_res)
                    st.download_button(
                        "⬇  Download Docs (.md)",
                        data=doc.encode('utf-8'),
                        file_name="petro_ml_multi_README.md",
                        mime="text/markdown",
                        use_container_width=True, key='ms_dl_doc')

                # ── Per-target individual download (Multi) ──
                st.divider()
                st.markdown("### Download Model per Target")
                st.markdown(
                    '<div class="ibox">Download model terpisah per target (VSH, PHIE, SW). '
                    'Setiap file berisi 1 model + dokumentasi input log + cascading dependency. '
                    'Cocok untuk migrasi ke sistem lain.</div>',
                    unsafe_allow_html=True)

                ms_gr_p = SS.get('ms_gr_norm_params', {})
                ms_tgt_list = list(ms_res['models'].keys())
                ms_tgt_cols = st.columns(len(ms_tgt_list))
                for i_t, t_name in enumerate(ms_tgt_list):
                    with ms_tgt_cols[i_t]:
                        t_info = ms_res['models'][t_name]
                        st.markdown(f"**{t_name}**")
                        st.caption(f"Model: {type(t_info['model']).__name__}")
                        st.caption(f"Features: {len(t_info['ft_tr'])}")

                        ms_pkg_s = export_single_target_package(
                            t_name, ms_res, ms_cfg, ms_gr_p)
                        st.download_button(
                            f"⬇ Model {t_name} (.pkl)",
                            data=ms_pkg_s,
                            file_name=f"petro_ml_multi_{t_name.lower()}_model.pkl",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key=f'ms_dl_single_{t_name}')

                        ms_doc_s = generate_single_target_doc(
                            t_name, ms_res, ms_cfg)
                        st.download_button(
                            f"⬇ Docs {t_name} (.md)",
                            data=ms_doc_s.encode('utf-8'),
                            file_name=f"petro_ml_multi_{t_name.lower()}_README.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key=f'ms_dl_single_doc_{t_name}')

                for t_name in ms_tgt_list:
                    with st.expander(f"📖 Preview Docs — {t_name}", expanded=False):
                        st.markdown(generate_single_target_doc(
                            t_name, ms_res, ms_cfg))

            else:
                st.markdown(
                    '<div class="ibox wbox">Belum ada model.</div>',
                    unsafe_allow_html=True)

        # ── Model Testing (Multi-Structure) ──
        with mt6:
            st.markdown(
                "### Model Testing — Prediksi Well Baru (Multi-Structure)")
            st.markdown(
                '<div class="ibox">Upload file <b>.las</b> per struktur. '
                'Setiap grup LAS diberi nama struktur agar ZONE dikombinasi '
                'dengan benar (<code>ZONE_STRUCTURE</code>).</div>',
                unsafe_allow_html=True)

            # Model source
            ms_model_src = st.radio(
                "Sumber Model",
                ["Gunakan model aktif (baru di-train)", "Upload file .pkl"],
                horizontal=True, key='ms_model_src')

            ms_test_pkg = None
            if ms_model_src == "Upload file .pkl":
                ms_pkl = st.file_uploader("Upload Model (.pkl)", type=['pkl'],
                                          key='ms_pkl_upload')
                if ms_pkl is not None:
                    try:
                        ms_test_pkg = pickle.load(ms_pkl)
                        st.success(
                            f"✅ Model loaded — targets: "
                            f"{', '.join(ms_test_pkg.get('models', {}).keys())}")
                    except Exception as e:
                        st.error(f"❌ Gagal load: {e}")
            else:
                if ms_res.get('models'):
                    ms_test_pkg = {
                        'models': {t: info['model'] for t, info in ms_res['models'].items()},
                        'feature_cols': {t: info['ft_tr'] for t, info in ms_res['models'].items()},
                        'feature_chosen': {t: info['ft_chosen'] for t, info in ms_res['models'].items()},
                        'config': {
                            'targets': ms_cfg.get('targets', []),
                            'model_name': ms_cfg.get('model_name', 'LightGBM'),
                            'training_mode': ms_cfg.get('training_mode', 'Standard ML'),
                            'opts': ms_cfg.get('opts', {}),
                        },
                        'gr_norm_params': SS.get('ms_gr_norm_params', {}),
                        'le_zone': ms_res.get('le_zone'),
                        'target_bounds': TARGET_BOUNDS,
                        'mnemonic_map': MNEMONIC_MAP,
                    }
                    st.success(
                        f"✅ Model aktif — targets: {', '.join(ms_res['models'].keys())}")

            # ── Upload per struktur ──
            st.markdown("---")
            st.markdown("#### Input Well Test per Struktur")
            n_struct_input = st.number_input(
                "Jumlah struktur test", min_value=1, max_value=10,
                value=1, step=1, key='ms_n_test_struct')

            ms_test_inputs = []
            for si in range(int(n_struct_input)):
                with st.expander(f"Struktur Test #{si+1}", expanded=(si == 0)):
                    sn = st.text_input(
                        "Nama Struktur", value="", key=f'mst_name_{si}',
                        help="Harus sesuai nama saat training (misal BN, MJ)")
                    las = st.file_uploader(
                        "Upload LAS", type=['las'],
                        accept_multiple_files=True,
                        key=f'mst_las_{si}')
                    zcsv = st.file_uploader(
                        "Zone CSV (opsional)", type=['csv'],
                        key=f'mst_zone_{si}')
                    ms_test_inputs.append(
                        {'name': sn, 'las': las, 'zone': zcsv})

            # Validate
            has_any = any(inp['name'].strip() and inp['las']
                          for inp in ms_test_inputs)
            ms_run_test = st.button(
                "▶  Jalankan Prediksi (Multi)",
                disabled=(ms_test_pkg is None or not has_any),
                use_container_width=True, key='ms_run_test_btn')

            if ms_run_test and ms_test_pkg and has_any:
                all_test_results = {}
                all_test_metrics = {}
                targets = list(ms_test_pkg['models'].keys())
                progress = st.progress(0, text="Memproses...")

                total_las = sum(len(inp['las'] or []) for inp in ms_test_inputs
                                if inp['name'].strip())
                done = 0

                for inp in ms_test_inputs:
                    sn = inp['name'].strip().upper()
                    if not sn or not inp['las']:
                        continue

                    # Parse zone CSV for this structure
                    inp_zone_df = None
                    if inp['zone'] is not None:
                        try:
                            inp_zone_df = read_zone_csv(inp['zone'])
                        except Exception:
                            pass

                    for las_file in inp['las']:
                        wname = las_file.name.rsplit('.', 1)[0].upper()
                        wname = _normalize_well_name(wname)
                        done += 1
                        progress.progress(
                            done / max(total_las, 1),
                            text=f"Memproses {sn} / {wname}...")

                        content = las_file.read()
                        df_w = read_las_bytes(content, wname)
                        if df_w is None:
                            st.warning(f"⚠ Skip {sn}/{wname}")
                            continue

                        df_w['STRUCTURE'] = sn
                        df_pred = predict_with_package(
                            ms_test_pkg, df_w, zone_df=inp_zone_df)

                        display_key = f"{sn}::{wname}"
                        all_test_results[display_key] = df_pred

                        wm = {}
                        for tgt in targets:
                            if (tgt in df_pred.columns
                                    and f'{tgt}_PRED' in df_pred.columns):
                                m = safe_m(df_pred[tgt],
                                           df_pred[f'{tgt}_PRED'])
                                if m['N'] >= 5:
                                    wm[tgt] = m
                        if wm:
                            all_test_metrics[display_key] = wm

                progress.empty()

                if all_test_results:
                    SS['ms_test_results'] = all_test_results
                    SS['ms_test_metrics'] = all_test_metrics
                    st.success(
                        f"✅ Prediksi selesai — {len(all_test_results)} well")
                else:
                    st.error("❌ Tidak ada well berhasil diproses.")

            # ── Display results ──
            _mtr = SS.get('ms_test_results')
            _mtm = SS.get('ms_test_metrics')
            if _mtr and ms_test_pkg:
                targets = list(ms_test_pkg['models'].keys())
                well_keys = sorted(_mtr.keys())

                # Metrics per well
                if _mtm:
                    st.markdown("#### Metrics per Well")
                    for wk in well_keys:
                        if wk not in _mtm:
                            continue
                        wm = _mtm[wk]
                        parts = wk.split('::', 1)
                        label = f"{parts[0]} / {parts[1]}" if len(
                            parts) == 2 else wk
                        html = (f'<div style="margin-bottom:4px;">'
                                f'<b>{label}</b></div>'
                                f'<div class="kpi-row">')
                        for tgt, m in wm.items():
                            r2 = m['R2']
                            cls = r2c(r2)
                            r2s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                            html += (
                                f'<div class="kpi">'
                                f'<div class="kl">{tgt}</div>'
                                f'<div class="kv {cls}">{r2s}</div>'
                                f'<div class="ks">RMSE={m["RMSE"]:.4f} · '
                                f'MAE={m["MAE"]:.4f} · N={m["N"]:,}</div>'
                                f'</div>')
                        html += '</div>'
                        st.markdown(html, unsafe_allow_html=True)

                # Metrics per Zone×Structure
                if _mtm:
                    st.markdown("#### Metrics per Zone × Structure")
                    df_all_mtr = pd.concat(_mtr.values(), ignore_index=True)
                    if 'ZONE' in df_all_mtr.columns:
                        zm = compute_zone_metrics(
                            df_all_mtr, targets, zone_col='ZONE')
                        if zm:
                            for tgt, zdf in zm.items():
                                st.markdown(f"**{tgt}**")
                                html = '<div class="kpi-row">'
                                for _, row in zdf.iterrows():
                                    r2 = row['R2']
                                    cls = r2c(r2)
                                    r2s = (f"{r2:.4f}" if not np.isnan(r2)
                                           else "N/A")
                                    html += (
                                        f'<div class="kpi">'
                                        f'<div class="kl">{row["ZONE"]}</div>'
                                        f'<div class="kv {cls}">{r2s}</div>'
                                        f'<div class="ks">'
                                        f'RMSE={row["RMSE"]:.4f} · '
                                        f'MAE={row["MAE"]:.4f} · '
                                        f'N={int(row["N"]):,}</div></div>')
                                html += '</div>'
                                st.markdown(html, unsafe_allow_html=True)

                if not _mtm:
                    st.markdown(
                        '<div class="ibox">Tidak ada label aktual — '
                        'metrics tidak dihitung (prediksi tetap tersedia).'
                        '</div>', unsafe_allow_html=True)

                # Log plot
                st.markdown("#### Log Plot")
                sel_tw = st.selectbox(
                    "Pilih Well", well_keys,
                    format_func=lambda x: (
                        f"{x.split('::')[0]} / {x.split('::')[-1]}"
                        if '::' in x else x),
                    key='mst_plot_w')
                if sel_tw and sel_tw in _mtr:
                    df_tw = _mtr[sel_tw]
                    if len(df_tw) > 0:
                        dmin = float(df_tw['DEPTH'].min())
                        dmax = float(df_tw['DEPTH'].max())
                        c1, c2 = st.columns(2)
                        with c1:
                            d0 = st.number_input(
                                "Dari (m)", dmin, dmax, dmin, key='mst_d0')
                        with c2:
                            d1 = st.number_input(
                                "Sampai (m)", dmin, dmax, dmax, key='mst_d1')
                        df_filt = df_tw[
                            (df_tw['DEPTH'] >= d0) & (df_tw['DEPTH'] <= d1)]
                        parts = sel_tw.split('::', 1)
                        disp = (f"{parts[0]} / {parts[1]}"
                                if len(parts) == 2 else sel_tw)

                        # Zone filter
                        zf_mtst = None
                        if 'ZONE' in df_filt.columns:
                            _zst = sorted([
                                z for z in df_filt['ZONE'].dropna().unique()
                                if str(z).upper()
                                not in ('UNKNOWN', 'NAN', '', 'NONE')])
                            if _zst:
                                sel_zst = st.multiselect(
                                    "Filter Zone (kosong = semua)",
                                    _zst, default=[],
                                    key='mst_log_zone_filter')
                                if sel_zst:
                                    zf_mtst = sel_zst
                        st.plotly_chart(
                            plot_log(df_filt, targets, disp,
                                     zone_filter=zf_mtst),
                            use_container_width=True)

                # Scatter
                if _mtm:
                    st.markdown("#### Scatter Plot")
                    sel_sc = st.multiselect(
                        "Filter Wells", well_keys, default=well_keys,
                        format_func=lambda x: (
                            f"{x.split('::')[0]}/{x.split('::')[-1]}"
                            if '::' in x else x),
                        key='mst_sc_wells')
                    if sel_sc:
                        df_sc = pd.concat(
                            [_mtr[w] for w in sel_sc if w in _mtr],
                            ignore_index=True)

                        # Zone filter
                        zf_mtsc = None
                        if 'ZONE' in df_sc.columns:
                            _zsc = sorted([
                                z for z in df_sc['ZONE'].dropna().unique()
                                if str(z).upper()
                                not in ('UNKNOWN', 'NAN', '', 'NONE')])
                            if _zsc:
                                sel_zsc = st.multiselect(
                                    "Filter Zone (kosong = semua)",
                                    _zsc, default=[],
                                    key='mst_sc_zone_filter')
                                if sel_zsc:
                                    zf_mtsc = sel_zsc
                        fig_sc = plot_scatter(df_sc, targets,
                                              zone_filter=zf_mtsc)
                        if fig_sc:
                            st.plotly_chart(fig_sc, use_container_width=True)

                # Download
                st.markdown("#### Download Hasil")
                df_all_exp = pd.concat(
                    [_mtr[w] for w in well_keys], ignore_index=True)
                base_c = ['STRUCTURE', 'WELL_NAME', 'DEPTH', 'ZONE',
                          'ZONE_ORIGINAL', 'MARKER',
                          'GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT']
                tgt_c = []
                for t in targets:
                    if t in df_all_exp.columns:
                        tgt_c.append(t)
                    if f'{t}_PRED' in df_all_exp.columns:
                        tgt_c.append(f'{t}_PRED')
                exp_c = [c for c in base_c + tgt_c
                         if c in df_all_exp.columns]
                df_exp_out = df_all_exp[exp_c].reset_index(drop=True)
                st.dataframe(df_exp_out.head(50),
                             use_container_width=True, height=250)
                st.caption(f"Preview 50 / {len(df_exp_out):,} baris")
                st.download_button(
                    "⬇  Download CSV",
                    data=df_exp_out.to_csv(index=False).encode('utf-8'),
                    file_name="petro_ml_multi_test_predictions.csv",
                    mime="text/csv", use_container_width=True,
                    key='mst_dl_csv')

        # ── ML Workflow ──
        with mt7:
            render_ml_workflow(cfg=ms_cfg, res=ms_res, mode='multi')

    elif not structs:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
          justify-content:center;height:46vh;gap:16px;opacity:0.5;">
          <div style="font-size:3.2rem;">🏗</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;
            color:#7d8590;text-align:center;line-height:1.9;">
            Multi-Structure Training<br>
            <span style="font-size:.74rem;color:#484f58;">
              Tambahkan struktur di sidebar (nama + ZIP + Zone CSV)
              lalu jalankan training
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
          justify-content:center;height:46vh;gap:16px;opacity:0.5;">
          <div style="font-size:3.2rem;">🪨</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;
            color:#7d8590;text-align:center;line-height:1.9;">
            Jalankan QC → GR Norm → Pilih Test → Training<br>
            <span style="font-size:.74rem;color:#484f58;">
              klik ▶ Jalankan Training (Multi) di sidebar
            </span>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — PAGE SELECTOR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo">🪨 Petro·ML</div>'
                '<div class="logo-sub">petrophysics machine learning</div>',
                unsafe_allow_html=True)
    app_page = st.radio(
        "Mode",
        ["Single Structure", "Multi-Structure Training"],
        horizontal=True, key='_app_page_radio',
        help="Single: 1 struktur · Multi: gabungkan beberapa struktur untuk training")
    st.session_state['app_page'] = app_page

# ══════════════════════════════════════════════════════════════════
# PAGE: MULTI-STRUCTURE TRAINING  (runs & st.stop() → skips single)
# ══════════════════════════════════════════════════════════════════
if app_page == "Multi-Structure Training":
    _run_multi_structure_page()   # defined below session state, before sidebar
    st.stop()

# ══════════════════════════════════════════════════════════════════
# PAGE: SINGLE STRUCTURE (existing code, untouched)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # 01 ZIP
    st.markdown('<div class="sec">01 · Upload ZIP</div>',
                unsafe_allow_html=True)
    zip_up = st.file_uploader("File ZIP berisi LAS", type=['zip'],
                              help="Semua file .las dalam satu ZIP")
    if zip_up is not None:
        # Ambil bytes dari uploader — ini tidak berubah selama file sama
        file_bytes = zip_up.read()

        # Cek apakah file ini BERBEDA dari yang sebelumnya di-load
        # (bandingkan hash bytes, bukan objek file)
        import hashlib
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.get('zip_hash') != file_hash:
            # File baru → load ulang
            with st.spinner("Membaca LAS dari ZIP... (sekali saja)"):
                wells = load_zip_cached(file_bytes)
            if wells:
                st.session_state['all_wells'] = wells
                st.session_state['zip_hash'] = file_hash
                st.session_state['normalized'] = False
                st.session_state['qc_log'] = None
                st.session_state['combined_df'] = None
                st.success(
                    f"✅ {len(wells)} sumur: {', '.join(sorted(wells.keys()))}")
            else:
                st.error("Tidak ada LAS valid di ZIP")
        else:
            # File sama → pakai session state yang sudah ada, tidak re-load
            if st.session_state['all_wells']:
                st.success(
                    f"✅ {len(st.session_state['all_wells'])} sumur (cached)")

    # 02 Zone & Marker
    st.markdown('<div class="sec">02 · Zone & Marker CSV (opsional)</div>',
                unsafe_allow_html=True)
    import hashlib

    zone_up = st.file_uploader("Zone CSV", type=['csv'],
                               help="Kolom: WELL_NAME/Well identifier, MD, Zone/Surface")
    if zone_up is not None:
        zb = zone_up.read()
        zh = hashlib.md5(zb).hexdigest()
        if st.session_state.get('zone_hash') != zh:
            zdf = read_zone_csv(io.BytesIO(zb))
            st.session_state['zone_df'] = zdf
            st.session_state['zone_hash'] = zh
            st.session_state['combined_df'] = None   # reset combined
            st.success(f"✅ Zone: {len(zdf):,} baris · "
                       f"{zdf['WELL_NAME'].nunique()} sumur")
        else:
            if st.session_state['zone_df'] is not None:
                st.success(
                    f"✅ Zone: {len(st.session_state['zone_df']):,} baris (cached)")

    marker_up = st.file_uploader("Marker CSV", type=['csv'],
                                 help="Kolom: WELL_NAME, DEPTH/MD, MARKER (opsional)")
    if marker_up is not None:
        mb = marker_up.read()
        mh = hashlib.md5(mb).hexdigest()
        if st.session_state.get('marker_hash') != mh:
            marker_up.seek(0)
            mdf = pd.read_csv(io.BytesIO(mb))
            st.session_state['marker_df'] = mdf
            st.session_state['marker_hash'] = mh
            st.session_state['combined_df'] = None
            st.success(f"✅ Marker: {len(mdf):,} baris")
        else:
            if st.session_state['marker_df'] is not None:
                st.success(
                    f"✅ Marker: {len(st.session_state['marker_df']):,} baris (cached)")
    # Build combined (raw — sebelum QC & normalisasi)
    all_wells = st.session_state['all_wells']
    if all_wells:
        combined_raw = build_combined(all_wells,
                                      st.session_state['zone_df'],
                                      st.session_state['marker_df'])
    else:
        combined_raw = None

    # 03 · QC Pipeline
    st.markdown('<div class="sec">03 · QC Data</div>', unsafe_allow_html=True)

    can_qc = combined_raw is not None and len(combined_raw) > 0
    use_zscore_qc = st.checkbox('Aktifkan Z-Score Outlier Filter', value=False,
                                help='Deteksi outlier depth-wise pada GR/NPHI/RHOB/RT sebelum QC utama.')
    zscore_thr = st.select_slider(
        'Z-Score threshold', options=[2.5, 3.0, 3.5], value=3.0)
    qc_btn = st.button("🧹  Jalankan QC Pipeline",
                       disabled=not can_qc, use_container_width=True)

    if qc_btn and can_qc:
        df_qc, qc_log = run_qc_pipeline(
            combined_raw.copy(), use_zscore=use_zscore_qc, zscore_threshold=zscore_thr)
        st.session_state['combined_df'] = df_qc
        st.session_state['qc_log'] = qc_log
        st.session_state['normalized'] = False   # reset normalisasi
        st.session_state['gr_norm_params'] = {}
        st.success(f"✅ QC selesai — {len(df_qc):,} baris tersisa")

    if st.session_state.get('qc_log'):
        ql = st.session_state['qc_log']
        st.markdown(
            f'<div class="ibox">'
            f'Drop all-NaN logs : <b>{ql.get("drop_all_nan_logs",0):,}</b><br>'
            f'Z-Score rows drop : <b>{ql.get("zscore_rows_dropped",0):,}</b><br>'
            f'RT invalid → NaN  : <b>{ql.get("rt_invalid_to_nan",0):,}</b><br>'
            f'Drop VSH out-of-range : <b>{ql.get("drop_vsh_out_of_range",0):,}</b><br>'
            f'Drop PHIE out-of-range: <b>{ql.get("drop_phie_out_of_range",0):,}</b><br>'
            f'Drop SW out-of-range  : <b>{ql.get("drop_sw_out_of_range",0):,}</b><br>'
            f'Drop label kosong     : <b>{ql.get("drop_empty_labels",0):,}</b><br>'
            f'<b>Total drop: {ql.get("total_dropped",0):,} · Sisa: {ql.get("remaining",0):,}</b><br>'
            f'<span style="color:var(--warn)">SW = 1 dipertahankan · jumlah: {ql.get("sw_eq_1_kept",0):,}</span>'
            f'</div>',
            unsafe_allow_html=True)

    # Pakai df setelah QC jika ada, fallback ke raw
    _cd = st.session_state.get('combined_df')
    combined = _cd if (_cd is not None and not _cd.empty) else combined_raw

    # 04 · GR Normalisasi
    st.markdown('<div class="sec">04 · Normalisasi GR</div>',
                unsafe_allow_html=True)

    has_zone = (combined is not None and 'ZONE' in combined.columns
                and combined['ZONE'].nunique() > 1)
    has_gr = (combined is not None and 'GR' in combined.columns
              and combined['GR'].notna().any())

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        pct_low_old = st.number_input(
            "PCT_LOW_OLD", 1.0, 40.0, 3.0, key='pct_low_old')
    with c_p2:
        pct_high_old = st.number_input(
            "PCT_HIGH_OLD", 60.0, 99.0, 97.0, key='pct_high_old')

    c_p3, c_p4 = st.columns(2)
    with c_p3:
        pct_low_new = st.number_input(
            "PCT_LOW_NEW", 1.0, 40.0, 3.0, key='pct_low_new')
    with c_p4:
        pct_high_new = st.number_input(
            "PCT_HIGH_NEW", 60.0, 99.0, 97.0, key='pct_high_new')

    if not has_zone:
        st.markdown('<div class="ibox wbox">Upload Zone CSV dulu agar '
                    'normalisasi per zona bisa dilakukan.</div>',
                    unsafe_allow_html=True)

    # Cek apakah GR_NORM sudah ada di data (dari LAS langsung)
    has_gr_norm_direct = (
        combined is not None and 'GR_NORM' in combined.columns
        and combined['GR_NORM'].notna().any())

    if has_gr_norm_direct and not st.session_state.get('normalized'):
        _n_valid = combined['GR_NORM'].notna().sum()
        st.info(
            f"ℹ GR_NORM sudah tersedia di data ({_n_valid:,} nilai valid). "
            f"Normalisasi di-skip. Jika ingin menghitung ulang, "
            f"hapus kolom GR_NORM dari LAS atau tetap klik tombol di bawah.")

    can_norm = has_gr and combined is not None
    norm_btn = st.button("📐  Hitung GR_NORM",
                         disabled=not can_norm, use_container_width=True)

    if norm_btn and can_norm:
        params = compute_gr_norm_params(
            combined,
            pct_low_old=pct_low_old,
            pct_high_old=pct_high_old,
            pct_low_new=pct_low_new,
            pct_high_new=pct_high_new
        )
        combined = apply_gr_norm(combined, params)
        st.session_state['combined_df'] = combined
        st.session_state['gr_norm_params'] = params
        st.session_state['normalized'] = True
        st.success(f"✅ GR_NORM selesai — "
                   f"{combined['GR_NORM'].notna().sum():,} nilai valid")
    if st.session_state['normalized'] or has_gr_norm_direct:
        params = st.session_state.get('gr_norm_params', {})
        if params:
            n_zones = len(params)
            st.markdown(
                f'<div class="ibox">'
                f'GR_NORM aktif · struktur = semua well dalam ZIP<br>'
                f'OLD: P{pct_low_old}/P{pct_high_old} per well<br>'
                f'NEW: P{pct_low_new}/P{pct_high_new} gabungan semua well'
                f'</div>',
                unsafe_allow_html=True
            )

            with st.expander("📊 Parameter per Zona", expanded=False):
                rows = []

                global_new = params.get('global_new', {})
                rows.append({
                    'Type': 'GLOBAL_NEW',
                    'Well': 'ALL_WELLS_IN_ZIP',
                    'N': global_new.get('N', 0),
                    'P_LOW_NEW': round(global_new.get('p_low', np.nan), 3) if global_new else np.nan,
                    'P_HIGH_NEW': round(global_new.get('p_high', np.nan), 3) if global_new else np.nan,
                    'Source': global_new.get('source', '-')
                })

                for w, p in params.get('well_old', {}).items():
                    rows.append({
                        'Type': 'WELL_OLD',
                        'Well': w,
                        'N': p.get('N', 0),
                        'P_LOW_OLD': round(p.get('p_low', np.nan), 3),
                        'P_HIGH_OLD': round(p.get('p_high', np.nan), 3),
                        'Source': p.get('source', '-')
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
        else:
            st.markdown('<div class="ibox">GR_NORM tersedia dari data input '
                        '(mis. log GRN), sehingga normalisasi bisa di-skip.</div>',
                        unsafe_allow_html=True)

    # 05 · Sumur Test
    st.markdown('<div class="sec">05 · Sumur Test</div>',
                unsafe_allow_html=True)
    well_list = sorted(all_wells.keys()) if all_wells else []
    test_wells = st.multiselect("Pilih sumur sebagai Test Set",
                                options=well_list,
                                help="Sisanya jadi Training otomatis")
    if test_wells:
        n_tr = len([w for w in well_list if w not in test_wells])
        st.markdown(f'<div class="ibox">🏋 Train: <b>{n_tr}</b> sumur &nbsp;·&nbsp; '
                    f'🔬 Test: <b>{len(test_wells)}</b> sumur</div>',
                    unsafe_allow_html=True)

    # 06 Target
    st.markdown('<div class="sec">06 · Target Prediksi</div>',
                unsafe_allow_html=True)
    avail_tgts = ([c for c in ALL_TARGETS
                   if combined is not None and c in combined.columns
                   and combined[c].notna().any()]
                  if combined is not None else ALL_TARGETS)
    sel_tgts = st.multiselect("Target Prediksi", options=avail_tgts,
                              default=avail_tgts)

    # 07 Feature per Target
    st.markdown('<div class="sec">07 · Feature per Target</div>',
                unsafe_allow_html=True)

    # Log dasar yang tersedia (termasuk GR_NORM jika normalisasi sudah dijalankan)
    avail_base = [c for c in ALL_LOGS if combined is not None
                  and len(combined) > 0 and c in combined.columns
                  and combined[c].notna().any()]
    if not avail_base:
        avail_base = [c for c in ['GR', 'NPHI', 'RHOB', 'RT']
                      if combined is not None and c in (combined.columns if combined is not None else [])]

    # Derived features
    has_nphi_rhob = ('NPHI' in avail_base and 'RHOB' in avail_base)
    derived_opts = []
    if has_nphi_rhob:
        derived_opts += ['DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS']
    derived_opts += ['ZONE_ENC']

    propagated = ['VSH_PRED', 'PHIE_PRED']

    # Default — gunakan GR_NORM jika tersedia, fallback ke GR
    gr_feat = 'GR_NORM' if 'GR_NORM' in avail_base else 'GR'
    _default_feats = {
        'VSH': [f for f in [gr_feat, 'VSH_LINEAR', 'RHOB', 'DN_SEP', 'NPHI_RHOB_CROSS', 'ZONE_ENC']
                if f in avail_base + derived_opts],
        'PHIE': [f for f in ['NPHI', 'RHOB', 'DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS',
                             'VSH_PRED', 'ZONE_ENC']
                 if f in avail_base + derived_opts + propagated],
        'SW': [f for f in ['RT', 'NPHI_RHOB_CROSS', 'CROSS_POS',
                           'VSH_PRED', 'PHIE_PRED', 'ZONE_ENC']
               if f in avail_base + derived_opts + propagated],
    }

    sel_logs = []   # tidak dipakai langsung lagi
    target_feats = {}   # {target: [feat, ...]}
    opts_per_target = {}

    for tgt in (sel_tgts or ALL_TARGETS):
        with st.expander(f"📐 {tgt} — pilih feature", expanded=True):
            all_opts = avail_base + derived_opts + propagated
            # hapus duplikat, jaga urutan
            all_opts = list(dict.fromkeys(all_opts))
            defaults = [f for f in _default_feats.get(tgt, avail_base)
                        if f in all_opts]

            chosen = st.multiselect(
                f"Feature untuk {tgt}",
                options=all_opts,
                default=defaults,
                key=f'feat_{tgt}',
                help=f"Pilih bebas feature untuk prediksi {tgt}"
            )
            target_feats[tgt] = chosen

            # Preview pill
            pills = ''.join(f'<span class="pill">{f}</span>' for f in chosen)
            st.markdown(f'<div class="ibox" style="margin-top:4px;">'
                        f'{pills if pills else "—"}</div>',
                        unsafe_allow_html=True)

    # opts global (untuk compute_features — perlu tahu apa yang dihitung)
    all_chosen_flat = [f for feats in target_feats.values() for f in feats]
    opts = {
        'use_dn_sep': 'DN_SEP' in all_chosen_flat,
        'use_crossover': ('NPHI_RHOB_CROSS' in all_chosen_flat
                          or 'CROSS_POS' in all_chosen_flat),
        'use_zone': 'ZONE_ENC' in all_chosen_flat,
    }

    # 08 Model Params
    st.markdown('<div class="sec">08 · Model Parameters</div>',
                unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Algoritma Model",
        options=['LightGBM', 'RandomForest', 'CatBoost', 'ANN',
                 'XGBoost', 'ExtraTrees', 'Stacking Ensemble'],
        index=0,
        help='Stacking Ensemble menggabungkan LightGBM+RF(+CatBoost) dengan Ridge meta-learner.'
    )
    if model_choice == 'CatBoost' and CatBoostRegressor is None:
        st.markdown('<div class="ibox wbox">CatBoost belum tersedia di environment ini. '
                    'Pilih LightGBM atau RandomForest, atau install catboost dulu.</div>',
                    unsafe_allow_html=True)
    if model_choice == 'XGBoost' and XGBRegressor is None:
        st.markdown('<div class="ibox wbox">XGBoost belum tersedia. '
                    'Jalankan: <code>pip install xgboost</code></div>',
                    unsafe_allow_html=True)

    training_mode = st.selectbox(
        'Mode Training',
        options=['Standard ML', 'Hybrid Residual VSH (Linear)'],
        index=0,
        help='Hybrid hanya diterapkan pada target VSH: model mempelajari residual terhadap VSH_LINEAR.'
    )

    if model_choice == 'LightGBM':
        with st.expander("⚙ LightGBM Settings"):
            ne = st.slider("n_estimators",    100, 2000, 800, 100)
            lr = st.select_slider("learning_rate",
                                  [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], value=0.02)
            nl = st.slider("num_leaves",       15, 255, 127, 16)
            mc = st.slider("min_child_samples", 5, 100,  10,  5)
            ss = st.slider("subsample",        0.4, 1.0, 0.7, 0.1)
            cb = st.slider("colsample_bytree", 0.4, 1.0, 0.7, 0.1)
            ra = st.number_input("reg_alpha",  0.0, 10.0, 0.05, 0.01)
            rl = st.number_input("reg_lambda", 0.0, 10.0, 0.50, 0.10)
        model_params = dict(n_estimators=ne, learning_rate=lr, num_leaves=nl,
                            min_child_samples=mc, subsample=ss, colsample_bytree=cb,
                            reg_alpha=ra, reg_lambda=rl,
                            random_state=42, n_jobs=-1, verbose=-1)
        model_name = 'lightgbm'
    elif model_choice == 'RandomForest':
        with st.expander("⚙ RandomForest Settings"):
            ne = st.slider("n_estimators", 100, 2000, 600, 100, key='rf_ne')
            md = st.slider("max_depth", 3, 40, 14, 1, key='rf_md')
            msl = st.slider("min_samples_leaf", 1, 20, 2, 1, key='rf_msl')
            ms = st.slider("min_samples_split", 2, 20, 4, 1, key='rf_mss')
            mf = st.selectbox(
                "max_features", ['sqrt', 'log2', None], index=0, key='rf_mf')
        model_params = dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl,
                            min_samples_split=ms, max_features=mf,
                            random_state=42, n_jobs=-1)
        model_name = 'randomforest'
    elif model_choice == 'ANN':
        with st.expander("⚙ ANN / MLP Settings"):
            hidden = st.selectbox('hidden_layer_sizes', options=[
                                  '64', '128', '64-32', '128-64', '128-64-32'], index=3)
            alpha = st.number_input(
                'alpha', 0.00001, 0.1, 0.0005, 0.0001, format='%.5f')
            lr_init = st.select_slider('learning_rate_init', [
                                       0.0005, 0.001, 0.003, 0.005, 0.01], value=0.001)
            act = st.selectbox('activation', options=['relu', 'tanh'], index=0)
            max_iter = st.slider('max_iter', 200, 3000, 1200, 100)
        hidden_map = {
            '64': (64,),
            '128': (128,),
            '64-32': (64, 32),
            '128-64': (128, 64),
            '128-64-32': (128, 64, 32),
        }
        model_params = dict(hidden_layer_sizes=hidden_map[hidden], activation=act, solver='adam',
                            alpha=float(alpha), learning_rate_init=float(lr_init),
                            max_iter=int(max_iter), early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=25, random_state=42)
        model_name = 'ann'
    elif model_choice == 'CatBoost':
        with st.expander("⚙ CatBoost Settings"):
            ne = st.slider("iterations", 100, 2000, 800, 100, key='cb_it')
            lr = st.select_slider("learning_rate",
                                  [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1], value=0.03, key='cb_lr')
            depth = st.slider("depth", 3, 10, 6, 1, key='cb_depth')
            l2 = st.number_input("l2_leaf_reg", 1.0, 20.0,
                                 3.0, 0.5, key='cb_l2')
        model_params = dict(iterations=ne, learning_rate=lr, depth=depth,
                            l2_leaf_reg=l2, loss_function='RMSE',
                            eval_metric='RMSE', random_seed=42,
                            verbose=0)
        model_name = 'catboost'
    elif model_choice == 'XGBoost':
        with st.expander("⚙ XGBoost Settings"):
            ne = st.slider("n_estimators", 100, 2000, 800, 100, key='xgb_ne')
            lr = st.select_slider("learning_rate",
                                  [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                                  value=0.03, key='xgb_lr')
            depth = st.slider("max_depth", 3, 10, 6, 1, key='xgb_depth')
            mcw = st.slider("min_child_weight", 1, 20, 5, 1, key='xgb_mcw')
            ss = st.slider("subsample", 0.4, 1.0, 0.8, 0.1, key='xgb_ss')
            cb = st.slider("colsample_bytree", 0.4,
                           1.0, 0.8, 0.1, key='xgb_cb')
            ra = st.number_input("reg_alpha", 0.0, 10.0,
                                 0.1, 0.05, key='xgb_ra')
            rl = st.number_input("reg_lambda", 0.0, 10.0,
                                 1.0, 0.1, key='xgb_rl')
        model_params = dict(
            n_estimators=ne, learning_rate=lr, max_depth=depth,
            min_child_weight=mcw, subsample=ss, colsample_bytree=cb,
            reg_alpha=ra, reg_lambda=rl,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
        model_name = 'xgboost'
    elif model_choice == 'ExtraTrees':
        with st.expander("⚙ ExtraTrees Settings"):
            ne = st.slider("n_estimators", 100, 2000, 600, 100, key='et_ne')
            md = st.slider("max_depth", 3, 40, 14, 1, key='et_md')
            msl = st.slider("min_samples_leaf", 1, 20, 2, 1, key='et_msl')
            mss = st.slider("min_samples_split", 2, 20, 4, 1, key='et_mss')
            mf = st.selectbox(
                "max_features", ['sqrt', 'log2', None], index=0, key='et_mf')
        model_params = dict(
            n_estimators=ne, max_depth=md, min_samples_leaf=msl,
            min_samples_split=mss, max_features=mf,
            random_state=42, n_jobs=-1)
        model_name = 'extratrees'
    else:  # Stacking Ensemble
        with st.expander("⚙ Stacking Ensemble Settings"):
            st.caption(
                "Menggabungkan LightGBM + RandomForest "
                "(+ CatBoost jika tersedia) dengan Ridge meta-learner. "
                "Training lebih lambat, tapi akurasi umumnya lebih baik.")
            st_lgb_ne = st.slider("LightGBM n_estimators", 200, 1000, 500, 100,
                                  key='st_lgb_ne')
            st_lgb_lr = st.select_slider("LightGBM learning_rate",
                                         [0.01, 0.02, 0.03, 0.05, 0.08],
                                         value=0.03, key='st_lgb_lr')
            st_rf_ne = st.slider("RF n_estimators", 100, 800, 300, 100,
                                 key='st_rf_ne')
            st_ridge_a = st.number_input("Ridge alpha (meta-learner)",
                                         0.01, 10.0, 1.0, 0.1, key='st_ridge_a')
            st_cv = st.slider("CV folds (internal)", 3, 7, 5, 1, key='st_cv')
        model_params = dict(
            lgb_n_estimators=st_lgb_ne, lgb_learning_rate=st_lgb_lr,
            rf_n_estimators=st_rf_ne, ridge_alpha=st_ridge_a,
            cv=st_cv, n_jobs=-1)
        model_name = 'stacking'

    # ── Target Rules (aturan filter per target saat training) ──
    with st.expander("📋 Target Rules (Training)", expanded=False):
        st.caption(
            "Rule membatasi data training per target. "
            "Nonaktifkan jika ingin model tetap belajar dari nilai ekstrem.")
        rule_vsh_drop_zero = st.checkbox(
            "Buang VSH = 0 (coal marker)",
            value=True, key='rule_vsh_drop_zero')
        rule_sw_drop_one = st.checkbox(
            "Buang SW = 1",
            value=False, key='rule_sw_drop_one')
        rule_phie_drop_zero = st.checkbox(
            "Buang PHIE = 0",
            value=False, key='rule_phie_drop_zero')
    target_rules = {
        'rule_vsh_drop_zero': rule_vsh_drop_zero,
        'rule_sw_drop_one': rule_sw_drop_one,
        'rule_phie_drop_zero': rule_phie_drop_zero,
    }

    st.markdown("---")
    can_train = (combined is not None and len(combined) > 0
                 and len(test_wells) > 0
                 and len(sel_tgts) > 0
                 and any(len(v) > 0 for v in target_feats.values())
                 and not (model_choice == 'CatBoost' and CatBoostRegressor is None)
                 and not (model_choice == 'XGBoost' and XGBRegressor is None))
    train_btn = st.button("▶  Jalankan Training",
                          disabled=not can_train,
                          use_container_width=True)
    if not can_train:
        miss = []
        if not all_wells:
            miss.append("upload ZIP")
        if not test_wells:
            miss.append("pilih test well")
        if not sel_tgts:
            miss.append("pilih target")
        if miss:
            st.markdown(f'<div class="ibox wbox">Perlu: {" · ".join(miss)}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TRAINING TRIGGER
# ══════════════════════════════════════════════════════════════════
if train_btn and can_train:
    with st.spinner("Melatih model..."):
        res = run_training(
            combined,
            test_wells, target_feats, opts, model_params, model_name=model_name,
            training_mode='hybrid_vsh_linear_residual' if training_mode == 'Hybrid Residual VSH (Linear)' else 'standard',
            target_rules=target_rules)

    # ✅ Cek apakah training berhasil (bukan None)
    if res is not None:
        st.session_state.update(results=res, trained=True,
                                cfg=dict(test_wells=test_wells,
                                         targets=sel_tgts,
                                         target_feats=target_feats,
                                         opts=opts,
                                         model_name=model_choice,
                                         model_params=model_params,
                                         training_mode=training_mode))

        # ── Propagate predictions back to combined_df ──
        # Sehingga VSH_PRED bisa dipakai sebagai feature untuk PHIE/SW
        _cdf = st.session_state.get('combined_df')
        if _cdf is not None:
            _cdf = _cdf.copy()
            for _pred_col in ['VSH_PRED', 'PHIE_PRED', 'SW_PRED']:
                if _pred_col not in _cdf.columns:
                    _cdf[_pred_col] = np.nan
            for _src_df in [res['df_tr'], res['df_te']]:
                if _src_df is None or len(_src_df) == 0:
                    continue
                for _pred_col in ['VSH_PRED', 'PHIE_PRED', 'SW_PRED']:
                    if _pred_col not in _src_df.columns:
                        continue
                    for _w in _src_df['WELL_NAME'].unique():
                        _src_w = _src_df.loc[
                            (_src_df['WELL_NAME'] ==
                             _w) & _src_df[_pred_col].notna(),
                            ['DEPTH', _pred_col]
                        ]
                        if _src_w.empty:
                            continue
                        _cdf_idx = _cdf.index[_cdf['WELL_NAME'] == _w]
                        if len(_cdf_idx) == 0:
                            continue
                        # Map by nearest depth
                        _depth_map = dict(
                            zip(_src_w['DEPTH'].values, _src_w[_pred_col].values))
                        for _idx in _cdf_idx:
                            _d = _cdf.at[_idx, 'DEPTH']
                            if _d in _depth_map:
                                _cdf.at[_idx, _pred_col] = _depth_map[_d]
            st.session_state['combined_df'] = _cdf
        st.success(
            "✅ Training selesai! Prediksi telah disimpan ke data gabungan.")
    else:
        st.session_state['trained'] = False


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="logo">Petro·ML</div>'
            '<div class="logo-sub" style="margin-bottom:.55rem;">'
            'Petrophysics Machine Learning Dashboard</div><hr>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA SUMMARY STRIP
# ══════════════════════════════════════════════════════════════════
if combined is not None and len(combined) > 0:
    n_w = len(all_wells)
    n_r = len(combined)
    html = '<div class="kpi-row">'
    html += (f'<div class="kpi"><div class="kl">Sumur</div>'
             f'<div class="kv ok">{n_w}</div>'
             f'<div class="ks">dalam ZIP</div></div>')
    html += (f'<div class="kpi"><div class="kl">Total Baris</div>'
             f'<div class="kv">{n_r:,}</div>'
             f'<div class="ks">setelah QC</div></div>')

    # Status QC
    qc_done = bool(st.session_state.get('qc_log'))
    html += (f'<div class="kpi"><div class="kl">QC Pipeline</div>'
             f'<div class="kv {"good" if qc_done else "na"}">{"✓" if qc_done else "—"}</div>'
             f'<div class="ks">{"selesai" if qc_done else "belum dijalankan"}</div></div>')

    # Status GR_NORM
    norm_done = st.session_state.get('normalized', False) or (
        'GR_NORM' in combined.columns and combined['GR_NORM'].notna().any())
    html += (f'<div class="kpi"><div class="kl">GR_NORM</div>'
             f'<div class="kv {"good" if norm_done else "na"}">{"✓" if norm_done else "—"}</div>'
             f'<div class="ks">{"tersedia" if norm_done else "belum dinormalisasi"}</div></div>')

    for lg in ['GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT']:
        if lg in combined.columns and combined[lg].notna().any():
            p = combined[lg].notna().mean()*100
            html += (f'<div class="kpi"><div class="kl">{lg}</div>'
                     f'<div class="kv {"ok" if p>50 else "bad"}">{p:.0f}%</div>'
                     f'<div class="ks">non-null</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("📋 Ringkasan per Sumur"):
        agg = {'N': ('DEPTH', 'count'),
               'Depth_Min': ('DEPTH', 'min'),
               'Depth_Max': ('DEPTH', 'max')}
        show_cols = ['GR', 'GR_NORM', 'VSH_LINEAR',
                     'NPHI', 'RHOB', 'RT'] + ALL_TARGETS
        for c in show_cols:
            if c in combined.columns:
                agg[c] = (c, lambda x, c=c: f"{x.notna().mean()*100:.0f}%")
        summ = combined.groupby('WELL_NAME').agg(**agg).reset_index()
        summ['Role'] = summ['WELL_NAME'].apply(
            lambda w: '🔬 Test' if w in test_wells else '🏋 Train')
        st.dataframe(summ, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# RESULT TABS
# ══════════════════════════════════════════════════════════════════
if st.session_state['trained'] and st.session_state['results']:
    res = st.session_state['results']
    cfg = st.session_state['cfg']

    if 'df_te' not in res or 'df_tr' not in res:
        st.warning(
            "⚠ Hasil training tidak lengkap — klik ▶ Jalankan Training ulang.")
        st.stop()

    df_te = res['df_te'] if res['df_te'] is not None else pd.DataFrame()
    df_tr = res['df_tr']

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Metrics & Importance",
        "🪵 Log Plot",
        "🎯 Scatter Plot",
        "💾 Export",
        "📦 Download Model",
        "🧪 Model Testing",
        "📋 ML Workflow",
    ])

    # ── TAB 1 ──
    with tab1:
        st.markdown("### Model Performance")
        mets = res['metrics']
        if mets:
            audit = res.get('train_audit', {})
            for tgt, wm in mets.items():
                st.markdown(f"**{tgt}**")
                html = '<div class="kpi-row">'
                for w, m in wm.items():
                    r2 = m['R2']
                    cls = r2c(r2)
                    r2s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                    html += (f'<div class="kpi"><div class="kl">{w}</div>'
                             f'<div class="kv {cls}">{r2s}</div>'
                             f'<div class="ks">RMSE={m["RMSE"]:.4f} · '
                             f'MAE={m["MAE"]:.4f} · N={m["N"]:,}</div></div>')
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)
                if tgt in audit:
                    st.caption(
                        f"Train rows: {audit[tgt]['n_train_rows']:,} · OOF rows: {audit[tgt]['n_oof_rows']:,} · Train wells: {audit[tgt]['n_train_wells']}")
        else:
            st.markdown('<div class="ibox">Label aktual tidak tersedia di '
                        'test well — metrics tidak dapat dihitung.</div>',
                        unsafe_allow_html=True)

        st.markdown("### Feature Importance")
        fig_fi = plot_fi(res['feat_imp'])
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── Metrics per Zone ──
        st.markdown("### Performance per Zone")
        if len(df_te) > 0 and 'ZONE' in df_te.columns:
            zm = compute_zone_metrics(df_te, cfg['targets'], zone_col='ZONE')
            if zm:
                for tgt, zdf in zm.items():
                    st.markdown(f"**{tgt}**")
                    html = '<div class="kpi-row">'
                    for _, row in zdf.iterrows():
                        r2 = row['R2']
                        cls = r2c(r2)
                        r2s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                        html += (
                            f'<div class="kpi"><div class="kl">{row["ZONE"]}</div>'
                            f'<div class="kv {cls}">{r2s}</div>'
                            f'<div class="ks">RMSE={row["RMSE"]:.4f} · '
                            f'MAE={row["MAE"]:.4f} · N={int(row["N"]):,}</div></div>')
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
                with st.expander("📋 Tabel Detail per Zone"):
                    for tgt, zdf in zm.items():
                        st.markdown(f"**{tgt}**")
                        st.dataframe(
                            zdf.style.format(
                                {'R2': '{:.4f}', 'RMSE': '{:.5f}',
                                 'MAE': '{:.5f}', 'N': '{:,.0f}'},
                                na_rep='—'),
                            use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    '<div class="ibox">Tidak cukup data per zona untuk '
                    'menghitung metrics (min 5 baris per zona).</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ibox">Data test tidak memiliki kolom ZONE.</div>',
                unsafe_allow_html=True)

        st.markdown("### Config")
        # Feature per target summary
        tf = cfg.get('target_feats', {})
        for tgt, feats in tf.items():
            pills = ''.join(f'<span class="pill">{f}</span>' for f in feats)
            st.markdown(
                f'<div class="ibox"><b>{tgt}</b>: {pills}</div>',
                unsafe_allow_html=True)

        n_tr_w = len([w for w in well_list if w not in cfg['test_wells']])
        st.markdown(
            f'<div class="ibox">'
            f'🏋 Training: <b>{n_tr_w}</b> sumur &nbsp;·&nbsp; '
            f'🔬 Test: <b>{len(cfg["test_wells"])}</b> sumur<br>'
            f'Target: {" · ".join(cfg["targets"])}<br>'
            f'Model: {cfg.get("model_name", "LightGBM")}<br>'
            f'Mode: {cfg.get("training_mode", "Standard ML")}<br>'
            f'Features: {pills}</div>',
            unsafe_allow_html=True)

    # ── TAB 2 ──
    with tab2:
        st.markdown("### Log Plot — Aktual vs Prediksi")
        plot_wells = (sorted(df_te['WELL_NAME'].unique()) if len(df_te) > 0
                      else sorted(df_tr['WELL_NAME'].unique()))
        sel_w = st.selectbox("Pilih Sumur", plot_wells, key='pw')
        df_src = df_te if (
            len(df_te) > 0 and sel_w in df_te['WELL_NAME'].values) else df_tr
        df_w = df_src[df_src['WELL_NAME'] == sel_w].copy()

        if len(df_w) > 0:
            dmin = float(df_w['DEPTH'].min())
            dmax = float(df_w['DEPTH'].max())
            c1, c2 = st.columns(2)
            with c1:
                d0 = st.number_input("Dari (m)", dmin, dmax, dmin, key='d0')
            with c2:
                d1 = st.number_input("Sampai (m)", dmin, dmax, dmax, key='d1')
            df_w = df_w[(df_w['DEPTH'] >= d0) & (df_w['DEPTH'] <= d1)]

            # Zone filter
            zf_log = None
            if 'ZONE' in df_w.columns:
                _zones = sorted([z for z in df_w['ZONE'].dropna().unique()
                                 if str(z).upper()
                                 not in ('UNKNOWN', 'NAN', '', 'NONE')])
                if _zones:
                    sel_zones = st.multiselect(
                        "Filter Zone (kosong = semua)",
                        _zones, default=[], key='log_zone_filter')
                    if sel_zones:
                        zf_log = sel_zones
            st.plotly_chart(
                plot_log(df_w, cfg['targets'], sel_w, zone_filter=zf_log),
                use_container_width=True)

    # ── TAB 3 ──
    with tab3:
        st.markdown("### Scatter — Aktual vs Prediksi")
        sc_wells = sorted(df_te['WELL_NAME'].unique()
                          ) if len(df_te) > 0 else []
        if not sc_wells:
            st.markdown('<div class="ibox wbox">Tidak ada data test '
                        'untuk scatter plot.</div>', unsafe_allow_html=True)
        else:
            sel_sw = st.multiselect("Filter Sumur", sc_wells, default=sc_wells)
            df_sc = df_te[df_te['WELL_NAME'].isin(sel_sw)]

            # Zone filter
            zf_sc = None
            if 'ZONE' in df_sc.columns:
                _zones_sc = sorted([z for z in df_sc['ZONE'].dropna().unique()
                                    if str(z).upper()
                                    not in ('UNKNOWN', 'NAN', '', 'NONE')])
                if _zones_sc:
                    sel_zones_sc = st.multiselect(
                        "Filter Zone (kosong = semua)",
                        _zones_sc, default=[], key='sc_zone_filter')
                    if sel_zones_sc:
                        zf_sc = sel_zones_sc
            fig_sc = plot_scatter(df_sc, cfg['targets'], zone_filter=zf_sc)
            if fig_sc:
                st.plotly_chart(fig_sc, use_container_width=True)

    # ── TAB 4 ──
    with tab4:
        st.markdown("### Download Hasil")
        choice = st.radio("Ekspor dari",
                          ["Test Wells", "Training Wells", "Semua Wells"],
                          horizontal=True)
        df_exp = (df_te if choice == "Test Wells" else
                  df_tr if choice == "Training Wells" else
                  pd.concat([df_tr, df_te], ignore_index=True)).copy()

        base_c = ['WELL_NAME', 'DEPTH', 'ZONE',
                  'MARKER', 'GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT']
        act_c = [t for t in cfg['targets'] if t in df_exp.columns]
        pred_c = [f'{t}_PRED' for t in cfg['targets']
                  if f'{t}_PRED' in df_exp.columns]
        drv_c = [c for c in ['LOG_RT', 'DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS', 'VSH_LINEAR']
                 if c in df_exp.columns]
        all_c = [c for c in base_c+act_c+pred_c+drv_c if c in df_exp.columns]
        df_out = df_exp[all_c].reset_index(drop=True)

        fmt = {c: '{:.4f}' for c in act_c+pred_c+drv_c if c in df_out.columns}
        st.dataframe(df_out.head(100).style.format(fmt, na_rep='—'),
                     use_container_width=True, height=300)
        st.caption(f"Preview 100 / {len(df_out):,} baris · {len(all_c)} kolom")
        st.download_button("⬇  Download CSV",
                           data=df_out.to_csv(index=False).encode('utf-8'),
                           file_name="petro_ml_predictions.csv",
                           mime="text/csv", use_container_width=True)

    # ── TAB 5: Download Model ──
    with tab5:
        st.markdown("### Download Model Package")
        st.markdown(
            '<div class="ibox">File <b>.pkl</b> berisi semua model yang sudah di-train, '
            'parameter GR normalisasi, zone encoder, dan konfigurasi. '
            'Bisa digunakan langsung di Python atau di tab <b>Model Testing</b>.</div>',
            unsafe_allow_html=True)

        if res.get('models'):
            # Model info summary
            for tgt, info in res['models'].items():
                model_type = type(info['model']).__name__
                n_feats = len(info['ft_tr'])
                st.markdown(
                    f'<div class="kpi" style="display:inline-block;margin:4px;">'
                    f'<div class="kl">{tgt}</div>'
                    f'<div class="kv ok" style="font-size:1rem;">{model_type}</div>'
                    f'<div class="ks">{n_feats} features: {", ".join(info["ft_tr"])}</div>'
                    f'</div>', unsafe_allow_html=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                pkg_bytes = export_model_package(
                    res, cfg, st.session_state.get('gr_norm_params', {}))
                st.download_button(
                    "⬇  Download Model (.pkl)",
                    data=pkg_bytes,
                    file_name="petro_ml_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True)
            with col_dl2:
                doc_md = generate_model_doc(cfg, res)
                st.download_button(
                    "⬇  Download Dokumentasi (.md)",
                    data=doc_md.encode('utf-8'),
                    file_name="petro_ml_model_README.md",
                    mime="text/markdown",
                    use_container_width=True)

            with st.expander("📖 Preview Dokumentasi", expanded=False):
                st.markdown(doc_md)

            # Config JSON export
            config_json = {
                'targets': cfg.get('targets', []),
                'model_name': cfg.get('model_name', 'LightGBM'),
                'training_mode': cfg.get('training_mode', 'Standard ML'),
                'feature_cols': {tgt: info['ft_tr'] for tgt, info in res['models'].items()},
                'feature_chosen': {tgt: info['ft_chosen'] for tgt, info in res['models'].items()},
                'opts': cfg.get('opts', {}),
                'target_bounds': TARGET_BOUNDS,
                'gr_norm_zones': list(st.session_state.get('gr_norm_params', {}).keys()),
            }
            st.download_button(
                "⬇  Download Config (.json)",
                data=json.dumps(config_json, indent=2,
                                default=str).encode('utf-8'),
                file_name="petro_ml_config.json",
                mime="application/json",
                use_container_width=True)

            # ── Per-target individual model download ──
            st.divider()
            st.markdown("### Download Model per Target")
            st.markdown(
                '<div class="ibox">Download model terpisah per target (VSH, PHIE, SW). '
                'Setiap file berisi 1 model + dokumentasi input log yang diperlukan + '
                'info cascading dependency. Cocok untuk migrasi ke sistem lain.</div>',
                unsafe_allow_html=True)

            gr_p = st.session_state.get('gr_norm_params', {})
            tgt_list = list(res['models'].keys())
            tgt_cols_dl = st.columns(len(tgt_list))
            for i_tgt, tgt_name in enumerate(tgt_list):
                with tgt_cols_dl[i_tgt]:
                    tgt_info = res['models'][tgt_name]
                    st.markdown(f"**{tgt_name}**")
                    st.caption(f"Model: {type(tgt_info['model']).__name__}")
                    st.caption(f"Features: {len(tgt_info['ft_tr'])}")

                    pkg_single = export_single_target_package(
                        tgt_name, res, cfg, gr_p)
                    st.download_button(
                        f"⬇ Model {tgt_name} (.pkl)",
                        data=pkg_single,
                        file_name=f"petro_ml_{tgt_name.lower()}_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                        key=f'dl_single_{tgt_name}')

                    doc_single = generate_single_target_doc(tgt_name, res, cfg)
                    st.download_button(
                        f"⬇ Docs {tgt_name} (.md)",
                        data=doc_single.encode('utf-8'),
                        file_name=f"petro_ml_{tgt_name.lower()}_README.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key=f'dl_single_doc_{tgt_name}')

            # Preview docs per target
            for tgt_name in tgt_list:
                with st.expander(f"📖 Preview Docs — {tgt_name}", expanded=False):
                    st.markdown(generate_single_target_doc(tgt_name, res, cfg))

        else:
            st.markdown('<div class="ibox wbox">Belum ada model yang di-train.</div>',
                        unsafe_allow_html=True)

    # ── TAB 6: Model Testing ──
    with tab6:
        st.markdown("### Model Testing — Prediksi pada Well Baru")
        st.markdown(
            '<div class="ibox">Upload file <b>.pkl</b> (dari tab Download Model) '
            'dan file <b>.las</b> (satu atau beberapa) untuk menjalankan prediksi. '
            'Atau gunakan model yang baru saja di-train.</div>',
            unsafe_allow_html=True)

        # Model source selection
        model_src = st.radio(
            "Sumber Model",
            ["Gunakan model aktif (baru di-train)", "Upload file .pkl"],
            horizontal=True, key='model_src')

        test_package = None
        if model_src == "Upload file .pkl":
            pkl_up = st.file_uploader("Upload Model (.pkl)", type=[
                                      'pkl'], key='pkl_upload')
            if pkl_up is not None:
                try:
                    test_package = pickle.load(pkl_up)
                    st.success(
                        f"✅ Model loaded — targets: {', '.join(test_package.get('models', {}).keys())}")
                except Exception as e:
                    st.error(f"❌ Gagal load model: {e}")
        else:
            if res.get('models'):
                test_package = {
                    'models': {tgt: info['model'] for tgt, info in res['models'].items()},
                    'feature_cols': {tgt: info['ft_tr'] for tgt, info in res['models'].items()},
                    'feature_chosen': {tgt: info['ft_chosen'] for tgt, info in res['models'].items()},
                    'config': {
                        'targets': cfg.get('targets', []),
                        'model_name': cfg.get('model_name', 'LightGBM'),
                        'training_mode': cfg.get('training_mode', 'Standard ML'),
                        'opts': cfg.get('opts', {}),
                    },
                    'gr_norm_params': st.session_state.get('gr_norm_params', {}),
                    'le_zone': res.get('le_zone'),
                    'target_bounds': TARGET_BOUNDS,
                    'mnemonic_map': MNEMONIC_MAP,
                }
                st.success(
                    f"✅ Model aktif — targets: {', '.join(res['models'].keys())}")
            else:
                st.markdown('<div class="ibox wbox">Belum ada model aktif. '
                            'Train dulu atau upload file .pkl.</div>',
                            unsafe_allow_html=True)

        # LAS upload
        las_files = st.file_uploader(
            "Upload LAS files", type=['las'], accept_multiple_files=True,
            key='test_las_upload',
            help="Upload satu atau beberapa file .las untuk prediksi")

        # Optional: structure name (untuk model multi-structure)
        test_struct_name = st.text_input(
            "Nama Struktur (opsional)",
            value="", key='test_struct_name',
            help="Isi jika model di-train multi-structure. "
                 "ZONE akan dikombinasi: ZONE_STRUCTURE (misal Upper TAF_BN)")

        # Optional zone CSV for test
        test_zone_up = st.file_uploader("Zone CSV untuk test (opsional)",
                                        type=['csv'], key='test_zone_csv')
        test_zone_df = None
        if test_zone_up is not None:
            try:
                test_zone_df = read_zone_csv(test_zone_up)
                st.success(f"✅ Zone: {len(test_zone_df)} baris")
            except Exception as e:
                st.warning(f"⚠ Zone CSV error: {e}")

        run_test_btn = st.button("▶  Jalankan Prediksi",
                                 disabled=(
                                     test_package is None or not las_files),
                                 use_container_width=True, key='run_test_btn')

        if run_test_btn and test_package and las_files:
            test_results = {}
            test_metrics = {}
            progress = st.progress(0, text="Memproses LAS files...")

            _struct = test_struct_name.strip().upper() if test_struct_name.strip() else None

            for i, las_file in enumerate(las_files):
                wname = las_file.name.rsplit('.', 1)[0].upper()
                wname = _normalize_well_name(wname)
                progress.progress((i + 1) / len(las_files),
                                  text=f"Memproses {wname}...")

                content = las_file.read()
                df_well = read_las_bytes(content, wname)
                if df_well is None:
                    st.warning(f"⚠ Skip {wname}: gagal baca LAS")
                    continue

                # Set STRUCTURE jika diisi (untuk kombinasi ZONE_STRUCTURE)
                if _struct:
                    df_well['STRUCTURE'] = _struct

                # Predict
                df_pred = predict_with_package(
                    test_package, df_well,
                    zone_df=test_zone_df)
                test_results[wname] = df_pred

                # Compute metrics if actual values exist
                targets = list(test_package['models'].keys())
                well_metrics = {}
                for tgt in targets:
                    if tgt in df_pred.columns and f'{tgt}_PRED' in df_pred.columns:
                        m = safe_m(df_pred[tgt], df_pred[f'{tgt}_PRED'])
                        if m['N'] >= 5:
                            well_metrics[tgt] = m
                if well_metrics:
                    test_metrics[wname] = well_metrics

            progress.empty()

            if not test_results:
                st.error("❌ Tidak ada well yang berhasil diproses.")
            else:
                st.session_state['test_results'] = test_results
                st.session_state['test_metrics'] = test_metrics
                st.success(
                    f"✅ Prediksi selesai untuk {len(test_results)} well")

        # Display results
        _tr = st.session_state.get('test_results')
        _tm = st.session_state.get('test_metrics')
        if _tr:
            targets = list(
                test_package['models'].keys()) if test_package else []
            well_names = sorted(_tr.keys())

            # Metrics summary
            if _tm:
                st.markdown("#### Metrics per Well")
                for wname in well_names:
                    if wname not in _tm:
                        continue
                    wm = _tm[wname]
                    html = f'<div style="margin-bottom:6px;"><b>{wname}</b></div><div class="kpi-row">'
                    for tgt, m in wm.items():
                        r2 = m['R2']
                        cls = r2c(r2)
                        r2s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                        html += (f'<div class="kpi"><div class="kl">{tgt}</div>'
                                 f'<div class="kv {cls}">{r2s}</div>'
                                 f'<div class="ks">RMSE={m["RMSE"]:.4f} · '
                                 f'MAE={m["MAE"]:.4f} · N={m["N"]:,}</div></div>')
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

            if not _tm and targets:
                st.markdown(
                    '<div class="ibox">Tidak ada label aktual di well test — '
                    'metrics tidak dihitung (prediksi tetap tersedia).</div>',
                    unsafe_allow_html=True)

            # Log Plot per well
            st.markdown("#### Log Plot")
            sel_test_w = st.selectbox(
                "Pilih Well", well_names, key='test_plot_well')
            if sel_test_w and sel_test_w in _tr:
                df_tw = _tr[sel_test_w]
                if len(df_tw) > 0:
                    dmin_t = float(df_tw['DEPTH'].min())
                    dmax_t = float(df_tw['DEPTH'].max())
                    ct1, ct2 = st.columns(2)
                    with ct1:
                        dt0 = st.number_input(
                            "Dari (m)", dmin_t, dmax_t, dmin_t, key='td0')
                    with ct2:
                        dt1 = st.number_input(
                            "Sampai (m)", dmin_t, dmax_t, dmax_t, key='td1')
                    df_tw_filt = df_tw[(df_tw['DEPTH'] >= dt0)
                                       & (df_tw['DEPTH'] <= dt1)]

                    # Zone filter
                    zf_tst = None
                    if 'ZONE' in df_tw_filt.columns:
                        _ztst = sorted([
                            z for z in df_tw_filt['ZONE'].dropna().unique()
                            if str(z).upper()
                            not in ('UNKNOWN', 'NAN', '', 'NONE')])
                        if _ztst:
                            sel_ztst = st.multiselect(
                                "Filter Zone (kosong = semua)",
                                _ztst, default=[], key='test_log_zone_filter')
                            if sel_ztst:
                                zf_tst = sel_ztst
                    st.plotly_chart(
                        plot_log(df_tw_filt, targets, sel_test_w,
                                 zone_filter=zf_tst),
                        use_container_width=True)

            # Crossplot
            if _tm:
                st.markdown("#### Scatter Plot — Aktual vs Prediksi")
                sel_test_sc = st.multiselect("Filter Wells", well_names,
                                             default=well_names, key='test_scatter_wells')
                if sel_test_sc:
                    df_sc_all = pd.concat(
                        [_tr[w] for w in sel_test_sc if w in _tr],
                        ignore_index=True)

                    # Zone filter
                    zf_tst_sc = None
                    if 'ZONE' in df_sc_all.columns:
                        _ztstsc = sorted([
                            z for z in df_sc_all['ZONE'].dropna().unique()
                            if str(z).upper()
                            not in ('UNKNOWN', 'NAN', '', 'NONE')])
                        if _ztstsc:
                            sel_ztstsc = st.multiselect(
                                "Filter Zone (kosong = semua)",
                                _ztstsc, default=[],
                                key='test_sc_zone_filter')
                            if sel_ztstsc:
                                zf_tst_sc = sel_ztstsc
                    fig_sc_test = plot_scatter(df_sc_all, targets,
                                               zone_filter=zf_tst_sc)
                    if fig_sc_test:
                        st.plotly_chart(fig_sc_test, use_container_width=True)

            # Download predictions
            st.markdown("#### Download Hasil Prediksi")
            df_all_test = pd.concat(
                [_tr[w] for w in well_names], ignore_index=True)
            base_cols = ['WELL_NAME', 'DEPTH', 'ZONE', 'MARKER',
                         'GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT']
            tgt_cols = []
            for t in targets:
                if t in df_all_test.columns:
                    tgt_cols.append(t)
                if f'{t}_PRED' in df_all_test.columns:
                    tgt_cols.append(f'{t}_PRED')
            drv_cols = [c for c in ['VSH_LINEAR', 'LOG_RT', 'DN_SEP']
                        if c in df_all_test.columns]
            export_cols = [c for c in base_cols + tgt_cols + drv_cols
                           if c in df_all_test.columns]
            df_test_out = df_all_test[export_cols].reset_index(drop=True)

            st.dataframe(df_test_out.head(
                50), use_container_width=True, height=250)
            st.caption(f"Preview 50 / {len(df_test_out):,} baris")
            st.download_button(
                "⬇  Download Prediksi Test CSV",
                data=df_test_out.to_csv(index=False).encode('utf-8'),
                file_name="petro_ml_test_predictions.csv",
                mime="text/csv", use_container_width=True,
                key='dl_test_csv')

    # ── ML Workflow ──
    with tab7:
        render_ml_workflow(cfg=cfg, res=res, mode='single')

else:
    # empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
      justify-content:center;height:46vh;gap:16px;opacity:0.5;">
      <div style="font-size:3.2rem;">🪨</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;
        color:#7d8590;text-align:center;line-height:1.9;">
        Upload ZIP → Pilih Sumur Test → Atur Feature<br>
        <span style="font-size:.74rem;color:#484f58;">
          klik ▶ Jalankan Training di sidebar
        </span>
      </div>
    </div>""", unsafe_allow_html=True)
