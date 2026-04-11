"""
Petro·ML  —  Petrophysics Machine Learning Dashboard
Upload 1 ZIP → pilih sumur test → merge zone/marker CSV → train → predict
"""

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
import lasio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
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

ALL_LOGS = ['GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT']
ALL_TARGETS = ['VSH', 'PHIE', 'SW']
TARGET_BOUNDS = {
    'VSH': (0.0, 1.0),
    'PHIE': (0.0, 0.5),
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
    'zip_hash': None,   # ← tambah
    'zone_hash': None,   # ← tambah
    'marker_hash': None,   # ← tambah
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


def compute_gr_norm_params(df: pd.DataFrame, p_low: float = 3,
                           p_high: float = 97) -> dict:
    if 'GR' not in df.columns or 'ZONE' not in df.columns:
        return {}

    params = {}
    gr_valid = df[df['GR'].notna()].copy()

    # Global fallback (semua zona termasuk UNKNOWN)
    g_low = float(np.nanpercentile(gr_valid['GR'], p_low))
    g_high = float(np.nanpercentile(gr_valid['GR'], p_high))

    for zone, grp in gr_valid.groupby('ZONE'):
        # ✅ Skip UNKNOWN — baris ini tidak punya zona valid
        # tidak representatif untuk normalisasi per zona
        if str(zone).upper() in ('UNKNOWN', 'NAN', '', 'NONE'):
            continue

        gr_zone = grp['GR'].dropna()
        if len(gr_zone) < 20:
            params[zone] = {
                'p_low': g_low, 'p_high': g_high,
                'source': 'global_fallback', 'N': int(len(gr_zone))
            }
        else:
            params[zone] = {
                'p_low': float(np.percentile(gr_zone, p_low)),
                'p_high': float(np.percentile(gr_zone, p_high)),
                'source': 'zone',
                'N': int(len(gr_zone)),
            }

    # UNKNOWN tetap ada sebagai fallback untuk baris yang tidak punya zona
    # tapi pakai global P3/P97, bukan dihitung sendiri
    params['UNKNOWN'] = {
        'p_low': g_low, 'p_high': g_high,
        'source': 'global_fallback', 'N': int(len(gr_valid))
    }
    return params


def apply_gr_norm(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Terapkan normalisasi GR per zona ke df.
    GR_NORM = (GR − p_low) / (p_high − p_low)
    Clipped ke [0, 1].
    """
    df = df.copy()
    df['GR_NORM'] = np.nan

    for zone, p in params.items():
        mask = df['ZONE'].fillna('UNKNOWN').astype(str) == zone
        if mask.sum() == 0:
            continue
        rng = p['p_high'] - p['p_low']
        if rng <= 0:
            df.loc[mask, 'GR_NORM'] = 0.5
        else:
            df.loc[mask, 'GR_NORM'] = (
                (df.loc[mask, 'GR'] - p['p_low']) / rng
            ).clip(0, 1)

    # Baris yang zonanya tidak ada di params → pakai global
    global_p = params.get('UNKNOWN', list(params.values())[0]
                          if params else None)
    if global_p:
        no_zone = df['GR_NORM'].isna() & df['GR'].notna()
        if no_zone.sum() > 0:
            rng = global_p['p_high'] - global_p['p_low']
            if rng > 0:
                df.loc[no_zone, 'GR_NORM'] = (
                    (df.loc[no_zone, 'GR'] - global_p['p_low']) / rng
                ).clip(0, 1)
    return df

# ══════════════════════════════════════════════════════════════════
# TARGET POLICY
# ══════════════════════════════════════════════════════════════════


def apply_target_training_policy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """
    Policy khusus per target untuk training/evaluasi.
    Saat ini:
    - VSH = 0 dianggap coal marker -> tidak dipakai training/evaluasi umum
    - SW = 1 tetap dipertahankan
    """
    df = df.copy()
    info = {
        'drop_vsh_zero_for_training': 0,
        'keep_sw_eq_1': 0,
    }

    if target == 'VSH' and 'VSH' in df.columns:
        mask_zero = df['VSH'].notna() & (df['VSH'] == 0)
        info['drop_vsh_zero_for_training'] = int(mask_zero.sum())
        df = df.loc[~mask_zero].copy()

    if target == 'SW' and 'SW' in df.columns:
        info['keep_sw_eq_1'] = int((df['SW'] == 1).sum())

    return df, info

# ══════════════════════════════════════════════════════════════════
# QC PIPELINE  (dari notebook A6)
# ══════════════════════════════════════════════════════════════════


def run_qc_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    if model_name == 'randomforest':
        rf_params = params.copy()
        rf_params.pop('verbosity', None)
        return RandomForestRegressor(**rf_params)
    if model_name == 'catboost':
        if CatBoostRegressor is None:
            raise ImportError("CatBoost belum terinstall di environment ini.")
        return CatBoostRegressor(**params)
    return lgb.LGBMRegressor(**params)


def clip_target_predictions(pred, target):
    lo, hi = TARGET_BOUNDS.get(target, (None, None))
    pred = np.asarray(pred, dtype=float)
    if lo is not None and hi is not None:
        pred = np.clip(pred, lo, hi)
    return pred


def make_oof_predictions(clean_df: pd.DataFrame, feature_cols: list, target_col: str,
                         model_name: str, model_params: dict) -> pd.Series:
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
            model.fit(tr[feature_cols], tr[target_col])
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
        model.fit(tr[feature_cols], tr[target_col])
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


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════
def run_training(combined, test_wells, target_feats, opts, params, model_name='lightgbm'):
    """
    Training per target dengan propagated feature non-leaky.
    VSH/PHIE/SW di train-set dipropagasikan memakai OOF prediction,
    bukan label aktual.
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

        # Policy khusus per target
        clean, tgt_policy = apply_target_training_policy(clean, tgt)

        if tgt == 'VSH' and tgt_policy.get('drop_vsh_zero_for_training', 0) > 0:
            st.info(
                f"ℹ VSH policy: {tgt_policy['drop_vsh_zero_for_training']:,} baris dengan VSH = 0 "
                f"tidak dipakai untuk training/evaluasi umum (diasumsikan coal)."
            )

        if len(clean) < 20:
            st.warning(f"⚠ {tgt}: hanya {len(clean)} baris training — skip")
            continue

        clean['_OOF_PRED_'] = make_oof_predictions(
            clean, feat_tr, tgt, model_name, params)
        df_tr.loc[clean.index, f'{tgt}_PRED'] = clean['_OOF_PRED_']

        model = build_model(model_name, params)
        model.fit(clean[feat_tr], clean[tgt])
        models[tgt] = {
            'model': model,
            'ft_tr': feat_tr,
            'ft_te': feat_te,
            'ft_chosen': chosen,
        }

        fi = getattr(model, 'feature_importances_', None)
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
                df_te.loc[pred_mask, f'{tgt}_PRED'] = clip_target_predictions(
                    model.predict(df_te.loc[pred_mask, feat_te]), tgt)
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
        }

    return dict(df_tr=df_tr, df_te=df_te, models=models,
                le_zone=le_zone, feat_imp=feat_imp,
                metrics=metrics, train_audit=train_audit)

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════


def plot_log(df, targets, well_name):
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


def plot_scatter(df, targets):
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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo">🪨 Petro·ML</div>'
                '<div class="logo-sub">petrophysics machine learning</div>',
                unsafe_allow_html=True)

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
    qc_btn = st.button("🧹  Jalankan QC Pipeline",
                       disabled=not can_qc, use_container_width=True)

    if qc_btn and can_qc:
        df_qc, qc_log = run_qc_pipeline(combined_raw.copy())
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
        p_low = st.number_input("Percentile low",  1, 20,  3, key='p_low')
    with c_p2:
        p_high = st.number_input("Percentile high", 80, 99, 97, key='p_high')

    if not has_zone:
        st.markdown('<div class="ibox wbox">Upload Zone CSV dulu agar '
                    'normalisasi per zona bisa dilakukan.</div>',
                    unsafe_allow_html=True)

    can_norm = has_gr and combined is not None
    norm_btn = st.button("📐  Hitung GR_NORM",
                         disabled=not can_norm, use_container_width=True)

    if norm_btn and can_norm:
        params = compute_gr_norm_params(combined, p_low=p_low, p_high=p_high)
        combined_normed = apply_gr_norm(combined, params)
        st.session_state['combined_df'] = combined_normed
        st.session_state['gr_norm_params'] = params
        st.session_state['normalized'] = True
        combined = combined_normed
        n_ok = combined_normed['GR_NORM'].notna().sum()
        st.success(f"✅ GR_NORM selesai — {n_ok:,} nilai valid")

    has_gr_norm_direct = (
        combined is not None and 'GR_NORM' in combined.columns and combined['GR_NORM'].notna().any())
    if st.session_state['normalized'] or has_gr_norm_direct:
        params = st.session_state.get('gr_norm_params', {})
        if params:
            n_zones = len(params)
            st.markdown(f'<div class="ibox">GR_NORM aktif · {n_zones} zona '
                        f'· P{p_low}/P{p_high}</div>',
                        unsafe_allow_html=True)

            with st.expander("📊 Parameter per Zona", expanded=False):
                rows = [{'Zona': z, 'N': p['N'],
                         f'P{p_low}': round(p["p_low"], 1),
                         f'P{p_high}': round(p["p_high"], 1),
                         'Sumber': p['source']}
                        for z, p in params.items()]
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
        'VSH': [f for f in [gr_feat, 'RHOB', 'DN_SEP', 'NPHI_RHOB_CROSS', 'ZONE_ENC']
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
        options=['LightGBM', 'RandomForest', 'CatBoost'],
        index=0,
        help='Gunakan baseline pembanding selain LightGBM bila diperlukan.'
    )
    if model_choice == 'CatBoost' and CatBoostRegressor is None:
        st.markdown('<div class="ibox wbox">CatBoost belum tersedia di environment ini. '
                    'Pilih LightGBM atau RandomForest, atau install catboost dulu.</div>',
                    unsafe_allow_html=True)

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
    else:
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

    st.markdown("---")
    can_train = (combined is not None and len(combined) > 0
                 and len(test_wells) > 0
                 and len(sel_tgts) > 0
                 and any(len(v) > 0 for v in target_feats.values())
                 and not (model_choice == 'CatBoost' and CatBoostRegressor is None))
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
            test_wells, target_feats, opts, model_params, model_name=model_name)

    # ✅ Cek apakah training berhasil (bukan None)
    if res is not None:
        st.session_state.update(results=res, trained=True,
                                cfg=dict(test_wells=test_wells,
                                         targets=sel_tgts,
                                         target_feats=target_feats,
                                         opts=opts,
                                         model_name=model_choice))
        st.success("✅ Training selesai!")
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
        show_cols = ['GR', 'GR_NORM', 'NPHI', 'RHOB', 'RT'] + ALL_TARGETS
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrics & Importance",
        "🪵 Log Plot",
        "🎯 Scatter Plot",
        "💾 Export",
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
            st.plotly_chart(plot_log(df_w, cfg['targets'], sel_w),
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
            fig_sc = plot_scatter(df_sc, cfg['targets'])
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
        drv_c = [c for c in ['LOG_RT', 'DN_SEP', 'NPHI_RHOB_CROSS', 'CROSS_POS']
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
