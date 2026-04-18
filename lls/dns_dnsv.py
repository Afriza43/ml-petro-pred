# app.py

import numpy as np
import pandas as pd
from autoplot import calculate_nphi_rhob_intersection


# ============================================================
# DNS CUTOFF Equations per STRUKTUR and ZONE
# Source: 2026-02-26 - GOWS Equations summary.pptx
# Format: {STRUKTUR: {ZONE: {"var": "PHIE"|"GR", "intercept": float, "slope": float}}}
# DNS_CUTOFF = intercept + slope * PHIE  (or GR for ABAB)
# ============================================================
DNS_CUTOFF_EQUATIONS = {
    "Gunung Kemala": {
        "TAF1": {"var": "PHIE", "intercept": -0.053, "slope": 0.89},
        "TAF2": {"var": "PHIE", "intercept": -0.01, "slope": 0.61},
    },
    "Abab": {
        "BRF": {"var": "GR", "intercept": 0.14, "slope": -0.001},
        "TAF": {"var": "GR", "intercept": 0.22, "slope": -0.002},
    },
    "Benuang": {
        "TAF": {"var": "PHIE", "intercept": -0.06, "slope": 1.25},
    },
    "Mangun Jaya": {
        "TAF": {"var": "PHIE", "intercept": -0.06, "slope": 0.24},
    },
    "Lembak": {
        "TAF": {"var": "PHIE", "intercept": -0.08, "slope": 1.02},
    },
    "Karangan": {
        "TAF": {"var": "PHIE", "intercept": -0.18, "slope": 1.30},
    },
    "Talang Jimar": {
        "TAF": {"var": "PHIE", "intercept": -0.13, "slope": 1.28},
    },
    "Belimbing": {
        "TAF1": {"var": "PHIE", "intercept": -0.06, "slope": 1.09},
        # TAF2: Data oil sedikit dan tidak ada data gas
        # TAF3: Tidak ada HC
    },
    "Bentayan": {
        "TAF": {"var": "PHIE", "intercept": 0.039, "slope": 0.61},
    },
    # TANJUNG TIGA BARAT: Tidak ada data gas (no DNS_CUTOFF)
    # TANJUNG MIRING BARAT: Tidak ada data gas (no DNS_CUTOFF)
    "Benakat Barat": {
        "GUF": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_1": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_2": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_3": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
        "GUF_4": {"var": "PHIE", "intercept": -0.185, "slope": 1.26},
    },
    "Limau Barat": {
        "TAF1": {"var": "PHIE", "intercept": -0.081, "slope": 0.796},
        "TAF2": {"var": "PHIE", "intercept": -0.012, "slope": 0.70},
    },
    "Limau Tengah": {
        "TAF": {"var": "PHIE", "intercept": -0.05, "slope": 0.8},
    },
    "Beringin": {
        "GUF": {"var": "PHIE", "intercept": -0.2579, "slope": 1.357},
        "BRF": {"var": "PHIE", "intercept": -0.037, "slope": 0.67},
        "TAF": {"var": "PHIE", "intercept": -0.078, "slope": 1.34},
    },
    "Prabumenang": {
        "BRF": {"var": "PHIE", "intercept": 0.09, "slope": 0.574},
    },
    "Musi": {
        "BRF": {"var": "PHIE", "intercept": 0.016, "slope": 0.72},
    },
    "Betung": {
        "TAF": {"var": "PHIE", "intercept": -0.10, "slope": 0.99},
    },
    "Niru": {
        "TAF": {"var": "PHIE", "intercept": -0.02, "slope": 0.79},
    },
    "Prabumulih Barat": {
        "TAF": {"var": "PHIE", "intercept": -0.005, "slope": 0.87},
    },
}


def dns(rhob_in, nphi_in):
    """Calculate DNS (Density-Neutron Separation)"""
    return ((2.71 - rhob_in) / 1.71) - nphi_in


def dnsv(rhob_in, nphi_in, rhob_sh, nphi_sh, vsh):
    """Calculate DNSV (Density-Neutron Separation corrected for shale Volume)"""
    rhob_corv = rhob_in + vsh * (2.65 - rhob_sh)
    nphi_corv = nphi_in + vsh * (0 - nphi_sh)
    return ((2.71 - rhob_corv) / 1.71) - nphi_corv


def calculate_dns_cutoff(df, struktur_col="STRUKTUR", zone_col="ZONE",
                         phie_col="PHIE", gr_col="GR"):
    """
    Calculate DNS_CUTOFF based on STRUKTUR and ZONE.
    Each row gets a DNS_CUTOFF value based on its structure's equation.
    Vectorized: groups by (STRUKTUR, ZONE) for performance.
    """
    dns_cutoff = pd.Series(np.nan, index=df.index)

    if struktur_col not in df.columns:
        print(
            f"Warning: Column '{struktur_col}' not found. DNS_CUTOFF will be NaN.")
        return dns_cutoff

    if zone_col not in df.columns:
        print(
            f"Warning: Column '{zone_col}' not found. DNS_CUTOFF will be NaN.")
        return dns_cutoff

    # Build uppercase lookup for fast matching
    eq_lookup = {}
    for struct_name, zones in DNS_CUTOFF_EQUATIONS.items():
        for zone_name, eq in zones.items():
            eq_lookup[(struct_name.upper(), zone_name.upper())] = eq

    # Vectorized: group by (STRUKTUR, ZONE) and apply equation per group
    strukt_upper = df[struktur_col].astype(str).str.strip().str.upper()
    zone_upper = df[zone_col].astype(str).str.strip().str.upper()

    for (s, z), eq in eq_lookup.items():
        mask = (strukt_upper == s) & (zone_upper == z)
        if not mask.any():
            continue

        var_col = phie_col if eq["var"] == "PHIE" else gr_col
        if var_col not in df.columns:
            continue

        vals = pd.to_numeric(df.loc[mask, var_col], errors="coerce")
        dns_cutoff.loc[mask] = eq["intercept"] + eq["slope"] * vals

    return dns_cutoff


def process_dns_dnsv(
    df: pd.DataFrame,
    params: dict = None,
    target_intervals: list = None,
    target_zones: list = None,
) -> pd.DataFrame:
    """
    Main function to process DNS-DNSV analysis with internal filtering.
    """
    if params is None:
        params = {}

    try:
        df_processed = df.copy()

        # Get column names from parameters
        nphi_col = params.get("NPHI_IN", "NPHI")
        rhob_col = params.get("RHOB_IN", "RHOB")

        # Output column names
        dns_col = params.get("DNS", "DNS")

        # 1. Prepare data and parameters
        # Make process idempotent by dropping old results
        df_processed.drop(
            columns=[dns_col, "DNSV", "DNS_CUTOFF", "FLUID_DNS"], inplace=True, errors="ignore")

        # Rename VSH_LINEAR if it exists and VSH does not
        if "VSH_LINEAR" in df_processed.columns and "VSH" not in df_processed.columns:
            df_processed["VSH"] = df_processed["VSH_LINEAR"]

        # Ensure required columns exist before proceeding
        required_cols = [rhob_col, nphi_col, "VSH"]
        missing_cols = [
            col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            print(
                f"Warning: Required columns {missing_cols} not found. Skipping calculation."
            )
            return df

        # Coerce to numeric, turning errors into NaN
        for col in required_cols:
            df_processed[col] = pd.to_numeric(
                df_processed[col], errors="coerce")

        # Calculate shale point from the full dataset for consistency
        # Use the specified column names for the intersection calculation
        shale_point = calculate_nphi_rhob_intersection(
            df_processed, params.get(
                "prcntz_qz", 5), params.get("prcntz_wtr", 5)
        )
        nphi_sh = shale_point["nphi_sh"]
        rhob_sh = shale_point["rhob_sh"]

        # 2. Create a mask to select rows for calculation
        mask = pd.Series(True, index=df_processed.index)
        has_filters = False
        if target_intervals and "MARKER" in df_processed.columns:
            mask = df_processed["MARKER"].isin(target_intervals)
            has_filters = True
        if target_zones and "ZONE" in df_processed.columns:
            zone_mask = df_processed["ZONE"].isin(target_zones)
            mask = (mask | zone_mask) if has_filters else zone_mask

        # Also, ensure we only calculate on valid data points
        valid_data_mask = df_processed[required_cols].notna().all(axis=1)
        final_mask = mask & valid_data_mask

        if not final_mask.any():
            print(
                "Warning: No data matched the filter criteria. No calculations performed."
            )
            return df

        # 3. Perform calculations only on the masked (selected) rows
        print(f"Calculating DNS-DNSV for {final_mask.sum()} rows.")

        rhob_masked = df_processed.loc[final_mask, rhob_col]
        nphi_masked = df_processed.loc[final_mask, nphi_col]
        vsh_masked = df_processed.loc[final_mask, "VSH"]

        df_processed.loc[final_mask, dns_col] = dns(rhob_masked, nphi_masked)
        df_processed.loc[final_mask, "DNSV"] = dnsv(
            rhob_masked, nphi_masked, rhob_sh, nphi_sh, vsh_masked
        )

        # 4. Calculate DNS_CUTOFF based on STRUKTUR and ZONE equations
        if "STRUKTUR" in df_processed.columns:
            phie_col = params.get("PHIE", "PHIE")
            gr_col = params.get("GR", "GR")
            masked_df = df_processed.loc[final_mask]
            df_processed.loc[final_mask, "DNS_CUTOFF"] = calculate_dns_cutoff(
                masked_df, struktur_col="STRUKTUR", zone_col="ZONE",
                phie_col=phie_col, gr_col=gr_col,
            )
            cutoff_count = df_processed.loc[final_mask,
                                            "DNS_CUTOFF"].notna().sum()
            print(f"Calculated DNS_CUTOFF for {cutoff_count} rows "
                  f"(out of {final_mask.sum()} total).")
        else:
            print("Warning: STRUKTUR column not found. DNS_CUTOFF not calculated.")

        # 5. Calculate FLUID_DNS based on DNS vs DNS_CUTOFF (only where IQUAL > 0)
        if "IQUAL" in df_processed.columns and "DNS_CUTOFF" in df_processed.columns:
            df_processed["FLUID_DNS"] = np.nan
            iqual_num = pd.to_numeric(df_processed["IQUAL"], errors="coerce")
            dns_vals = pd.to_numeric(df_processed[dns_col], errors="coerce")
            cutoff_vals = pd.to_numeric(
                df_processed["DNS_CUTOFF"], errors="coerce")

            iqual_mask = iqual_num.fillna(0) > 0
            valid = iqual_mask & dns_vals.notna() & cutoff_vals.notna()

            # G = Gas when DNS > DNS_CUTOFF, O = Oil when DNS < DNS_CUTOFF
            gas_mask = valid & (dns_vals > cutoff_vals)
            oil_mask = valid & (dns_vals < cutoff_vals)
            df_processed.loc[gas_mask, "FLUID_DNS"] = "G"
            df_processed.loc[oil_mask, "FLUID_DNS"] = "O"

            print(f"Calculated FLUID_DNS: {gas_mask.sum()} Gas, "
                  f"{oil_mask.sum()} Oil rows.")
        else:
            print("Warning: IQUAL or DNS_CUTOFF missing. FLUID_DNS not calculated.")

        return df_processed

    except Exception as e:
        print(f"Error in process_dns_dnsv: {str(e)}")
        raise e
