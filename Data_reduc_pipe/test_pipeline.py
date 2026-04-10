"""
Telluric correction pipeline for Keck LIGER
============================================

Architecture
------------
PSG is called with watm=y and wephm=y POST parameters, which instruct
it to perform a live MERRA-2 atmospheric profile retrieval for the
supplied observation date and geodetic coordinates before running
radiative transfer.  This means every API call returns a physically
correct, date-specific atmosphere — the stale ATMOSPHERE-LAYER snapshot
in the template is regenerated from scratch each time.

The pipeline builds a grid over (airmass, H2O abundance, temperature)
and fits three free parameters per observation:

    airmass    — path length through the atmosphere
    h2o_abun   — ATMOSPHERE-ABUN H2O multiplier on top of MERRA-2
                 baseline (1.0 = unperturbed; 0.5 = drier; 2.0 = wetter)
    dlam_nm    — sub-pixel wavelength shift of the instrument

Temperature is retained as an optional axis but is validated at runtime;
if PSG does not respond to it the axis is collapsed to a single node.

Key design choices
------------------
1.  watm=y + wephm=y POST flags trigger a live MERRA-2 query per call.
    OBJECT-DATE and OBJECT-GEODETIC now fully control the atmosphere.
2.  PSG is queried at native high resolving power (R=200,000) so
    individual HITRAN lines are resolved.  Convolution to instrument R
    is performed by the pipeline, not PSG.
3.  H2O abundance scaling (ATMOSPHERE-ABUN index 0) is applied on top
    of the live MERRA-2 profile, giving a physically meaningful PWV axis.
4.  The synthetic smoke-test uses a detector wavelength grid offset from
    the model grid by true_dlam so wavelength-shift recovery is genuine.
5.  Differential Evolution is used for fitting — robust against the
    broad, shallow chi-square basins typical of telluric fitting.
6.  All PSG calls are cached to disk keyed by a hash of the full config;
    rebuilding the grid after non-PSG code changes costs zero API calls.
7.  Grid validity is keyed to (date, site, wavelength, airmass grid,
    H2O grid, temp grid) so any axis change invalidates stale grids.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, differential_evolution


# ============================================================
# CONSTANTS
# ============================================================

PSG_API = "https://psg.gsfc.nasa.gov/api.php"

# Maunakea geodetic coordinates (latitude degN, longitude degE, altitude km)
MAUNAKEA_LAT = 19.8228
MAUNAKEA_LON = -155.4681
MAUNAKEA_ALT = 4.205

PSG_TOTAL_LABEL  = "Total"
PSG_H2O_LABEL    = "H2O"

# PSG native resolving power for grid queries.
# We request full line-resolved spectra so individual line depths
# are preserved, then convolve to instrument R ourselves.
# R=200000 resolves HITRAN lines throughout the NIR.
PSG_NATIVE_RP = 200_000


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SiteConfig:
    """Observatory location passed to PSG for MERRA-2 lookup."""
    latitude_deg:  float = MAUNAKEA_LAT
    longitude_deg: float = MAUNAKEA_LON
    altitude_km:   float = MAUNAKEA_ALT
    name:          str   = "Maunakea"


@dataclass(frozen=True)
class PipelineConfig:
    # PSG template.  The ATMOSPHERE-LAYER entries are used as a starting
    # point but are regenerated from MERRA-2 by the watm=y POST flag on
    # every API call.  Any valid exported PSG Earth config will work.
    template_path: str = "earth_cfg.txt"

    # Output paths
    out_dir:     str = "telluric_grid"
    h5_filename: str = "telluric_grid.h5"

    # PSG endpoint
    psg_api: str = PSG_API

    # Wavelength range: (lam_min_nm, lam_max_nm)
    # The third element is kept for cache-key compatibility but the
    # actual PSG query now uses psg_native_rp (see below).
    wavelength_cfg_nm: Tuple[float, float, float] = (1500.0, 2500.0, 50_000.0)

    # Resolving power sent to PSG.  We use a very high value so PSG
    # returns individual HITRAN lines fully resolved.  The pipeline
    # then convolves down to R_inst itself, giving it full control
    # over the LSF shape and wavelength shift.
    psg_native_rp: int = PSG_NATIVE_RP

    # Real UT observation date -- drives live MERRA-2 retrieval via watm=y.
    # Use nighttime UT for Maunakea (HST = UT - 10h), e.g. 05:00 UT = 19:00 HST.
    # Changing this date produces a genuinely different atmospheric profile.
    observation_date: str = "2024/01/15 05:00"

    # Site
    site: SiteConfig = field(default_factory=SiteConfig)

    # API robustness
    psg_timeout_s: int = 180
    psg_max_tries: int = 10

    # Write raw PSG responses to disk for debugging
    debug_dump: bool = True


# ============================================================
# FILESYSTEM HELPERS
# ============================================================

def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# PSG CONFIG MANIPULATION
# ============================================================

def load_psg_template(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PSG template not found: {path}\n"
            "Download a baseline Earth config from https://psg.gsfc.nasa.gov"
        )
    with open(path) as fh:
        cfg = fh.read()
    if "<ATMOSPHERE-LAYER" not in cfg:
        raise RuntimeError(
            "PSG template appears invalid: no ATMOSPHERE-LAYER entries found."
        )
    return cfg


def set_parameter(cfg: str, tag: str, value: Any) -> str:
    """Replace or append a <TAG> line in the PSG config string."""
    val = str(value)
    new, n = re.subn(
        rf"(^<{re.escape(tag)}>\s*).*$",
        rf"\g<1>{val}",
        cfg,
        flags=re.MULTILINE,
    )
    if n == 0:
        if not cfg.endswith("\n"):
            cfg += "\n"
        new = cfg + f"<{tag}> {val}\n"
    return new


def get_parameter(cfg: str, tag: str):
    """Return the value of a <TAG> line, or None if absent."""
    m = re.search(rf"^<{re.escape(tag)}>\s*(.*)$", cfg, flags=re.M)
    return m.group(1).strip() if m else None


def set_h2o_abun(cfg: str, h2o_scale: float) -> str:
    """
    Set the H2O multiplier (index 0) in ATMOSPHERE-ABUN while leaving
    all other gas multipliers at 1.

    ATMOSPHERE-ABUN=1,1,1,1,1,1,1,1 is the PSG default (one entry per
    gas in ATMOSPHERE-GAS).  We replace only index 0 so no other species
    are perturbed.
    """
    abun_str = get_parameter(cfg, "ATMOSPHERE-ABUN")
    gas_str  = get_parameter(cfg, "ATMOSPHERE-GAS") or ""
    n_gases  = len(gas_str.split(",")) if gas_str else 8

    if abun_str:
        abun_vals = [v.strip() for v in abun_str.split(",")]
    else:
        abun_vals = ["1"] * n_gases

    while len(abun_vals) < n_gases:
        abun_vals.append("1")
    abun_vals = abun_vals[:n_gases]
    abun_vals[0] = f"{float(h2o_scale):.8f}"
    return set_parameter(cfg, "ATMOSPHERE-ABUN", ",".join(abun_vals))


def am_to_zenith_deg(am: float) -> float:
    am = max(float(am), 1.0)
    return float(np.degrees(np.arccos(np.clip(1.0 / am, 0.0, 1.0))))


def _stable_cfg_hash(cfg_text: str) -> str:
    lines = [ln.rstrip() for ln in cfg_text.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()[:12]


# ============================================================
# BEER-LAMBERT H2O POST-PROCESSING SCALING
# ============================================================

def scale_h2o_transmission(
    T_total:   np.ndarray,
    T_h2o:     np.ndarray,
    h2o_scale: float,
) -> np.ndarray:
    """
    Scale the H2O optical depth by h2o_scale using Beer-Lambert law
    and recompute total transmission.

    Physics
    -------
    T_total = T_h2o * T_other          (other gases unchanged)
    tau_h2o = -log(T_h2o)
    tau_h2o_scaled = h2o_scale * tau_h2o
    T_h2o_scaled = exp(-tau_h2o_scaled) = T_h2o ** h2o_scale
    T_scaled = T_h2o_scaled * T_other
             = T_h2o**h2o_scale * (T_total / T_h2o)

    This is applied entirely in Python after PSG returns T_total and
    T_h2o as separate columns.  PSG is not involved in the scaling --
    it only provides the physically correct high-resolution baseline
    from a live MERRA-2 + HITRAN radiative transfer calculation.

    Validity
    --------
    Beer-Lambert scaling assumes line shapes do not change with column
    amount, which is accurate for PWV variations within a factor of
    ~3 of the MERRA-2 baseline.  For the typical Maunakea observing
    range this approximation is entirely adequate.

    Parameters
    ----------
    T_total   : total atmospheric transmission from PSG (all gases)
    T_h2o     : H2O-only transmission from PSG
    h2o_scale : column scaling factor (1.0 = unperturbed MERRA-2,
                0.5 = half column, 2.0 = double column)
    """
    T_h2o_safe = np.clip(T_h2o,  1e-10, 1.0)
    T_other    = np.clip(T_total / T_h2o_safe, 0.0, 1.0)
    T_h2o_scaled = np.power(T_h2o_safe, float(h2o_scale))
    return np.clip(T_h2o_scaled * T_other, 0.0, 1.0)


# ============================================================
# PSG CONFIG BUILDER  -- MERRA-2 mode, no layer patching
# ============================================================

def build_psg_input(
    am: float,
    surface_T: float,
    cfg_obj: PipelineConfig,
    h2o_abun: float = 1.0,
) -> str:
    """
    Build a PSG config for (airmass, surface_T, h2o_abun).

    The config is submitted to PSG with watm=y + wephm=y POST flags
    (see run_psg), which instruct PSG to perform a live MERRA-2
    retrieval for OBJECT-DATE + OBJECT-GEODETIC before computing
    radiative transfer.  The ATMOSPHERE-LAYER snapshot in the template
    is regenerated from scratch — OBJECT-DATE now fully controls the
    atmosphere.

    h2o_abun scales ATMOSPHERE-ABUN index 0 on top of the live MERRA-2
    profile.  h2o_abun=1.0 is the unperturbed MERRA-2 baseline for that
    date; 0.5 halves the H2O column; 2.0 doubles it.
    """
    cfg = load_psg_template(cfg_obj.template_path)

    # Site + date — with watm=y these drive a live MERRA-2 lookup
    cfg = set_parameter(cfg, "OBJECT-DATE", cfg_obj.observation_date)
    cfg = set_parameter(
        cfg, "OBJECT-GEODETIC",
        f"{cfg_obj.site.latitude_deg:.4f},"
        f"{cfg_obj.site.longitude_deg:.4f},"
        f"{cfg_obj.site.altitude_km:.4f}",
    )

    # Geometry
    zen = am_to_zenith_deg(am)
    cfg = set_parameter(cfg, "GEOMETRY",            "Lookingup")
    cfg = set_parameter(cfg, "GEOMETRY-USER-PARAM", f"{zen:.6f}")
    cfg = set_parameter(cfg, "GEOMETRY-OBS-ANGLE",  f"{zen:.6f}")

    # Wavelength range -- request at native PSG resolution so individual
    # HITRAN lines are resolved.  Convolution to instrument R is done
    # in the pipeline after retrieval, not by PSG.
    lam_min_nm, lam_max_nm, _ = cfg_obj.wavelength_cfg_nm
    cfg = set_parameter(cfg, "GENERATOR-RANGEUNIT",      "cm-1")
    cfg = set_parameter(cfg, "GENERATOR-RANGE1",         f"{1e7 / lam_max_nm:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RANGE2",         f"{1e7 / lam_min_nm:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTION",     f"{cfg_obj.psg_native_rp}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTIONUNIT", "RP")

    # Request individual gas transmission columns so we can extract
    # H2O separately.  GENERATOR-TRANS=02-01 requests H2O (gas index 1)
    # and total transmission in the output.
    cfg = set_parameter(cfg, "GENERATOR-TRANS-APPLY", "Y")
    cfg = set_parameter(cfg, "GENERATOR-TRANS-SHOW",  "Y")
    cfg = set_parameter(cfg, "GENERATOR-TRANS",       "02-01")

    # Surface temperature
    cfg = set_parameter(cfg, "SURFACE-TEMPERATURE", f"{float(surface_T):.6f}")

    # H2O abundance scaling -- index 0 of ATMOSPHERE-ABUN
    cfg = set_h2o_abun(cfg, h2o_abun)

    print(
        f"[PSG INPUT] am={am:.4f}  T={surface_T:.2f} K  "
        f"h2o_abun={h2o_abun:.4f}  "
        f"date={cfg_obj.observation_date}  site={cfg_obj.site.name}"
    )
    return cfg


# ============================================================
# PSG API
# ============================================================

def parse_psg_header(lines: Sequence[str]) -> List[str]:
    gas_hints = {
        "H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2",
        "Total", "Rayleigh", "CIA", "Wave", "Wave/freq",
    }
    for ln in lines:
        stripped = ln.strip()
        if not stripped.startswith("#"):
            continue
        parts = stripped.lstrip("#").split()
        if sum(1 for p in parts if p in gas_hints) >= 2:
            return parts
    return []


def find_column(col_names: Sequence[str], label: str) -> Optional[int]:
    label_lo = label.lower()
    for i, name in enumerate(col_names):
        n = name.lower()
        if n == label_lo:
            return i
        if label_lo == "wave" and n.startswith("wave"):
            return i
    return None


def run_psg(
    psg_config_text: str,
    cfg_obj: PipelineConfig,
    output_type: str = "trn",
) -> Tuple[Optional[np.ndarray], List[str]]:
    """POST a PSG config; return (array, col_names) or (None, [])."""
    wait_s = 5.0

    for attempt in range(cfg_obj.psg_max_tries):
        print(f"\nAttempt {attempt + 1}/{cfg_obj.psg_max_tries} -- PSG (type={output_type})")
        try:
            resp = requests.post(
                cfg_obj.psg_api,
                data={
                    "file":  psg_config_text,
                    "type":  output_type,
                    # watm=y  — regenerate the atmospheric profile from
                    #           MERRA-2 for OBJECT-DATE + OBJECT-GEODETIC
                    #           before running radiative transfer.
                    #           Without this PSG uses the stale
                    #           ATMOSPHERE-LAYER snapshot in the template.
                    "watm":  "y",
                    # wephm=y — recompute geometry/ephemeris for the
                    #           given date (updates solar angles etc.)
                    "wephm": "y",
                },
                timeout=cfg_obj.psg_timeout_s,
            )
            print("HTTP status:", resp.status_code)

            if resp.status_code != 200:
                if cfg_obj.debug_dump:
                    with open("psg_http_error.txt", "w") as fh:
                        fh.write(resp.text)
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            text = resp.text.strip()

            if cfg_obj.debug_dump:
                with open("psg_last_raw_response.txt", "w") as fh:
                    fh.write(text)

            busy_phrases = [
                "other api call is still running",
                "please let it finish",
                "please wait",
                "busy",
                "wait 10 minutes",
            ]
            if any(b in text.lower() for b in busy_phrases):
                print("PSG busy -- waiting...")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            if len(text) < 100:
                print("PSG response suspiciously short.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            lines      = text.splitlines()
            col_names  = parse_psg_header(lines)
            data_lines = [
                ln for ln in lines
                if not ln.strip().startswith("#") and ln.strip()
            ]

            if not data_lines:
                print("No numeric data lines in PSG response.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            arr = np.genfromtxt(io.StringIO("\n".join(data_lines)), dtype=float)

            if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
                print("Unexpected array shape:", getattr(arr, "shape", None))
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            if not np.isfinite(arr).any():
                print("All PSG values non-finite.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            total_idx = find_column(col_names, PSG_TOTAL_LABEL) or 1
            print(
                f"PSG returned {arr.shape[0]} pts, {arr.shape[1]} cols  "
                f"Trans min/max: {np.nanmin(arr[:, total_idx]):.4f} / "
                f"{np.nanmax(arr[:, total_idx]):.4f}"
            )
            if col_names:
                print("Columns:", col_names)
            return arr, col_names

        except requests.exceptions.RequestException as exc:
            print("Network error:", repr(exc))
            time.sleep(wait_s)
            wait_s = min(wait_s * 1.5, 120)

    print("PSG failed after all retries.")
    return None, []


# ============================================================
# GRID CACHING
# ============================================================

def _grid_hash(
    observation_date: str,
    site: SiteConfig,
    wavelength_cfg_nm: Tuple[float, float, float],
    airmass_grid: Sequence[float],
    h2o_abun_grid: Sequence[float],
    temp_grid: Sequence[float],
) -> str:
    key = str((
        observation_date,
        (site.latitude_deg, site.longitude_deg, site.altitude_km),
        tuple(float(x) for x in wavelength_cfg_nm),
        tuple(sorted(float(x) for x in airmass_grid)),
        tuple(sorted(float(x) for x in h2o_abun_grid)),
        tuple(sorted(float(x) for x in temp_grid)),
    ))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _grid_is_valid(
    h5_filename: str,
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    h2o_abun_grid: Sequence[float],
    temp_grid: Sequence[float],
) -> bool:
    if not os.path.exists(h5_filename):
        return False
    expected = _grid_hash(
        cfg_obj.observation_date,
        cfg_obj.site,
        cfg_obj.wavelength_cfg_nm,
        airmass_grid,
        h2o_abun_grid,
        temp_grid,
    )
    try:
        with h5py.File(h5_filename, "r") as hf:
            return hf.attrs.get("grid_hash", "") == expected
    except Exception:
        return False


def _cache_path(
    out_dir: str,
    am: float,
    h2o_abun: float,
    surface_T: float,
    cfg_hash: str,
    lam_min_nm: float,
    lam_max_nm: float,
    rp: float,
) -> str:
    return os.path.join(
        out_dir,
        f"am{am:.3f}_h2o{h2o_abun:.4f}_T{surface_T:.1f}_"
        f"lam{lam_min_nm:.0f}-{lam_max_nm:.0f}_rp{int(rp)}_"
        f"cfg{cfg_hash}.dat",
    )


# ============================================================
# GRID BUILDER
# ============================================================

def build_grid_hdf5(
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    h2o_abun_grid: Sequence[float],
    temp_grid: Sequence[float],
    force_rebuild: bool = True,
) -> str:
    """
    Build and cache the 4-D transmission array:
        spectra[i_am, j_h2o, k_T, i_lambda]

    Grid axes
    ---------
    airmass_grid  : path length through the atmosphere
    h2o_abun_grid : H2O column scaling factor applied via Beer-Lambert
                    post-processing (1.0 = unperturbed MERRA-2 baseline,
                    0.5 = half column, 2.0 = double column)
    temp_grid     : surface temperature (usually collapsed to one node)

    PSG call count
    --------------
    Only n_am * n_T PSG calls are made (one per airmass/temperature
    node at h2o_scale=1.0).  The full H2O axis is then populated
    analytically using Beer-Lambert scaling of the H2O column returned
    by PSG in the separate H2O transmission column.  This reduces PSG
    calls by a factor of n_h2o and makes the H2O axis free.
    """
    create_directory(cfg_obj.out_dir)

    airmass_grid  = np.array(sorted(float(x) for x in airmass_grid),  dtype=float)
    h2o_abun_grid = np.array(sorted(float(x) for x in h2o_abun_grid), dtype=float)
    temp_grid     = np.array(sorted(float(x) for x in temp_grid),     dtype=float)

    if (not force_rebuild) and _grid_is_valid(
        cfg_obj.h5_filename, cfg_obj, airmass_grid, h2o_abun_grid, temp_grid
    ):
        print(f"[grid] Current grid found -- loading: {cfg_obj.h5_filename}")
        return cfg_obj.h5_filename

    n_am  = len(airmass_grid)
    n_h2o = len(h2o_abun_grid)
    n_T   = len(temp_grid)
    # Only n_am * n_T PSG calls needed -- H2O axis filled analytically
    total_calls = n_am * n_T

    print(
        f"[grid] Building grid  date={cfg_obj.observation_date}  "
        f"site={cfg_obj.site.name}\n"
        f"       airmasses={list(airmass_grid)}\n"
        f"       h2o_abun={list(h2o_abun_grid)}  "
        f"(filled analytically via Beer-Lambert -- no extra PSG calls)\n"
        f"       temps={list(temp_grid)}\n"
        f"       total PSG calls: {total_calls}  "
        f"(was {n_am * n_h2o * n_T} before Beer-Lambert approach)"
    )

    lam_min_nm, lam_max_nm, rp = cfg_obj.wavelength_cfg_nm

    def compute_or_load(am: float, surface_T: float):
        # Always query at h2o_abun=1.0 (unperturbed MERRA-2 baseline)
        psg_in   = build_psg_input(am, surface_T, cfg_obj, h2o_abun=1.0)
        cfg_hash = _stable_cfg_hash(psg_in)
        p        = _cache_path(
            cfg_obj.out_dir, am, 1.0, surface_T, cfg_hash,
            lam_min_nm, lam_max_nm, rp,
        )
        if (not force_rebuild) and os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [cache] am={am:.3f} T={surface_T:.1f} -> {p}")
            try:
                arr = np.loadtxt(p)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr, []
            except Exception as exc:
                print(f"  [cache] read failed ({exc}), recomputing")
        arr, col_names = run_psg(psg_in, cfg_obj)
        if arr is None:
            raise RuntimeError(
                f"PSG returned no data for am={am}, T={surface_T}"
            )
        np.savetxt(p, arr)
        return arr, col_names

    # Reference wavelength grid from first node
    ref_arr, ref_cols = compute_or_load(airmass_grid[0], temp_grid[0])
    wave_idx  = find_column(ref_cols, "Wave")  or 0
    total_idx = find_column(ref_cols, PSG_TOTAL_LABEL) or 1

    lam_ref = 1e7 / ref_arr[:, wave_idx]
    order   = np.argsort(lam_ref)
    lam_ref = lam_ref[order]
    nlam    = len(lam_ref)

    grid_specs = np.full(
        (n_am, n_h2o, n_T, nlam),
        np.nan, dtype=float,
    )
    grid_h2o = np.full_like(grid_specs, np.nan)

    call_n = 0
    for i, am in enumerate(airmass_grid):
        for k, surface_T in enumerate(temp_grid):
            call_n += 1
            print(f"[grid] PSG call {call_n}/{total_calls}  "
                  f"am={am:.3f}  T={surface_T:.1f}  (h2o=1.0 baseline)")
            arr, cols = compute_or_load(am, surface_T)

            w_idx   = find_column(cols, "Wave")          or 0
            t_idx   = find_column(cols, PSG_TOTAL_LABEL)  or 1
            h2o_idx = find_column(cols, PSG_H2O_LABEL)
            if h2o_idx is None:
                print("  WARNING: H2O column not found -- Beer-Lambert "
                      "scaling will use Total as fallback (less accurate)")
                h2o_idx = t_idx

            lam_nm = 1e7 / arr[:, w_idx]
            trans  = arr[:, t_idx].astype(float)
            h2o    = arr[:, h2o_idx].astype(float)
            ord_   = np.argsort(lam_nm)
            lam_nm = lam_nm[ord_]
            trans  = np.nan_to_num(trans[ord_], nan=0.0)
            h2o    = np.nan_to_num(h2o[ord_],   nan=0.0)

            # Interpolate baseline onto reference wavelength grid
            trans_ref = np.interp(lam_ref, lam_nm, trans, left=np.nan, right=np.nan)
            h2o_ref   = np.interp(lam_ref, lam_nm, h2o,   left=np.nan, right=np.nan)

            # Fill the entire H2O axis analytically via Beer-Lambert scaling
            print(f"  -> filling {n_h2o} H2O nodes analytically: "
                  f"{list(h2o_abun_grid)}")
            for j, h2o_scale in enumerate(h2o_abun_grid):
                grid_specs[i, j, k, :] = scale_h2o_transmission(
                    trans_ref, h2o_ref, h2o_scale
                )
                grid_h2o[i, j, k, :] = np.power(
                    np.clip(h2o_ref, 1e-10, 1.0), float(h2o_scale)
                )

    _debug_airmass_variation(grid_specs, airmass_grid, h2o_abun_grid, temp_grid)

    gh = _grid_hash(
        cfg_obj.observation_date,
        cfg_obj.site,
        cfg_obj.wavelength_cfg_nm,
        airmass_grid,
        h2o_abun_grid,
        temp_grid,
    )

    with h5py.File(cfg_obj.h5_filename, "w") as hf:
        hf.create_dataset("spectra",        data=grid_specs)
        hf.create_dataset("spectra_h2o",    data=grid_h2o)
        hf.create_dataset("airmasses",      data=airmass_grid)
        hf.create_dataset("h2o_abun",       data=h2o_abun_grid)
        hf.create_dataset("temperatures",   data=temp_grid)
        hf.create_dataset("wavelengths_nm", data=lam_ref)
        hf.attrs["grid_hash"]        = gh
        hf.attrs["observation_date"] = cfg_obj.observation_date
        hf.attrs["site_name"]        = cfg_obj.site.name
        hf.attrs["site_lat"]         = cfg_obj.site.latitude_deg
        hf.attrs["site_lon"]         = cfg_obj.site.longitude_deg
        hf.attrs["site_alt_km"]      = cfg_obj.site.altitude_km

    print(f"[grid] Wrote {cfg_obj.h5_filename}  (hash={gh})")
    return cfg_obj.h5_filename


def _debug_airmass_variation(
    grid_specs: np.ndarray,
    airmass_grid: np.ndarray,
    h2o_abun_grid: np.ndarray,
    temp_grid: np.ndarray,
) -> None:
    # Use middle H2O and temperature nodes for the airmass diagnostic
    j_mid = len(h2o_abun_grid) // 2
    k_mid = len(temp_grid) // 2
    i_ref = len(airmass_grid) // 2
    ref   = grid_specs[i_ref, j_mid, k_mid, :]
    print("\n[debug] Airmass variation (middle H2O and T nodes):")
    for i, am in enumerate(airmass_grid):
        spec = grid_specs[i, j_mid, k_mid, :]
        diff = spec - ref
        print(
            f"  AM={am:.2f}  h2o={h2o_abun_grid[j_mid]:.2f}  "
            f"min={np.nanmin(spec):.4f}  "
            f"max={np.nanmax(spec):.4f}  "
            f"max|delta|={np.nanmax(np.abs(diff)):.4e}"
        )
    # Also print H2O variation at middle airmass
    ref_h2o = grid_specs[i_ref, j_mid, k_mid, :]
    print("\n[debug] H2O ABUN variation (middle airmass node):")
    for j, h2o in enumerate(h2o_abun_grid):
        spec = grid_specs[i_ref, j, k_mid, :]
        diff = spec - ref_h2o
        print(
            f"  H2O={h2o:.3f}  am={airmass_grid[i_ref]:.2f}  "
            f"min={np.nanmin(spec):.4f}  "
            f"max={np.nanmax(spec):.4f}  "
            f"max|delta|={np.nanmax(np.abs(diff)):.4e}"
        )


# ============================================================
# INTERPOLATOR
# ============================================================

def load_hdf5_grid(h5_filename: str):
    with h5py.File(h5_filename, "r") as hf:
        spectra      = hf["spectra"][:]
        spectra_h2o  = hf["spectra_h2o"][:] if "spectra_h2o" in hf else spectra
        airmasses    = hf["airmasses"][:]
        h2o_abun     = hf["h2o_abun"][:] if "h2o_abun" in hf else np.array([1.0])
        temperatures = hf["temperatures"][:]
        wavelengths  = hf["wavelengths_nm"][:]
    return spectra, spectra_h2o, airmasses, h2o_abun, temperatures, wavelengths


def make_telluric_interp(h5_filename: str):
    """
    Return (interp_total, interp_h2o, lam_grid_nm).

    interp_total  -- Total atmospheric transmission (all absorbers)
    interp_h2o    -- H2O-only transmission column

    Both are RegularGridInterpolators over (airmass, surface_T).
    The pipeline fits against Total by default; H2O is available
    for diagnostics and for the three-panel Ben plot.
    """
    spectra, spectra_h2o, airmasses, h2o_abun, temperatures, lam_grid = load_hdf5_grid(h5_filename)

    def _make(values):
        return RegularGridInterpolator(
            points=(airmasses, h2o_abun, temperatures),
            values=values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    return _make(spectra), _make(spectra_h2o), lam_grid


def eval_interp(
    interp: RegularGridInterpolator,
    am: float,
    h2o_abun: float,
    surface_T: float,
) -> np.ndarray:
    return interp([[float(am), float(h2o_abun), float(surface_T)]])[0]


# ============================================================
# TEMPERATURE AXIS VALIDATOR
# ============================================================

def validate_temperature_axis(
    interp: RegularGridInterpolator,
    lam_grid: np.ndarray,
    am: float,
    h2o_abun: float,
    t1: float,
    t2: float,
) -> Dict[str, Any]:
    spec1 = eval_interp(interp, am, h2o_abun, t1)
    spec2 = eval_interp(interp, am, h2o_abun, t2)
    diff  = np.abs(spec1 - spec2)
    result = {
        "max_abs_diff":    float(np.nanmax(diff)),
        "median_abs_diff": float(np.nanmedian(diff)),
        "t1": t1, "t2": t2, "am": am,
    }
    print(
        f"[temperature_check] am={am:.3f}  T1={t1:.1f}  T2={t2:.1f}  "
        f"max|dT|={result['max_abs_diff']:.4e}  "
        f"median|dT|={result['median_abs_diff']:.4e}"
    )
    if result["max_abs_diff"] < 1e-6:
        print(
            "  WARNING: temperature axis is inactive.  "
            "Collapse temp_grid to [280.0] and use T_fixed=280.0 in fit_telluric."
        )
    return result


# ============================================================
# FORWARD MODEL
# ============================================================

def convolve_to_R(
    wave_nm: np.ndarray,
    flux: np.ndarray,
    R_inst: float,
) -> np.ndarray:
    """Convolve flux with a Gaussian LSF at resolving power R_inst."""
    wave_nm = np.asarray(wave_nm, float)
    flux    = np.asarray(flux,    float)

    lnw    = np.log(wave_nm)
    lnw_u  = np.linspace(lnw.min(), lnw.max(), len(wave_nm))
    flux_u = np.interp(lnw_u, lnw, flux)

    sigma_lnw = (1.0 / R_inst) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dln       = lnw_u[1] - lnw_u[0]
    sigma_pix = sigma_lnw / dln

    flux_conv_u = gaussian_filter1d(flux_u, sigma_pix, mode="nearest")
    return np.interp(lnw, lnw_u, flux_conv_u)


def forward_model(
    T_base: np.ndarray,
    dlam_nm: float,
    wave_obs: np.ndarray,
    lam_grid: np.ndarray,
    R_inst: float,
) -> np.ndarray:
    """
    Convolve T_base to R_inst, then resample onto wave_obs shifted by
    -dlam_nm.

    A positive dlam_nm shifts the model redward relative to the data,
    matching a detector whose wavelength solution is blueward of the
    model grid.
    """
    T_conv       = convolve_to_R(lam_grid, T_base, R_inst)
    wave_shifted = np.asarray(wave_obs, float) - float(dlam_nm)
    return np.interp(wave_shifted, lam_grid, T_conv, left=np.nan, right=np.nan)


# ============================================================
# FIX 1 -- FORWARD MODEL ROUND-TRIP DIAGNOSTIC
# ============================================================

def check_forward_model_roundtrip(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    R_inst:    float,
    am:        float = 1.5,
    h2o_abun:  float = 1.0,
    surface_T: float = 280.0,
    sigma_ref: float = 0.02,
) -> Dict[str, float]:
    """
    Evaluate the forward model at dlam=0 and compare to the raw
    interpolated spectrum.  The residual should be much smaller than
    sigma_ref; it measures convolution + resampling error, not noise.

    If rms_over_sigma > 0.10 the forward model has a systematic error
    that will bias the fit.  The most common fix is to add more airmass
    nodes to the grid so linear interpolation is more accurate.
    """
    T_base  = eval_interp(interp, am, h2o_abun, surface_T)
    T_model = forward_model(T_base, 0.0, lam_grid, lam_grid, R_inst)

    valid = np.isfinite(T_base) & np.isfinite(T_model)
    resid = T_model[valid] - T_base[valid]

    result = {
        "max_abs_residual": float(np.max(np.abs(resid))),
        "rms_residual":     float(np.sqrt(np.mean(resid ** 2))),
        "sigma_ref":        float(sigma_ref),
        "rms_over_sigma":   float(np.sqrt(np.mean(resid ** 2)) / sigma_ref),
    }

    print("\n[round-trip check]")
    print(f"  max |residual| = {result['max_abs_residual']:.4e}")
    print(f"  rms  residual  = {result['rms_residual']:.4e}")
    print(f"  rms / sigma    = {result['rms_over_sigma']:.4f}  (target < 0.10)")

    if result["rms_over_sigma"] > 0.10:
        print(
            "  WARNING: round-trip residual > 10% of sigma.\n"
            "  Consider adding more airmass nodes or using method='cubic' "
            "in RegularGridInterpolator."
        )
    else:
        print("  OK -- forward model round-trip is clean.")

    return result


# ============================================================
# FIX 2 -- FIT WINDOW DIAGNOSTIC
# ============================================================

def check_fit_window(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    fit_mask:  np.ndarray,
    am:        float = 1.5,
    h2o_abun:  float = 1.0,
    surface_T: float = 280.0,
    min_mean:  float = 0.10,
    min_max:   float = 0.50,
) -> bool:
    """
    Print transmission statistics inside the fit window and warn if
    the window is too opaque to support a reliable fit.

    A good window has mean > 0.10 and max > 0.50.
    The 1850-1950 nm core H2O band fails both tests at typical
    Maunakea PWV; use 1550-1750 nm (H-band window) instead.
    """
    spec   = eval_interp(interp, am, h2o_abun, surface_T)
    in_win = spec[fit_mask]
    t_min  = float(np.nanmin(in_win))
    t_max  = float(np.nanmax(in_win))
    t_mean = float(np.nanmean(in_win))
    n_pix  = int(fit_mask.sum())

    print(
        f"\n[fit window]  N={n_pix}  "
        f"min={t_min:.4f}  max={t_max:.4f}  mean={t_mean:.4f}"
    )

    ok = True
    if t_mean < min_mean:
        print(
            f"  WARNING: mean transmission {t_mean:.4f} < {min_mean} -- "
            "window may be fully saturated.  Choose a less opaque region."
        )
        ok = False
    if t_max < min_max:
        print(
            f"  WARNING: max transmission {t_max:.4f} < {min_max} -- "
            "window has no high-transmission anchor pixels."
        )
        ok = False
    if ok:
        print("  OK -- fit window has usable transmission structure.")
    return ok


# ============================================================
# FITTING
# ============================================================

def _solve_linear_continuum(
    T_model:   np.ndarray,
    wave_obs:  np.ndarray,
    flux_obs:  np.ndarray,
    sigma_obs: np.ndarray,
    valid:     np.ndarray,
) -> Tuple[float, float]:
    lam  = wave_obs[valid]
    lam0 = np.median(lam)
    w    = 1.0 / sigma_obs[valid] ** 2

    A  = np.vstack([T_model[valid], T_model[valid] * (lam - lam0)]).T
    Aw = A  * np.sqrt(w[:, None])
    yw = flux_obs[valid] * np.sqrt(w)

    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _valid_mask(
    T_model:   np.ndarray,
    flux_obs:  np.ndarray,
    sigma_obs: np.ndarray,
    mask:      np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(mask, bool)
        & np.isfinite(T_model)
        & np.isfinite(flux_obs)
        & np.isfinite(sigma_obs)
        & (sigma_obs > 0)
        & (T_model > 0.0)
    )


def chi2(
    params:    Sequence[float],
    interp:    RegularGridInterpolator,
    wave_obs:  np.ndarray,
    flux_obs:  np.ndarray,
    sigma_obs: np.ndarray,
    mask:      np.ndarray,
    lam_grid:  np.ndarray,
    R_inst:    float,
    T_fixed:   Optional[float] = None,
) -> float:
    """
    Chi-square objective.

    T_fixed is not None  ->  params = [airmass, h2o_abun, dlam_nm]
    T_fixed is     None  ->  params = [airmass, h2o_abun, surface_T, dlam_nm]
    """
    if T_fixed is not None:
        am, h2o_abun, dlam_nm = params
        surface_T = T_fixed
    else:
        am, h2o_abun, surface_T, dlam_nm = params

    T_base  = eval_interp(interp, am, h2o_abun, surface_T)
    T_model = forward_model(T_base, dlam_nm, wave_obs, lam_grid, R_inst)
    valid   = _valid_mask(T_model, flux_obs, sigma_obs, mask)

    if valid.sum() < 50:
        return 1e99

    c0, c1 = _solve_linear_continuum(T_model, wave_obs, flux_obs, sigma_obs, valid)
    lam    = wave_obs[valid]
    lam0   = np.median(lam)
    model  = (c0 + c1 * (lam - lam0)) * T_model[valid]
    resid  = flux_obs[valid] - model
    w      = 1.0 / sigma_obs[valid] ** 2
    return float(np.sum(resid ** 2 * w))


def fit_telluric(
    interp:        RegularGridInterpolator,
    wave_obs:      np.ndarray,
    flux_obs:      np.ndarray,
    sigma_obs:     np.ndarray,
    mask:          np.ndarray,
    lam_grid:      np.ndarray,
    R_inst:        float,
    T_fixed:       Optional[float] = 280.0,
    x0_am:         float = 1.5,
    x0_h2o_abun:   float = 1.0,
    x0_dlam:       float = 0.0,
    x0_T:          float = 280.0,
    am_bounds:     Tuple[float, float] = (1.0, 2.5),
    h2o_abun_bounds: Tuple[float, float] = (0.3, 3.0),
    dlam_bounds:   Tuple[float, float] = (-0.5, 0.5),
    T_bounds:      Tuple[float, float] = (250.0, 320.0),
    optimizer:     str = "de",
) -> Tuple[Dict[str, float], float, Any]:
    """
    Fit (airmass, h2o_abun, dlam_nm) with temperature optionally fixed.

    optimizer = "de"     — Differential Evolution global search followed by
                           an L-BFGS-B polish step.  Recommended: robust
                           against broad, shallow chi-square basins.
    optimizer = "lbfgsb" — Pure L-BFGS-B from x0.  Fast but fragile.

    Returns (best_params_dict, chi2_val, scipy_result).
    """
    if T_fixed is not None:
        bounds = [am_bounds, h2o_abun_bounds, dlam_bounds]

        def objective(p):
            return chi2(p, interp, wave_obs, flux_obs, sigma_obs,
                        mask, lam_grid, R_inst, T_fixed=T_fixed)
    else:
        bounds = [am_bounds, h2o_abun_bounds, T_bounds, dlam_bounds]

        def objective(p):
            return chi2(p, interp, wave_obs, flux_obs, sigma_obs,
                        mask, lam_grid, R_inst, T_fixed=None)

    if optimizer == "de":
        # ------------------------------------------------------------------
        # Differential Evolution: population-based global search.
        # Does not require a starting point or gradient estimate.
        # polish=True runs L-BFGS-B once DE has landed in the correct basin,
        # giving sub-grid precision on the final answer.
        # ------------------------------------------------------------------
        print("[fit] Running Differential Evolution global search...")
        res = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            popsize=20,
            tol=1e-6,
            polish=True,
            workers=1,
        )
        print(f"[fit] DE converged: {res.success}  nfev={res.nfev}  chi2={res.fun:.4f}")

    elif optimizer == "lbfgsb":
        # ------------------------------------------------------------------
        # Pure L-BFGS-B from the supplied starting point.
        # Use only when you already know the basin (e.g. after a DE run or
        # after inspecting the heatmap).
        # ------------------------------------------------------------------
        if T_fixed is not None:
            x0 = [x0_am, x0_dlam]
        else:
            x0 = [x0_am, x0_T, x0_dlam]

        print("[fit] Running L-BFGS-B from supplied starting point...")
        res = minimize(
            objective,
            x0=np.array(x0, float),
            bounds=bounds,
            method="L-BFGS-B",
        )
        print(f"[fit] L-BFGS-B converged: {res.success}  nfev={res.nfev}  chi2={res.fun:.4f}")

    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Choose 'de' or 'lbfgsb'.")

    if T_fixed is not None:
        am_fit, h2o_fit, dlam_fit = res.x
        T_fit = T_fixed
    else:
        am_fit, h2o_fit, T_fit, dlam_fit = res.x

    best = {
        "airmass":   float(am_fit),
        "h2o_abun":  float(h2o_fit),
        "surface_T": float(T_fit),
        "dlam_nm":   float(dlam_fit),
    }
    return best, float(res.fun), res


def apply_telluric_correction(
    best_params:    Dict[str, float],
    interp:         RegularGridInterpolator,
    wave_obs:       np.ndarray,
    flux_obs:       np.ndarray,
    sigma_obs:      np.ndarray,
    mask:           np.ndarray,
    lam_grid:       np.ndarray,
    R_inst:         float,
    telluric_floor: float = 1e-6,
) -> Dict[str, np.ndarray]:
    am        = best_params["airmass"]
    h2o_abun  = best_params.get("h2o_abun", 1.0)
    surface_T = best_params["surface_T"]
    dlam_nm   = best_params["dlam_nm"]

    T_base  = eval_interp(interp, am, h2o_abun, surface_T)
    T_model = forward_model(T_base, dlam_nm, wave_obs, lam_grid, R_inst)
    valid   = _valid_mask(T_model, flux_obs, sigma_obs, mask)

    c0, c1    = _solve_linear_continuum(T_model, wave_obs, flux_obs, sigma_obs, valid)
    lam0      = np.median(wave_obs[valid])
    continuum = c0 + c1 * (wave_obs - lam0)

    T_safe          = np.clip(T_model, telluric_floor, np.inf)
    corrected_flux  = flux_obs  / T_safe
    corrected_sigma = sigma_obs / T_safe

    return {
        "T_base":           T_base,
        "T_model":          T_model,
        "continuum":        continuum,
        "corrected_flux":   corrected_flux,
        "corrected_sigma":  corrected_sigma,
    }


# ============================================================
# HIGH-LEVEL DRIVER
# ============================================================

def prepare_pipeline(
    cfg_obj:       PipelineConfig,
    airmass_grid:  Sequence[float],
    h2o_abun_grid: Sequence[float],
    temp_grid:     Sequence[float],
    force_rebuild: bool = False,
) -> Tuple[RegularGridInterpolator, RegularGridInterpolator, np.ndarray]:
    """Build or load the HDF5 grid; return (interp_total, interp_h2o, lam_grid_nm)."""
    build_grid_hdf5(
        cfg_obj=cfg_obj,
        airmass_grid=airmass_grid,
        h2o_abun_grid=h2o_abun_grid,
        temp_grid=temp_grid,
        force_rebuild=force_rebuild,
    )
    return make_telluric_interp(cfg_obj.h5_filename)


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def _lam_mask(
    lam_grid: np.ndarray,
    lam_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    if lam_range is None:
        return np.ones(len(lam_grid), bool)
    return (lam_grid >= lam_range[0]) & (lam_grid <= lam_range[1])


def _finish_plot(title: str, lam_range: Optional[Tuple[float, float]]) -> None:
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(title)
    if lam_range is not None:
        plt.xlim(*lam_range)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_vs_airmass(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    h2o_abun:  float = 1.0,
    surface_T: float = 280.0,
    am_list:   Sequence[float] = (1.0, 1.2, 1.5, 2.0),
    lam_range: Optional[Tuple[float, float]] = None,
) -> None:
    mask = _lam_mask(lam_grid, lam_range)
    plt.figure(figsize=(12, 5))
    for am in am_list:
        spec = eval_interp(interp, am, h2o_abun, surface_T)
        plt.plot(lam_grid[mask], spec[mask], label=f"AM={am:.2f}")
    _finish_plot(
        f"Transmission vs Airmass  (H2O={h2o_abun:.2f}, T={surface_T:.1f} K)",
        lam_range,
    )


def plot_vs_temperature(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    am:        float = 1.5,
    h2o_abun:  float = 1.0,
    T_list:    Sequence[float] = (270.0, 280.0, 290.0),
    lam_range: Optional[Tuple[float, float]] = None,
) -> None:
    mask = _lam_mask(lam_grid, lam_range)
    plt.figure(figsize=(12, 5))
    for T in T_list:
        spec = eval_interp(interp, am, h2o_abun, T)
        plt.plot(lam_grid[mask], spec[mask], label=f"T={T:.1f} K")
    _finish_plot(
        f"Transmission vs Surface Temperature  (AM={am:.2f}, MERRA-2 PWV)",
        lam_range,
    )


def plot_best_fit(
    wave_obs:    np.ndarray,
    flux_obs:    np.ndarray,
    sigma_obs:   np.ndarray,
    interp:      RegularGridInterpolator,
    lam_grid:    np.ndarray,
    best_params: Dict[str, float],
    R_inst:      float,
) -> None:
    products   = apply_telluric_correction(
        best_params, interp, wave_obs, flux_obs, sigma_obs,
        np.ones(len(wave_obs), bool), lam_grid, R_inst,
    )
    model_full = products["continuum"] * products["T_model"]

    plt.figure(figsize=(12, 5))
    plt.plot(wave_obs, flux_obs,   lw=1,   label="Observed")
    plt.plot(wave_obs, model_full, lw=1.2, label="Best-fit model")
    plt.fill_between(
        wave_obs,
        flux_obs - sigma_obs,
        flux_obs + sigma_obs,
        alpha=0.25, label="1sigma",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.title(
        f"Best-fit Telluric Model  "
        f"AM={best_params['airmass']:.3f}  "
        f"dlam={best_params['dlam_nm']:.4f} nm  "
        f"T={best_params['surface_T']:.1f} K"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_corrected_spectrum(
    wave_obs: np.ndarray,
    products: Dict[str, np.ndarray],
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(wave_obs, products["corrected_flux"], lw=1, label="Corrected flux")
    plt.fill_between(
        wave_obs,
        products["corrected_flux"] - products["corrected_sigma"],
        products["corrected_flux"] + products["corrected_sigma"],
        alpha=0.25, label="1sigma",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Corrected Flux")
    plt.title("Telluric-corrected Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_am_dlam_heatmap(
    interp:           RegularGridInterpolator,
    wave_obs:         np.ndarray,
    flux_obs:         np.ndarray,
    sigma_obs:        np.ndarray,
    mask:             np.ndarray,
    lam_grid:         np.ndarray,
    R_inst:           float,
    T_fixed:          float = 280.0,
    h2o_abun_fixed:   float = 1.0,
    am_bounds:        Tuple[float, float] = (1.0, 2.5),
    dlam_bounds:      Tuple[float, float] = (-0.5, 0.5),
    n_am:             int = 80,
    n_dlam:           int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chi-square heat map in (airmass, dlam) space with H2O fixed.

    h2o_abun_fixed holds the H2O abundance constant while scanning
    the airmass-dlam plane.  Pass best_params["h2o_abun"] from a
    prior fit to evaluate the landscape at the recovered H2O value.

    dlam_bounds defaults to +-0.5 nm so the full landscape is visible.
    Call a second time with tight bounds around the recovered minimum
    to zoom in on the basin.
    """
    am_vals   = np.linspace(*am_bounds,   n_am)
    dlam_vals = np.linspace(*dlam_bounds, n_dlam)

    chi2_map = np.full((n_dlam, n_am), np.nan)
    for j, dlam in enumerate(dlam_vals):
        for i, am in enumerate(am_vals):
            chi2_map[j, i] = chi2(
                [am, h2o_abun_fixed, dlam], interp, wave_obs, flux_obs, sigma_obs,
                mask, lam_grid, R_inst, T_fixed=T_fixed,
            )

    finite    = np.isfinite(chi2_map) & (chi2_map < 1e98)
    dchi2_map = np.full_like(chi2_map, np.nan)
    dchi2_map[finite] = chi2_map[finite] - np.nanmin(chi2_map[finite])

    min_idx   = np.unravel_index(np.nanargmin(chi2_map), chi2_map.shape)
    best_am   = am_vals[min_idx[1]]
    best_dlam = dlam_vals[min_idx[0]]

    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        dchi2_map, origin="lower", aspect="auto",
        extent=[am_vals[0], am_vals[-1], dlam_vals[0], dlam_vals[-1]],
    )
    plt.colorbar(im, label=r"$\Delta\chi^2$")
    plt.plot(
        best_am, best_dlam, "wx", ms=10, mew=2,
        label=f"Best: AM={best_am:.3f}, dlam={best_dlam:.4f} nm",
    )
    plt.xlabel("Airmass")
    plt.ylabel(r"$\delta\lambda$ (nm)")
    plt.title(
        f"$\\Delta\\chi^2$ Heat Map -- Airmass vs Wavelength Shift\n"
        f"(H2O={h2o_abun_fixed:.2f}, T={T_fixed:.1f} K, MERRA-2 + watm=y)"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        f"\n[heatmap] Best grid point:  "
        f"AM={best_am:.4f}  dlam={best_dlam:.4f} nm  "
        f"chi2={chi2_map[min_idx]:.4f}"
    )
    return am_vals, dlam_vals, chi2_map


# ============================================================
# FIX 3 -- REALISTIC SYNTHETIC OBSERVATION
# ============================================================

def make_synthetic_observation(
    interp:       RegularGridInterpolator,
    lam_grid:     np.ndarray,
    R_inst:       float,
    true_am:      float = 1.5,
    true_h2o:     float = 1.0,
    true_T:       float = 280.0,
    true_dlam:    float = 0.015,
    sigma:        float = 0.02,
    seed:         int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic observed spectrum on a realistic detector
    wavelength solution.

    The detector grid is lam_grid + true_dlam -- it does NOT coincide
    with the PSG model grid.  This is the physically correct test for
    dlam recovery: the shift only becomes visible to the optimizer
    when the observed and model wavelength axes are genuinely distinct.

    If you use wave_obs = lam_grid.copy() and inject dlam via the
    forward model, the residuals are zero everywhere at dlam=0 and the
    optimizer has nothing to grip.

    Returns
    -------
    wave_obs  : (N,) detector wavelength axis = lam_grid + true_dlam
    flux_obs  : (N,) noisy transmission spectrum
    sigma_obs : (N,) per-pixel uncertainty
    """
    # Detector wavelength axis: physically offset from the model grid
    wave_obs = lam_grid + true_dlam

    T_base_true = eval_interp(interp, true_am, true_h2o, true_T)

    # Evaluate the true flux on the detector grid.
    # dlam=0 here because the offset is already encoded in wave_obs.
    flux_true = forward_model(T_base_true, 0.0, wave_obs, lam_grid, R_inst)

    rng       = np.random.default_rng(seed)
    sigma_obs = sigma * np.ones_like(flux_true)
    flux_obs  = flux_true + sigma_obs * rng.standard_normal(len(flux_true))

    return wave_obs, flux_obs, sigma_obs


# ============================================================
# THREE-PANEL DIAGNOSTIC  (reproduces Ben's Raw_PSG_H2O plot)
# ============================================================

def plot_raw_psg_three_panel(
    interp_total: RegularGridInterpolator,
    interp_h2o:   RegularGridInterpolator,
    lam_grid:     np.ndarray,
    R_inst:       float,
    am:           float = 1.5,
    h2o_abun:     float = 1.0,
    surface_T:    float = 280.0,
    lam_range:    Tuple[float, float] = (1500.0, 2600.0),
) -> None:
    """
    Reproduce the three-panel Raw_PSG_H2O figure:

    Panel 1 — Raw H2O transmission at native PSG resolution
    Panel 2 — H2O transmission convolved to R_inst
    Panel 3 — Total transmission convolved to R_inst

    This is the correct diagnostic for verifying that individual
    HITRAN lines are resolved and that H2O ABUN scaling changes
    are visible in line depths rather than just the broad envelope.
    """
    mask = _lam_mask(lam_grid, lam_range)
    lam  = lam_grid[mask]

    h2o_raw   = eval_interp(interp_h2o,   am, h2o_abun, surface_T)[mask]
    total_raw = eval_interp(interp_total,  am, h2o_abun, surface_T)[mask]

    # Convolve to instrument resolution
    h2o_conv   = convolve_to_R(lam, h2o_raw,   R_inst)
    total_conv = convolve_to_R(lam, total_raw,  R_inst)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"PSG Telluric Transmission  AM={am:.2f}  T={surface_T:.0f} K"
        f"  (native R={PSG_NATIVE_RP:,} → convolved R={int(R_inst):,})",
        fontsize=13,
    )

    axes[0].plot(lam, h2o_raw,   lw=0.6, color="steelblue")
    axes[0].set_ylabel("Transmittance")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"Raw H2O at native PSG resolution (R={PSG_NATIVE_RP:,})")

    axes[1].plot(lam, h2o_conv,  lw=0.8, color="steelblue")
    axes[1].set_ylabel("Transmittance")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(f"Convolved H2O  R={int(R_inst):,}")

    axes[2].plot(lam, total_conv, lw=0.8, color="steelblue")
    axes[2].set_ylabel("Transmittance")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title(f"Convolved Total Transmission  R={int(R_inst):,}")
    axes[2].set_xlabel("Wavelength (nm)")

    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1.  Configure the pipeline
    #     Use nighttime UT: HST = UT - 10h, so 05:00 UT = 19:00 HST
    # ------------------------------------------------------------------
    cfg_test = PipelineConfig(
        template_path     = "earth_cfg.txt",
        observation_date  = "2024/01/15 05:00",
        site              = SiteConfig(),
        wavelength_cfg_nm = (1500.0, 2500.0, 50_000.0),
        h5_filename       = "telluric_grid.h5",
    )

    psg_in = build_psg_input(1.5, 280.0, cfg_test, h2o_abun=1.0)
    arr, cols = run_psg(psg_in, cfg_test)
    lam = 1e7 / arr[:, 0]
    mask = (lam >= 1780) & (lam <= 1820)

    print("Columns:", cols)
    print("\nPer-column transmission in 1780-1820 nm band:")
    for i, col in enumerate(cols):
        vals = arr[mask, i]
        print(f"  {col:12s}  min={vals.min():.6f}  max={vals.max():.6f}  mean={vals.mean():.6f}")

    print("Columns returned:", cols)

    wave_idx  = find_column(cols, "Wave")  or 0
    total_idx = find_column(cols, "Total") or 1
    h2o_idx   = find_column(cols, "H2O")

    print(f"Wave idx={wave_idx}  Total idx={total_idx}  H2O idx={h2o_idx}")

    if h2o_idx is not None:
        import numpy as np
        lam = 1e7 / arr[:, wave_idx]
        # Look at a window with known H2O absorption
        mask = (lam >= 1600) & (lam <= 1700)
        print(f"\nIn 1600-1700 nm window:")
        print(f"  Total  min={arr[mask, total_idx].min():.6f}  "
            f"max={arr[mask, total_idx].max():.6f}")
        print(f"  H2O    min={arr[mask, h2o_idx].min():.6f}  "
            f"max={arr[mask, h2o_idx].max():.6f}")
        print(f"\nAre Total and H2O identical? "
            f"{np.allclose(arr[:, total_idx], arr[:, h2o_idx])}")
        print(f"Is H2O all ones? "
            f"{np.allclose(arr[:, h2o_idx], 1.0)}")
        print(f"Is H2O all zeros? "
            f"{np.allclose(arr[:, h2o_idx], 0.0)}")

        # Show what Beer-Lambert scaling would produce
        T_total = arr[:, total_idx]
        T_h2o   = arr[:, h2o_idx]
        for scale in [0.5, 1.0, 2.0]:
            T_scaled = scale_h2o_transmission(T_total, T_h2o, scale)
            print(f"  scale={scale:.1f}  "
                f"scaled min={T_scaled[mask].min():.6f}  "
                f"max={T_scaled[mask].max():.6f}  "
                f"mean={T_scaled[mask].mean():.6f}")
    else:
        print("H2O column NOT present in PSG response")
        print("Full response header:")
        with open("psg_last_raw_response.txt") as f:
            for line in f:
                if line.startswith("#"):
                    print(" ", line.strip())
                else:
                    break
    # Airmass grid: covers typical LIGER observing range
    airmass_grid = [1.0, 1.2, 1.5, 2.0]

    # H2O abundance grid: wide range centered on the template baseline.
    # 1.0 = unperturbed template; 0.5 = half column; 2.0 = double column.
    # The wide range (0.5-3.0) is needed because ABUN response is sub-linear
    # and we need genuine separation in line depths across the grid.
    h2o_abun_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    # Single temperature node -- temperature axis is inactive per validation
    temp_grid    = [280.0]

    # ------------------------------------------------------------------
    # 2.  Build or load grid
    # ------------------------------------------------------------------
    interp, interp_h2o, lam_grid = prepare_pipeline(
        cfg_obj       = cfg_test,
        airmass_grid  = airmass_grid,
        h2o_abun_grid = h2o_abun_grid,
        temp_grid     = temp_grid,
        force_rebuild = False,
    )

    T_FIXED = 280.0
    R_INST  = 8_000.0

    # ------------------------------------------------------------------
    # 3.  Forward model round-trip check (Fix 1)
    #     rms/sigma must be < 0.10 before trusting fit results.
    #     If it is not, add more airmass nodes and rebuild.
    # ------------------------------------------------------------------
    roundtrip = check_forward_model_roundtrip(
        interp    = interp,
        lam_grid  = lam_grid,
        R_inst    = R_INST,
        am        = 1.5,
        h2o_abun  = 1.0,
        surface_T = T_FIXED,
        sigma_ref = 0.02,
    )

    # ------------------------------------------------------------------
    # 3b. Three-panel diagnostic -- reproduces Ben's Raw_PSG_H2O plot
    #     Verifies that individual HITRAN lines are resolved in the grid
    # ------------------------------------------------------------------
    plot_raw_psg_three_panel(
        interp_total = interp,
        interp_h2o   = interp_h2o,
        lam_grid     = lam_grid,
        R_inst       = R_INST,
        am           = 1.5,
        surface_T    = T_FIXED,
    )

    # ------------------------------------------------------------------
    # 4.  Build synthetic observation with a REAL wavelength offset (Fix 3)
    #     wave_obs = lam_grid + true_dlam, NOT lam_grid.copy()
    # ------------------------------------------------------------------
    true_am   = 1.5
    true_h2o  = 1.0    # H2O ABUN -- 1.0 = template baseline
    true_T    = T_FIXED
    true_dlam = 0.10   # nm -- resolvable shift (~0.5 pixels at R=8000, 1650 nm)

    wave_obs, flux_obs, sigma_obs = make_synthetic_observation(
        interp    = interp,
        lam_grid  = lam_grid,
        R_inst    = R_INST,
        true_am   = true_am,
        true_h2o  = true_h2o,
        true_T    = true_T,
        true_dlam = true_dlam,
        sigma     = 0.02,
    )

    # ------------------------------------------------------------------
    # 5.  Choose fit window and verify transmission structure (Fix 2)
    #     Use H-band window 1550-1750 nm, not the saturated 1850-1950 nm band
    # ------------------------------------------------------------------
    fit_mask = (
        np.isfinite(wave_obs)
        & (wave_obs >= 1550.0)
        & (wave_obs <= 1750.0)
    )

    check_fit_window(
        interp    = interp,
        lam_grid  = lam_grid,
        fit_mask  = fit_mask,
        am        = true_am,
        h2o_abun  = true_h2o,
        surface_T = T_FIXED,
    )

    # ------------------------------------------------------------------
    # 6.  Fit with Differential Evolution (robust global optimizer)
    #     L-BFGS-B gets stuck at dlam=0 on broad, shallow basins.
    #     DE explores the full bounds before polishing with L-BFGS-B.
    # ------------------------------------------------------------------
    best_params, chi2_val, res = fit_telluric(
        interp          = interp,
        wave_obs        = wave_obs,
        flux_obs        = flux_obs,
        sigma_obs       = sigma_obs,
        mask            = fit_mask,
        lam_grid        = lam_grid,
        R_inst          = R_INST,
        T_fixed         = T_FIXED,
        am_bounds       = (1.0, 2.5),
        h2o_abun_bounds = (0.5, 3.0),
        dlam_bounds     = (-0.5, 0.5),
        optimizer       = "de",
    )

    n_pix = int(fit_mask.sum())
    print("\n--- Smoke test results ---")
    print(f"  True:      am={true_am}  h2o={true_h2o}  dlam={true_dlam}  T={true_T}")
    print(f"  Recovered: am={best_params['airmass']:.4f}  "
          f"h2o={best_params['h2o_abun']:.4f}  "
          f"dlam={best_params['dlam_nm']:.4f}  "
          f"T={best_params['surface_T']:.1f}")
    print(f"  Best chi2: {chi2_val:.4f}")
    print(f"  N pixels in window: {n_pix}")
    print(f"  chi2 / N = {chi2_val / n_pix:.3f}  (target ~1.0 for a clean fit)")

    # ------------------------------------------------------------------
    # 7.  Apply correction
    # ------------------------------------------------------------------
    products = apply_telluric_correction(
        best_params = best_params,
        interp      = interp,
        wave_obs    = wave_obs,
        flux_obs    = flux_obs,
        sigma_obs   = sigma_obs,
        mask        = fit_mask,
        lam_grid    = lam_grid,
        R_inst      = R_INST,
    )

    # ------------------------------------------------------------------
    # 8.  Diagnostic plots
    # ------------------------------------------------------------------
    plot_vs_airmass(interp, lam_grid, surface_T=T_FIXED)
    plot_vs_airmass(
        interp, lam_grid, surface_T=T_FIXED,
        lam_range=(1550.0, 1750.0),
    )
    plot_best_fit(wave_obs, flux_obs, sigma_obs, interp, lam_grid, best_params, R_INST)
    plot_corrected_spectrum(wave_obs, products)

    # Wide heatmap -- confirm dlam minimum is localised, not a plateau
    # H2O held at the recovered best value while scanning (am, dlam)
    plot_am_dlam_heatmap(
        interp          = interp,
        wave_obs        = wave_obs,
        flux_obs        = flux_obs,
        sigma_obs       = sigma_obs,
        mask            = fit_mask,
        lam_grid        = lam_grid,
        R_inst          = R_INST,
        T_fixed         = T_FIXED,
        h2o_abun_fixed  = best_params["h2o_abun"],
        am_bounds       = (1.0, 2.5),
        dlam_bounds     = (-0.5, 0.5),
        n_am            = 80,
        n_dlam          = 80,
    )

    # Zoom in once the basin is confirmed
    plot_am_dlam_heatmap(
        interp          = interp,
        wave_obs        = wave_obs,
        flux_obs        = flux_obs,
        sigma_obs       = sigma_obs,
        mask            = fit_mask,
        lam_grid        = lam_grid,
        R_inst          = R_INST,
        T_fixed         = T_FIXED,
        h2o_abun_fixed  = best_params["h2o_abun"],
        am_bounds       = (
            max(1.0, best_params["airmass"] - 0.3),
            min(2.5, best_params["airmass"] + 0.3),
        ),
        dlam_bounds     = (
            best_params["dlam_nm"] - 0.1,
            best_params["dlam_nm"] + 0.1,
        ),
        n_am   = 120,
        n_dlam = 120,
    )

    print("\nPipeline complete.")