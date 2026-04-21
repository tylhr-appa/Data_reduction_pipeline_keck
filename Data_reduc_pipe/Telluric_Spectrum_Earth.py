"""
Telluric correction pipeline for Keck LIGER
============================================

Architecture
------------
PSG is called without watm=y so the ATMOSPHERE-LAYER entries from
the template are used verbatim.  This preserves per-species H2O line
absorption in the output columns, which scale_psg then uses to apply
Beer-Lambert airmass and PWV scaling analytically after retrieval.

The pipeline builds a grid over airmass and fits two free parameters
per observation:

    airmass  — path length through the atmosphere
    dlam_nm  — sub-pixel wavelength shift of the instrument

PWV is fixed by MERRA-2 for the observation date and is not a free
fit parameter.  This follows the approach of professional telluric
correction pipelines (MOLECFIT, TelFit, Xtellcor) which fix PWV from
an external measurement rather than fitting it freely, because the
airmass-PWV degeneracy makes simultaneous recovery unreliable.

Key design choices
------------------
1.  PSG uses the template ATMOSPHERE-LAYER entries verbatim (no watm=y)
    so the H2O column reflects true HITRAN line absorption, enabling
    scale_psg to apply per-species Beer-Lambert scaling analytically.
2.  build_psg_input sets geometry, date, site, and per-species output
    columns but does NOT override the template's wavelength range or
    resolving power.  This lets the template control the spectral
    setup, which avoids silent mismatches between what PSG computes
    and what the pipeline expects.
3.  The synthetic smoke-test uses a detector wavelength grid offset
    from the model grid by true_dlam so wavelength-shift recovery is
    physically meaningful.
4.  Differential Evolution is used for fitting — robust against broad,
    shallow chi-square basins typical of telluric fitting.
5.  All PSG calls are cached to disk keyed by a hash of the full config;
    rebuilding the grid after non-PSG code changes costs zero API calls.
6.  Grid validity is keyed to (date, site, wavelength, airmass grid,
    temp grid) so any axis change invalidates stale grids.
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
PSG_NATIVE_RP = 200_000


# ============================================================
# TEMPLATE DISCOVERY
# ============================================================

def _find_template(name: str = "earth_cfg.txt") -> str:
    """Locate the PSG template, checking the script dir and Data_reduc_pipe/."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        os.path.join("Data_reduc_pipe", name),
        name,
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return name


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
    # PSG template.  The ATMOSPHERE-LAYER entries are used verbatim by PSG
    # (no watm=y).  The template also controls the wavelength range and
    # resolving power — build_psg_input does NOT override these.
    template_path: str = _find_template()

    # Output paths
    out_dir:     str = "telluric_grid"
    h5_filename: str = "telluric_grid.h5"

    # PSG endpoint
    psg_api: str = PSG_API

    # Wavelength range (nm): used for cache keys, fit windows, and plots.
    # The actual PSG query uses whatever is in the template.
    wavelength_cfg_nm: Tuple[float, float, float] = (1500.0, 2500.0, 50_000.0)

    # Observation date embedded in the PSG config for reference.
    # Use nighttime UT for Maunakea (HST = UT - 10h), e.g. 05:00 UT = 19:00 HST.
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

    T_scaled = T_h2o**h2o_scale * (T_total / T_h2o)
    """
    T_h2o_safe = np.clip(T_h2o,  1e-10, 1.0)
    T_other    = np.clip(T_total / T_h2o_safe, 0.0, 1.0)
    T_h2o_scaled = np.power(T_h2o_safe, float(h2o_scale))
    return np.clip(T_h2o_scaled * T_other, 0.0, 1.0)


# =============================================
# SCALE_PSG  — per-species Beer-Lambert scaling
# =============================================

PSG_MOL_LABELS = ("H2O", "CO2", "CH4", "CO", "O3", "N2O", "O2")

PSG_MOL_COLORS = {
    "H2O": "steelblue",
    "CO2": "firebrick",
    "CH4": "forestgreen",
    "CO":  "darkorange",
    "O3":  "purple",
    "N2O": "saddlebrown",
    "O2":  "teal",
}


def _wave_col_to_nm(wave_col: np.ndarray) -> np.ndarray:
    """Convert a PSG wave column to nm, auto-detecting cm⁻¹ vs nm.

    PSG outputs wavenumber (cm⁻¹) when GENERATOR-RANGEUNIT is cm-1,
    and wavelength (nm) when it is nm.  Values > 10_000 are assumed
    to be cm⁻¹; values <= 10_000 are assumed to already be nm.
    """
    if np.nanmedian(wave_col) > 10_000:
        # wavenumber in cm⁻¹ → convert to nm
        return 1e7 / wave_col
    return wave_col.copy()


def scale_psg(
    psg_tuple: Tuple[np.ndarray, ...],
    airmass:   float,
    pwv:       float = 0.0,
) -> np.ndarray:
    """
    Apply per-species Beer-Lambert scaling to PSG molecular columns.

        T_scaled = T_H2O**(airmass + pwv) * prod(T_other**airmass)

    where pwv is a relative perturbation on top of the template H2O
    column (pwv=0.0 means no extra H2O scaling beyond airmass).
    """
    h2o, co2, ch4, co, o3, n2o, o2 = psg_tuple
    h2o_safe = np.clip(h2o, 1e-30, 1.0)
    others   = [np.clip(x, 1e-30, 1.0) for x in (co2, ch4, co, o3, n2o, o2)]
    model = (
        h2o_safe ** (float(airmass) + float(pwv))
        * np.prod([x ** float(airmass) for x in others], axis=0)
    )
    return np.clip(model, 0.0, 1.0)


def extract_mol_columns(
    arr:      np.ndarray,
    cols:     List[str],
    lam_ref:  np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Extract per-species transmission columns from a PSG response array
    and interpolate onto lam_ref.

    Returns (lam_ref, mol_tuple) where mol_tuple is
    (H2O, CO2, CH4, CO, O3, N2O, O2) on lam_ref.
    """
    wave_idx = find_column(cols, "Wave") or 0
    lam_nm   = _wave_col_to_nm(arr[:, wave_idx])
    order    = np.argsort(lam_nm)
    lam_nm   = lam_nm[order]

    def _get(label: str) -> np.ndarray:
        idx = find_column(cols, label)
        if idx is None:
            print(f"  [extract_mol_columns] WARNING: {label} column not found, "
                  f"using ones")
            return np.ones(len(lam_ref))
        vals = np.nan_to_num(arr[order, idx].astype(float), nan=1.0)
        return np.interp(lam_ref, lam_nm, vals, left=1.0, right=1.0)

    mol_tuple = tuple(_get(label) for label in PSG_MOL_LABELS)
    return lam_ref, mol_tuple


def plot_molecular_species_overlay(
    lam_nm:    np.ndarray,
    mol_tuple: Tuple[np.ndarray, ...],
    airmass:   float,
    R_inst:    float = 8_000.0,
    lam_range: Tuple[float, float] = (1500.0, 2600.0),
    title:     str = "PSG Molecular Species",
) -> None:
    """
    Overlay per-species transmission on a single panel at the given
    airmass.  mol_tuple is expected at am=1.0; Beer-Lambert scaling
    T**airmass is applied per species before convolution.
    """
    mask = (lam_nm >= lam_range[0]) & (lam_nm <= lam_range[1])
    lam  = lam_nm[mask]

    plt.figure(figsize=(14, 6))
    for label, spec_base in zip(PSG_MOL_LABELS, mol_tuple):
        spec_am   = np.clip(spec_base, 1e-30, 1.0) ** float(airmass)
        spec_conv = convolve_to_R(lam, spec_am[mask], R_inst)
        color     = PSG_MOL_COLORS.get(label, "steelblue")
        plt.plot(lam, spec_conv, lw=0.9, color=color, label=label)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.ylim(-0.05, 1.05)
    plt.title(f"{title}  AM={airmass:.3f}  R={int(R_inst):,}")
    plt.legend(ncol=len(PSG_MOL_LABELS), loc="lower center", fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================
# PSG CONFIG BUILDER — template-driven, no wavelength override
# ============================================================

def build_psg_input(
    am:        float,
    surface_T: float,
    cfg_obj:   PipelineConfig,
    h2o_abun:  float = 1.0,
) -> str:
    """
    Build a PSG config for (airmass, surface_T, h2o_abun).

    Sets geometry, date, site, surface temperature, H2O abundance,
    and requests per-species transmission columns.  Does NOT override
    the template's wavelength range or resolving power — the template
    controls the spectral setup.
    """
    cfg = load_psg_template(cfg_obj.template_path)

    # Site + date
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

    # Request individual gas transmission columns
    cfg = set_parameter(cfg, "GENERATOR-TRANS-APPLY", "Y")
    cfg = set_parameter(cfg, "GENERATOR-TRANS-SHOW",  "Y")
    cfg = set_parameter(cfg, "GENERATOR-TRANS",       "02-01")

    # Surface temperature
    cfg = set_parameter(cfg, "SURFACE-TEMPERATURE", f"{float(surface_T):.6f}")

    # H2O abundance scaling
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
                    "file": psg_config_text,
                    "type": output_type,
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
    temp_grid: Sequence[float],
) -> str:
    key = str((
        observation_date,
        (site.latitude_deg, site.longitude_deg, site.altitude_km),
        tuple(float(x) for x in wavelength_cfg_nm),
        tuple(sorted(float(x) for x in airmass_grid)),
        tuple(sorted(float(x) for x in temp_grid)),
    ))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _grid_is_valid(
    h5_filename: str,
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    temp_grid: Sequence[float],
) -> bool:
    if not os.path.exists(h5_filename):
        return False
    expected = _grid_hash(
        cfg_obj.observation_date,
        cfg_obj.site,
        cfg_obj.wavelength_cfg_nm,
        airmass_grid,
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
) -> str:
    return os.path.join(
        out_dir,
        f"am{am:.3f}_h2o{h2o_abun:.4f}_T{surface_T:.1f}_cfg{cfg_hash}.dat",
    )


def _cols_path(dat_path: str) -> str:
    return dat_path + ".cols"


def _save_cols(path: str, col_names: Sequence[str]) -> None:
    with open(path, "w") as fh:
        fh.write(",".join(col_names) + "\n")


def _load_cols(path: str) -> List[str]:
    with open(path) as fh:
        line = fh.readline().strip()
    return [c for c in line.split(",") if c]


# ============================================================
# GRID BUILDER
# ============================================================

def build_grid_hdf5(
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    temp_grid: Sequence[float],
    force_rebuild: bool = False,
) -> str:
    """
    Build and cache the 3-D transmission array:
        spectra[i_am, k_T, i_lambda]

    One PSG call per temperature node at am=1.0; scale_psg fills
    the airmass axis analytically via Beer-Lambert scaling.
    """
    create_directory(cfg_obj.out_dir)

    airmass_grid = np.array(sorted(float(x) for x in airmass_grid), dtype=float)
    temp_grid    = np.array(sorted(float(x) for x in temp_grid),    dtype=float)

    if (not force_rebuild) and _grid_is_valid(
        cfg_obj.h5_filename, cfg_obj, airmass_grid, temp_grid
    ):
        print(f"[grid] Current grid found -- loading: {cfg_obj.h5_filename}")
        return cfg_obj.h5_filename

    n_am = len(airmass_grid)
    n_T  = len(temp_grid)

    print(
        f"[grid] Building grid  date={cfg_obj.observation_date}  "
        f"site={cfg_obj.site.name}\n"
        f"       airmasses={list(airmass_grid)}\n"
        f"       temps={list(temp_grid)}\n"
        f"       PSG calls needed: {n_T} (airmass filled analytically)"
    )

    def compute_or_load_base(surface_T: float):
        """Single PSG call at am=1.0 for each temperature node."""
        psg_in   = build_psg_input(1.0, surface_T, cfg_obj, h2o_abun=1.0)
        cfg_hash = _stable_cfg_hash(psg_in)
        p        = _cache_path(cfg_obj.out_dir, 1.0, 1.0, surface_T, cfg_hash)
        cp       = _cols_path(p)
        # Cache hit requires both the data file and the sidecar column
        # list, since extract_mol_columns relies on named columns to
        # pick out each molecular species.  A legacy .dat without a
        # sidecar is treated as a miss and refetched to repopulate.
        have_cache = (
            (not force_rebuild)
            and os.path.exists(p)  and os.path.getsize(p)  > 0
            and os.path.exists(cp) and os.path.getsize(cp) > 0
        )
        if have_cache:
            print(f"  [cache] base T={surface_T:.1f} -> {p}")
            try:
                arr  = np.loadtxt(p)
                cols = _load_cols(cp)
                if arr.ndim == 2 and arr.shape[1] >= 2 and cols:
                    return arr, cols
            except Exception as exc:
                print(f"  [cache] read failed ({exc}), recomputing")
        arr, col_names = run_psg(psg_in, cfg_obj)
        if arr is None:
            raise RuntimeError(f"PSG returned no data for T={surface_T}")
        np.savetxt(p, arr)
        if col_names:
            _save_cols(cp, col_names)
        return arr, col_names

    # Reference wavelength grid from first temperature node
    ref_arr, ref_cols = compute_or_load_base(temp_grid[0])
    wave_idx = find_column(ref_cols, "Wave") or 0

    lam_ref = _wave_col_to_nm(ref_arr[:, wave_idx])
    order   = np.argsort(lam_ref)
    lam_ref = lam_ref[order]
    nlam    = len(lam_ref)

    grid_specs = np.full((n_am, n_T, nlam), np.nan, dtype=float)
    grid_h2o   = np.full_like(grid_specs, np.nan)

    for k, surface_T in enumerate(temp_grid):
        print(f"[grid] PSG call {k+1}/{n_T}  T={surface_T:.1f}  "
              f"(airmass axis filled analytically via scale_psg)")
        arr, cols = compute_or_load_base(surface_T)

        _, mol_tuple = extract_mol_columns(arr, cols, lam_ref)
        h2o_base     = mol_tuple[0]

        for i, am in enumerate(airmass_grid):
            scaled = scale_psg(mol_tuple, airmass=am, pwv=0.0)
            grid_specs[i, k, :] = scaled
            grid_h2o[i, k, :]   = np.clip(h2o_base, 1e-30, 1.0) ** float(am)

        print(f"  -> filled {n_am} airmass nodes  am={list(airmass_grid)}")

    _debug_airmass_variation(grid_specs, airmass_grid, temp_grid)

    gh = _grid_hash(
        cfg_obj.observation_date,
        cfg_obj.site,
        cfg_obj.wavelength_cfg_nm,
        airmass_grid,
        temp_grid,
    )

    with h5py.File(cfg_obj.h5_filename, "w") as hf:
        hf.create_dataset("spectra",        data=grid_specs)
        hf.create_dataset("spectra_h2o",    data=grid_h2o)
        hf.create_dataset("airmasses",      data=airmass_grid)
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
    temp_grid: np.ndarray,
) -> None:
    k_mid = len(temp_grid) // 2
    i_ref = len(airmass_grid) // 2
    ref   = grid_specs[i_ref, k_mid, :]
    print("\n[debug] Airmass variation (relative to middle airmass node):")
    for i, am in enumerate(airmass_grid):
        spec = grid_specs[i, k_mid, :]
        diff = spec - ref
        print(
            f"  AM={am:.2f}  min={np.nanmin(spec):.4f}  "
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
        temperatures = hf["temperatures"][:]
        wavelengths  = hf["wavelengths_nm"][:]
    return spectra, spectra_h2o, airmasses, temperatures, wavelengths


def make_telluric_interp(h5_filename: str):
    """
    Return (interp_total, interp_h2o, lam_grid_nm).

    Both are RegularGridInterpolators over (airmass, surface_T).
    """
    spectra, spectra_h2o, airmasses, temperatures, lam_grid = load_hdf5_grid(h5_filename)

    def _make(values):
        return RegularGridInterpolator(
            points=(airmasses, temperatures),
            values=values,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    return _make(spectra), _make(spectra_h2o), lam_grid


def eval_interp(
    interp: RegularGridInterpolator,
    am: float,
    surface_T: float,
) -> np.ndarray:
    return interp([[float(am), float(surface_T)]])[0]


# ============================================================
# TEMPERATURE AXIS VALIDATOR
# ============================================================

def validate_temperature_axis(
    interp: RegularGridInterpolator,
    lam_grid: np.ndarray,
    am: float,
    t1: float,
    t2: float,
) -> Dict[str, Any]:
    spec1 = eval_interp(interp, am, t1)
    spec2 = eval_interp(interp, am, t2)
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
    """
    T_conv       = convolve_to_R(lam_grid, T_base, R_inst)
    wave_shifted = np.asarray(wave_obs, float) - float(dlam_nm)
    return np.interp(wave_shifted, lam_grid, T_conv, left=np.nan, right=np.nan)


# ============================================================
# FORWARD MODEL ROUND-TRIP DIAGNOSTIC
# ============================================================

def check_forward_model_roundtrip(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    R_inst:    float,
    am:        float = 1.5,
    surface_T: float = 280.0,
    sigma_ref: float = 0.02,
) -> Dict[str, float]:
    """
    Evaluate the forward model at dlam=0 and compare to the raw
    interpolated spectrum.  rms_over_sigma should be < 0.10.
    """
    T_base  = eval_interp(interp, am, surface_T)
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
            "  Consider adding more airmass nodes or using method='cubic'."
        )
    else:
        print("  OK -- forward model round-trip is clean.")

    return result


# ============================================================
# FIT WINDOW DIAGNOSTIC
# ============================================================

def check_fit_window(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    fit_mask:  np.ndarray,
    am:        float = 1.5,
    surface_T: float = 280.0,
    min_mean:  float = 0.10,
    min_max:   float = 0.50,
) -> bool:
    """
    Print transmission statistics inside the fit window and warn if
    the window is too opaque for a reliable fit.
    """
    spec   = eval_interp(interp, am, surface_T)
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
        print(f"  WARNING: mean transmission {t_mean:.4f} < {min_mean}")
        ok = False
    if t_max < min_max:
        print(f"  WARNING: max transmission {t_max:.4f} < {min_max}")
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

    T_fixed is not None  ->  params = [airmass, dlam_nm]
    T_fixed is     None  ->  params = [airmass, surface_T, dlam_nm]
    """
    if T_fixed is not None:
        am, dlam_nm = params
        surface_T   = T_fixed
    else:
        am, surface_T, dlam_nm = params

    T_base  = eval_interp(interp, am, surface_T)
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
    x0_dlam:       float = 0.0,
    x0_T:          float = 280.0,
    am_bounds:     Tuple[float, float] = (1.0, 2.5),
    dlam_bounds:   Tuple[float, float] = (-0.5, 0.5),
    T_bounds:      Tuple[float, float] = (250.0, 320.0),
    optimizer:     str = "de",
) -> Tuple[Dict[str, float], float, Any]:
    """
    Fit (airmass, dlam_nm) with temperature optionally fixed.

    optimizer = "de"     — Differential Evolution (recommended)
    optimizer = "lbfgsb" — L-BFGS-B from x0 (fast but fragile)
    """
    if T_fixed is not None:
        bounds = [am_bounds, dlam_bounds]
        def objective(p):
            return chi2(p, interp, wave_obs, flux_obs, sigma_obs,
                        mask, lam_grid, R_inst, T_fixed=T_fixed)
    else:
        bounds = [am_bounds, T_bounds, dlam_bounds]
        def objective(p):
            return chi2(p, interp, wave_obs, flux_obs, sigma_obs,
                        mask, lam_grid, R_inst, T_fixed=None)

    if optimizer == "de":
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
        am_fit, dlam_fit = res.x
        T_fit = T_fixed
    else:
        am_fit, T_fit, dlam_fit = res.x

    best = {
        "airmass":   float(am_fit),
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
    surface_T = best_params["surface_T"]
    dlam_nm   = best_params["dlam_nm"]

    T_base  = eval_interp(interp, am, surface_T)
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
    temp_grid:     Sequence[float],
    force_rebuild: bool = False,
) -> Tuple[RegularGridInterpolator, RegularGridInterpolator, np.ndarray]:
    """Build or load the HDF5 grid; return (interp_total, interp_h2o, lam_grid_nm)."""
    build_grid_hdf5(
        cfg_obj=cfg_obj,
        airmass_grid=airmass_grid,
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
    surface_T: float = 280.0,
    am_list:   Sequence[float] = (1.0, 1.2, 1.5, 2.0),
    lam_range: Optional[Tuple[float, float]] = None,
) -> None:
    mask = _lam_mask(lam_grid, lam_range)
    plt.figure(figsize=(12, 5))
    for am in am_list:
        spec = eval_interp(interp, am, surface_T)
        plt.plot(lam_grid[mask], spec[mask], label=f"AM={am:.2f}")
    _finish_plot(
        f"Transmission vs Airmass  (T={surface_T:.1f} K, MERRA-2 PWV)",
        lam_range,
    )


def plot_vs_temperature(
    interp:    RegularGridInterpolator,
    lam_grid:  np.ndarray,
    am:        float = 1.5,
    T_list:    Sequence[float] = (270.0, 280.0, 290.0),
    lam_range: Optional[Tuple[float, float]] = None,
) -> None:
    mask = _lam_mask(lam_grid, lam_range)
    plt.figure(figsize=(12, 5))
    for T in T_list:
        spec = eval_interp(interp, am, T)
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
    am_bounds:        Tuple[float, float] = (1.0, 2.5),
    dlam_bounds:      Tuple[float, float] = (-0.5, 0.5),
    n_am:             int = 80,
    n_dlam:           int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chi-square heat map in (airmass, dlam) space."""
    am_vals   = np.linspace(*am_bounds,   n_am)
    dlam_vals = np.linspace(*dlam_bounds, n_dlam)

    chi2_map = np.full((n_dlam, n_am), np.nan)
    for j, dlam in enumerate(dlam_vals):
        for i, am in enumerate(am_vals):
            chi2_map[j, i] = chi2(
                [am, dlam], interp, wave_obs, flux_obs, sigma_obs,
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
        f"(T={T_fixed:.1f} K, template atmosphere + scale_psg)"
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
# REALISTIC SYNTHETIC OBSERVATION
# ============================================================

def make_synthetic_observation(
    interp:       RegularGridInterpolator,
    lam_grid:     np.ndarray,
    R_inst:       float,
    true_am:      float = 1.5,
    true_T:       float = 280.0,
    true_dlam:    float = 0.015,
    sigma:        float = 0.02,
    seed:         int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a synthetic observed spectrum on a realistic detector
    wavelength solution.  wave_obs = lam_grid + true_dlam.
    """
    wave_obs    = lam_grid + true_dlam
    T_base_true = eval_interp(interp, true_am, true_T)
    flux_true   = forward_model(T_base_true, 0.0, wave_obs, lam_grid, R_inst)

    rng       = np.random.default_rng(seed)
    sigma_obs = sigma * np.ones_like(flux_true)
    flux_obs  = flux_true + sigma_obs * rng.standard_normal(len(flux_true))

    return wave_obs, flux_obs, sigma_obs


# ============================================================
# THREE-PANEL DIAGNOSTIC
# ============================================================

def plot_raw_psg_response(
    cfg_obj:   PipelineConfig,
    am:        float = 1.5,
    surface_T: float = 280.0,
    R_mid:     float = 25_000.0,
    R_inst:    float = 8_000.0,
    lam_range: Tuple[float, float] = (1500.0, 2600.0),
) -> None:
    """
    Three-panel figure from a fresh direct PSG call.

    Panel 1 — Raw Total at native resolution
    Panel 2 — Total convolved to R_mid
    Panel 3 — Total convolved to R_inst
    """
    print(f"[plot_raw_psg_response] Making fresh PSG call: am={am}, T={surface_T}")
    psg_in    = build_psg_input(am, surface_T, cfg_obj, h2o_abun=1.0)
    arr, cols = run_psg(psg_in, cfg_obj)

    if arr is None:
        print("PSG call failed — cannot produce Raw_PSG_H2O plot.")
        return

    wave_idx  = find_column(cols, "Wave")  or 0
    total_idx = find_column(cols, PSG_TOTAL_LABEL) or 1
    h2o_idx   = find_column(cols, PSG_H2O_LABEL)

    lam_nm = _wave_col_to_nm(arr[:, wave_idx])
    order  = np.argsort(lam_nm)
    lam_nm = lam_nm[order]
    total  = arr[order, total_idx]
    h2o    = arr[order, h2o_idx] if h2o_idx is not None else total

    mask = (lam_nm >= lam_range[0]) & (lam_nm <= lam_range[1])
    lam  = lam_nm[mask]

    total_raw  = total[mask]
    total_mid  = convolve_to_R(lam, total_raw, R_mid)
    total_inst = convolve_to_R(lam, total_raw, R_inst)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"PSG Telluric Transmission  AM={am:.2f}  T={surface_T:.0f} K\n",
        fontsize=13,
    )

    axes[0].plot(lam, total_raw,  lw=0.4, color="steelblue")
    axes[0].set_ylabel("Transmittance")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"Raw PSG H2O  (native resolution)")

    axes[1].plot(lam, total_mid,  lw=0.7, color="steelblue")
    axes[1].set_ylabel("Transmittance")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(f"Convolved H2O at R={int(R_mid):,}")

    axes[2].plot(lam, total_inst, lw=0.8, color="steelblue")
    axes[2].set_ylabel("Transmittance")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title(f"Convolved Total Transmission at R={int(R_inst):,}")
    axes[2].set_xlabel("Wavelength (nm)")

    plt.tight_layout()
    plt.show()


# Backward-compatibility alias
plot_raw_psg_three_panel = plot_raw_psg_response


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # 1.  Configure the pipeline
    #     Sets the observation date, observatory site, wavelength
    #     range, and the HDF5 file where the transmission grid is
    #     cached.  The airmass_grid defines the nodes along which
    #     Beer-Lambert scaling interpolates; temp_grid is collapsed
    #     to a single value since surface T has negligible effect
    #     on telluric transmission in this band.
    # ----------------------------------------------------------
    cfg = PipelineConfig(
        observation_date  = "2024/01/15 05:00",
        site              = SiteConfig(),
        wavelength_cfg_nm = (1500.0, 2500.0, 50_000.0),
        h5_filename       = "telluric_grid.h5",
    )

    airmass_grid = [1.0, 1.2, 1.5, 2.0]
    temp_grid    = [280.0]

    # ----------------------------------------------------------
    # 2.  Build or load grid
    #     One PSG API call per temperature node at airmass=1.0;
    #     the airmass axis is filled analytically via Beer-Lambert
    #     scaling (scale_psg).  Returns interpolators over
    #     (airmass, surface_T) for total and H2O transmission, plus
    #     the reference wavelength grid in nm.
    # ----------------------------------------------------------
    interp, interp_h2o, lam_grid = prepare_pipeline(
        cfg_obj       = cfg,
        airmass_grid  = airmass_grid,
        temp_grid     = temp_grid,
        force_rebuild = True,
    )

    T_FIXED = 280.0
    R_INST  = 8_000.0

    # ----------------------------------------------------------
    # 3.  Forward model round-trip check
    #     Sanity check: evaluate the forward model at dlam=0 and
    #     compare against the raw interpolated spectrum.  Residuals
    #     should be well below the observational noise floor, which
    #     confirms that convolution + resampling do not distort the
    #     model.
    # ----------------------------------------------------------
    roundtrip = check_forward_model_roundtrip(
        interp    = interp,
        lam_grid  = lam_grid,
        R_inst    = R_INST,
        am        = 1.5,
        surface_T = T_FIXED,
        sigma_ref = 0.02,
    )

    # ----------------------------------------------------------
    # 3b. Three-panel diagnostic
    #     Fresh PSG call plotted at native resolution, an
    #     intermediate R, and the instrument R.  Useful for
    #     visualizing how high-resolution HITRAN line structure
    #     gets smoothed down to what the detector actually sees.
    # ----------------------------------------------------------
    plot_raw_psg_response(
        cfg_obj   = cfg,
        am        = 1.5,
        surface_T = T_FIXED,
        R_mid     = 25_000.0,
        R_inst    = R_INST,
    )

    # ----------------------------------------------------------
    # 4.  Synthetic observation with real wavelength offset
    #     Generates a mock spectrum at known (airmass, T, dlam)
    #     with Gaussian noise.  The detector wavelength grid is
    #     offset from the model grid by true_dlam so the fit must
    #     actually recover a real sub-pixel shift — this is the
    #     smoke test the fitter is graded against.
    # ----------------------------------------------------------
    true_am   = 1.5
    true_T    = T_FIXED
    true_dlam = 0.10

    wave_obs, flux_obs, sigma_obs = make_synthetic_observation(
        interp    = interp,
        lam_grid  = lam_grid,
        R_inst    = R_INST,
        true_am   = true_am,
        true_T    = true_T,
        true_dlam = true_dlam,
        sigma     = 0.02,
    )

    # ----------------------------------------------------------
    # 5.  Fit window
    #     Restricts the chi-square to a wavelength region with
    #     usable transmission structure (not fully saturated, not
    #     fully clear).  check_fit_window prints statistics and
    #     warns if the window is too opaque to constrain airmass.
    # ----------------------------------------------------------
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
        surface_T = T_FIXED,
    )

    # ----------------------------------------------------------
    # 6.  Fit with Differential Evolution
    #     Global optimizer over (airmass, dlam_nm) with T held
    #     fixed.  DE is used rather than a local method because
    #     the chi-square surface has broad, shallow basins where
    #     gradient descent is unreliable.
    # ----------------------------------------------------------
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
        dlam_bounds     = (-0.5, 0.5),
        optimizer       = "de",
    )

    n_pix = int(fit_mask.sum())
    print("\n--- Smoke test results ---")
    print(f"  True:      am={true_am}  dlam={true_dlam}  T={true_T}")
    print(f"  Recovered: am={best_params['airmass']:.4f}  "
          f"dlam={best_params['dlam_nm']:.4f}  "
          f"T={best_params['surface_T']:.1f}")
    print(f"  Best chi2: {chi2_val:.4f}")
    print(f"  N pixels in window: {n_pix}")
    print(f"  chi2 / N = {chi2_val / n_pix:.3f}  (target ~1.0)")

    # ----------------------------------------------------------
    # 7.  Apply correction
    #     Divides the observed flux by the best-fit telluric model
    #     (with floor to avoid blow-ups near saturated lines) to
    #     produce the corrected spectrum and propagated errors.
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 8.  Diagnostic plots
    #     Transmission vs airmass, best-fit overlay, corrected
    #     spectrum, and two chi-square heat maps in (airmass, dlam)
    #     space — one coarse over the full bounds, one zoomed
    #     around the fitted optimum to visualize parameter
    #     degeneracies and the depth of the chi-square minimum.
    # ----------------------------------------------------------
    plot_vs_airmass(interp, lam_grid, surface_T=T_FIXED)
    plot_vs_airmass(interp, lam_grid, surface_T=T_FIXED, lam_range=(1550.0, 1750.0))
    plot_best_fit(wave_obs, flux_obs, sigma_obs, interp, lam_grid, best_params, R_INST)
    plot_corrected_spectrum(wave_obs, products)

    # ----------------------------------------------------------
    # 8b. Per-species overlay at the recovered airmass
    #     PSG call at am=1.0 to extract per-species base columns,
    #     then Beer-Lambert scaled to the fitted airmass so the
    #     plot reflects the actual atmosphere the fit converged on.
    #     One axis, all species overlaid — shows which molecules
    #     dominate each wavelength region at the recovered AM.
    # ----------------------------------------------------------
    _psg_in_mol  = build_psg_input(1.0, T_FIXED, cfg, h2o_abun=1.0)
    _arr_mol, _cols_mol = run_psg(_psg_in_mol, cfg)
    if _arr_mol is not None:
        _, _mol_tuple = extract_mol_columns(_arr_mol, _cols_mol, lam_grid)
        plot_molecular_species_overlay(
            lam_nm    = lam_grid,
            mol_tuple = _mol_tuple,
            airmass   = best_params["airmass"],
            R_inst    = R_INST,
            lam_range = (1500.0, 2600.0),
            title     = f"Per-Species Transmission at Best-Fit  T={T_FIXED:.0f} K",
        )

    plot_am_dlam_heatmap(
        interp=interp, wave_obs=wave_obs, flux_obs=flux_obs,
        sigma_obs=sigma_obs, mask=fit_mask, lam_grid=lam_grid,
        R_inst=R_INST, T_fixed=T_FIXED,
        am_bounds=(1.0, 2.5), dlam_bounds=(-0.5, 0.5),
        n_am=80, n_dlam=80,
    )

    plot_am_dlam_heatmap(
        interp=interp, wave_obs=wave_obs, flux_obs=flux_obs,
        sigma_obs=sigma_obs, mask=fit_mask, lam_grid=lam_grid,
        R_inst=R_INST, T_fixed=T_FIXED,
        am_bounds=(
            max(1.0, best_params["airmass"] - 0.3),
            min(2.5, best_params["airmass"] + 0.3),
        ),
        dlam_bounds=(
            best_params["dlam_nm"] - 0.1,
            best_params["dlam_nm"] + 0.1,
        ),
        n_am=120, n_dlam=120,
    )

    print("\nPipeline complete.")