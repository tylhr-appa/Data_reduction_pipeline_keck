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
2.  build_psg_input keeps the template's wavelength range by default
    but accepts optional lam_min/lam_max overrides for sub-band
    stitching.  The resolving power is left to the template.
3.  For high resolving powers (R > ~100k) over wide wavelength ranges,
    the PSG call is split into sub-bands of configurable width with
    overlap, stitched back into a single spectrum, and cached per
    sub-band.  Everything above the PSG call layer (grid, HDF5,
    interpolators, fitting) sees one continuous spectrum.
4.  The synthetic smoke-test uses a detector wavelength grid offset
    from the model grid by true_dlam so wavelength-shift recovery is
    physically meaningful.
5.  Differential Evolution is used for fitting — robust against broad,
    shallow chi-square basins typical of telluric fitting.
6.  All PSG calls are cached to disk keyed by a hash of the full config;
    rebuilding the grid after non-PSG code changes costs zero API calls.
7.  Grid validity is keyed to (date, site, wavelength, airmass grid,
    temp grid) so any axis change invalidates stale grids.
8.  The three-panel diagnostic plot reproduces the Raw_PSG_H2O figure
    showing Total transmission at native, intermediate, and instrument
    resolving power.
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

MAUNAKEA_LAT = 19.8228
MAUNAKEA_LON = -155.4681
MAUNAKEA_ALT = 4.205

PSG_TOTAL_LABEL  = "Total"
PSG_H2O_LABEL    = "H2O"
PSG_NATIVE_RP    = 200_000


# ============================================================
# TEMPLATE DISCOVERY
# ============================================================

def _find_template(name: str = "earth_cfg.txt") -> str:
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
    latitude_deg:  float = MAUNAKEA_LAT
    longitude_deg: float = MAUNAKEA_LON
    altitude_km:   float = MAUNAKEA_ALT
    name:          str   = "Maunakea"


@dataclass(frozen=True)
class PipelineConfig:
    template_path: str = _find_template()

    out_dir:     str = "telluric_grid"
    h5_filename: str = "telluric_grid.h5"

    psg_api: str = PSG_API

    # Wavelength range (nm) for the full spectrum.
    # 0.96 - 2.5 microns, the full HISPEC range.
    wavelength_range_nm: Tuple[float, float] = (960.0, 2500.0)

    # Sub-band stitching.  At R=200k over 1000 nm PSG returns ~100k pts
    # per call; wider ranges or higher R need splitting.
    sub_band_width_nm:   float = 500.0
    sub_band_overlap_nm: float = 10.0

    observation_date: str = "2024/01/15 05:00"

    site: SiteConfig = field(default_factory=SiteConfig)

    psg_timeout_s: int = 180
    psg_max_tries: int = 10

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
        raise RuntimeError("PSG template appears invalid: no ATMOSPHERE-LAYER entries.")
    return cfg


def set_parameter(cfg: str, tag: str, value: Any) -> str:
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
    m = re.search(rf"^<{re.escape(tag)}>\s*(.*)$", cfg, flags=re.M)
    return m.group(1).strip() if m else None


def set_h2o_abun(cfg: str, h2o_scale: float) -> str:
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
# BEER-LAMBERT H2O SCALING
# ============================================================

def scale_h2o_transmission(
    T_total: np.ndarray, T_h2o: np.ndarray, h2o_scale: float,
) -> np.ndarray:
    T_h2o_safe   = np.clip(T_h2o,  1e-10, 1.0)
    T_other      = np.clip(T_total / T_h2o_safe, 0.0, 1.0)
    T_h2o_scaled = np.power(T_h2o_safe, float(h2o_scale))
    return np.clip(T_h2o_scaled * T_other, 0.0, 1.0)


# ============================================================
# SCALE_PSG — per-species Beer-Lambert scaling
# ============================================================

PSG_MOL_LABELS = ("H2O", "CO2", "CH4", "CO", "O3", "N2O", "O2")

PSG_MOL_COLORS = {
    "H2O": "steelblue",  "CO2": "firebrick",  "CH4": "forestgreen",
    "CO":  "darkorange",  "O3": "purple",      "N2O": "saddlebrown",
    "O2":  "teal",
}


def _wave_col_to_nm(wave_col: np.ndarray) -> np.ndarray:
    """Convert PSG wave column to nm, auto-detecting cm-1 vs nm."""
    if np.nanmedian(wave_col) > 10_000:
        return 1e7 / wave_col
    return wave_col.copy()


def scale_psg(
    psg_tuple: Tuple[np.ndarray, ...],
    airmass:   float,
    pwv:       float = 0.0,
) -> np.ndarray:
    h2o, co2, ch4, co, o3, n2o, o2 = psg_tuple
    h2o_safe = np.clip(h2o, 1e-30, 1.0)
    others   = [np.clip(x, 1e-30, 1.0) for x in (co2, ch4, co, o3, n2o, o2)]
    model = (
        h2o_safe ** (float(airmass) + float(pwv))
        * np.prod([x ** float(airmass) for x in others], axis=0)
    )
    return np.clip(model, 0.0, 1.0)


def extract_mol_columns(
    arr: np.ndarray, cols: List[str], lam_ref: np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    wave_idx = find_column(cols, "Wave") or 0
    lam_nm   = _wave_col_to_nm(arr[:, wave_idx])
    order    = np.argsort(lam_nm)
    lam_nm   = lam_nm[order]

    def _get(label: str) -> np.ndarray:
        idx = find_column(cols, label)
        if idx is None:
            print(f"  [extract_mol_columns] WARNING: {label} not found, using ones")
            return np.ones(len(lam_ref))
        vals = np.nan_to_num(arr[order, idx].astype(float), nan=1.0)
        return np.interp(lam_ref, lam_nm, vals, left=1.0, right=1.0)

    mol_tuple = tuple(_get(label) for label in PSG_MOL_LABELS)
    return lam_ref, mol_tuple


def plot_molecular_species(
    lam_nm: np.ndarray, mol_tuple: Tuple[np.ndarray, ...],
    R_inst: float = 8_000.0,
    lam_range: Tuple[float, float] = (1500.0, 2600.0),
    title: str = "PSG Molecular Species",
) -> None:
    mask = (lam_nm >= lam_range[0]) & (lam_nm <= lam_range[1])
    lam  = lam_nm[mask]
    fig, axes = plt.subplots(
        len(PSG_MOL_LABELS), 1,
        figsize=(14, 2.8 * len(PSG_MOL_LABELS)), sharex=True,
    )
    fig.suptitle(f"{title}  (convolved R={int(R_inst):,})", fontsize=13)
    for ax, label, spec in zip(axes, PSG_MOL_LABELS, mol_tuple):
        spec_conv = convolve_to_R(lam, spec[mask], R_inst)
        ax.plot(lam, spec_conv, lw=0.8, color=PSG_MOL_COLORS.get(label, "steelblue"))
        ax.set_ylabel("Trans.")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Wavelength (nm)")
    plt.tight_layout()
    plt.show()


# ============================================================
# PSG CONFIG BUILDER
# ============================================================

def build_psg_input(
    am:          float,
    surface_T:   float,
    cfg_obj:     PipelineConfig,
    h2o_abun:    float = 1.0,
    lam_min_nm:  Optional[float] = None,
    lam_max_nm:  Optional[float] = None,
) -> str:
    """
    Build a PSG config.  If lam_min_nm/lam_max_nm are given, override
    the template's wavelength range for sub-band stitching.
    """
    cfg = load_psg_template(cfg_obj.template_path)

    cfg = set_parameter(cfg, "OBJECT-DATE", cfg_obj.observation_date)
    cfg = set_parameter(
        cfg, "OBJECT-GEODETIC",
        f"{cfg_obj.site.latitude_deg:.4f},"
        f"{cfg_obj.site.longitude_deg:.4f},"
        f"{cfg_obj.site.altitude_km:.4f}",
    )

    zen = am_to_zenith_deg(am)
    cfg = set_parameter(cfg, "GEOMETRY",            "Lookingup")
    cfg = set_parameter(cfg, "GEOMETRY-USER-PARAM", f"{zen:.6f}")
    cfg = set_parameter(cfg, "GEOMETRY-OBS-ANGLE",  f"{zen:.6f}")

    # Sub-band wavelength override
    if lam_min_nm is not None and lam_max_nm is not None:
        range_unit = (get_parameter(cfg, "GENERATOR-RANGEUNIT") or "nm").strip()
        if range_unit.lower() == "cm-1":
            cfg = set_parameter(cfg, "GENERATOR-RANGE1", f"{1e7 / lam_max_nm:.6f}")
            cfg = set_parameter(cfg, "GENERATOR-RANGE2", f"{1e7 / lam_min_nm:.6f}")
        else:
            cfg = set_parameter(cfg, "GENERATOR-RANGE1", f"{lam_min_nm:.6f}")
            cfg = set_parameter(cfg, "GENERATOR-RANGE2", f"{lam_max_nm:.6f}")

    cfg = set_parameter(cfg, "GENERATOR-TRANS-APPLY", "Y")
    cfg = set_parameter(cfg, "GENERATOR-TRANS-SHOW",  "Y")
    cfg = set_parameter(cfg, "GENERATOR-TRANS",       "02-01")

    cfg = set_parameter(cfg, "SURFACE-TEMPERATURE", f"{float(surface_T):.6f}")
    cfg = set_h2o_abun(cfg, h2o_abun)

    band_str = ""
    if lam_min_nm is not None and lam_max_nm is not None:
        band_str = f"  band={lam_min_nm:.0f}-{lam_max_nm:.0f} nm"
    print(
        f"[PSG INPUT] am={am:.4f}  T={surface_T:.2f} K  "
        f"h2o_abun={h2o_abun:.4f}  "
        f"date={cfg_obj.observation_date}  site={cfg_obj.site.name}"
        f"{band_str}"
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
                data={"file": psg_config_text, "type": output_type},
                timeout=cfg_obj.psg_timeout_s,
            )
            print("HTTP status:", resp.status_code)

            if resp.status_code != 200:
                if cfg_obj.debug_dump:
                    with open("psg_http_error.txt", "w") as fh:
                        fh.write(resp.text)
                time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)
                continue

            text = resp.text.strip()
            if cfg_obj.debug_dump:
                with open("psg_last_raw_response.txt", "w") as fh:
                    fh.write(text)

            busy_phrases = [
                "other api call is still running", "please let it finish",
                "please wait", "busy", "wait 10 minutes",
            ]
            if any(b in text.lower() for b in busy_phrases):
                print("PSG busy -- waiting...")
                time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)
                continue

            if len(text) < 100:
                print("PSG response suspiciously short.")
                time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)
                continue

            lines      = text.splitlines()
            col_names  = parse_psg_header(lines)
            data_lines = [ln for ln in lines
                          if not ln.strip().startswith("#") and ln.strip()]
            if not data_lines:
                print("No numeric data lines.")
                time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)
                continue

            arr = np.genfromtxt(io.StringIO("\n".join(data_lines)), dtype=float)
            if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
                print("Unexpected array shape:", getattr(arr, "shape", None))
                time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)
                continue
            if not np.isfinite(arr).any():
                print("All PSG values non-finite.")
                time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)
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
            time.sleep(wait_s); wait_s = min(wait_s * 1.5, 120)

    print("PSG failed after all retries.")
    return None, []


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_path(out_dir: str, am: float, h2o_abun: float,
                surface_T: float, cfg_hash: str) -> str:
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


def _load_or_call_psg(
    psg_config_text: str,
    cfg_obj:         PipelineConfig,
    dat_path:        str,
    cols_path:       str,
    force_rebuild:   bool,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """Load cached PSG response or call PSG and cache the result."""
    have_cache = (
        (not force_rebuild)
        and os.path.exists(dat_path)  and os.path.getsize(dat_path)  > 0
        and os.path.exists(cols_path) and os.path.getsize(cols_path) > 0
    )
    if have_cache:
        print(f"  [cache] loading {dat_path}")
        try:
            arr  = np.loadtxt(dat_path)
            cols = _load_cols(cols_path)
            if arr.ndim == 2 and arr.shape[1] >= 2 and cols:
                return arr, cols
        except Exception as exc:
            print(f"  [cache] read failed ({exc}), recomputing")

    arr, col_names = run_psg(psg_config_text, cfg_obj)
    if arr is not None:
        np.savetxt(dat_path, arr)
        if col_names:
            _save_cols(cols_path, col_names)
    return arr, col_names


# ============================================================
# SUB-BAND STITCHING
# ============================================================

def compute_sub_bands(
    lam_min_nm: float,
    lam_max_nm: float,
    width_nm:   float,
    overlap_nm: float,
) -> List[Tuple[float, float]]:
    """
    Divide [lam_min_nm, lam_max_nm] into sub-bands with overlap.

    Returns list of (band_start, band_end) in nm.  Adjacent bands
    share overlap_nm of spectral coverage; the stitcher trims half
    the overlap from each side.
    """
    if width_nm <= 0:
        raise ValueError(f"sub_band_width_nm must be positive, got {width_nm}")
    if overlap_nm < 0:
        raise ValueError(f"sub_band_overlap_nm must be >= 0, got {overlap_nm}")

    step = width_nm - overlap_nm
    if step <= 0:
        raise ValueError(
            f"overlap ({overlap_nm}) must be less than width ({width_nm})"
        )

    bands = []
    start = lam_min_nm
    while start < lam_max_nm:
        end = min(start + width_nm, lam_max_nm)
        bands.append((start, end))
        start += step

    # Ensure the last band reaches lam_max_nm
    if bands and bands[-1][1] < lam_max_nm:
        bands[-1] = (bands[-1][0], lam_max_nm)

    return bands


def stitch_psg_responses(
    sub_results: List[Tuple[np.ndarray, List[str]]],
    sub_bands:   List[Tuple[float, float]],
    overlap_nm:  float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Stitch sub-band PSG responses into a single array.

    For each overlap region between adjacent bands, the midpoint is
    used as the cut: left band keeps the left half, right band keeps
    the right half.  This avoids convolution edge artifacts.
    """
    if len(sub_results) == 1:
        return sub_results[0]

    col_names = sub_results[0][1]
    wave_idx  = find_column(col_names, "Wave") or 0
    half_overlap = overlap_nm / 2.0

    trimmed_parts = []
    for i, ((arr, _), (band_start, band_end)) in enumerate(
        zip(sub_results, sub_bands)
    ):
        lam_nm = _wave_col_to_nm(arr[:, wave_idx])

        # First band: no left trim.  Last band: no right trim.
        trim_min = band_start if i == 0 else band_start + half_overlap
        trim_max = band_end   if i == len(sub_results) - 1 else band_end - half_overlap

        mask = (lam_nm >= trim_min) & (lam_nm <= trim_max)
        trimmed_parts.append(arr[mask])

    stitched = np.vstack(trimmed_parts)

    # Sort by wavelength (ascending nm)
    lam_all  = _wave_col_to_nm(stitched[:, wave_idx])
    order    = np.argsort(lam_all)
    stitched = stitched[order]

    lam_sorted = _wave_col_to_nm(stitched[:, wave_idx])
    print(
        f"[stitch] Combined {len(sub_results)} sub-bands -> "
        f"{stitched.shape[0]} pts  "
        f"range={lam_sorted.min():.1f}-{lam_sorted.max():.1f} nm"
    )
    return stitched, col_names


def fetch_stitched_spectrum(
    am:            float,
    surface_T:     float,
    cfg_obj:       PipelineConfig,
    h2o_abun:      float = 1.0,
    force_rebuild: bool = False,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Fetch a full-range PSG spectrum, splitting into sub-bands if the
    wavelength range exceeds sub_band_width_nm.

    Each sub-band is independently cached.  After all sub-bands are
    fetched they are stitched into a single array.  Everything above
    this function (grid builder, HDF5, fitting) sees one continuous
    spectrum.
    """
    lam_min, lam_max = cfg_obj.wavelength_range_nm
    width   = cfg_obj.sub_band_width_nm
    overlap = cfg_obj.sub_band_overlap_nm
    total_range = lam_max - lam_min

    create_directory(cfg_obj.out_dir)

    # Single-band: no splitting needed
    if width <= 0 or total_range <= width:
        print(f"[fetch] Single-band mode: {lam_min:.0f}-{lam_max:.0f} nm")
        psg_in   = build_psg_input(am, surface_T, cfg_obj, h2o_abun)
        cfg_hash = _stable_cfg_hash(psg_in)
        p  = _cache_path(cfg_obj.out_dir, am, h2o_abun, surface_T, cfg_hash)
        cp = _cols_path(p)
        return _load_or_call_psg(psg_in, cfg_obj, p, cp, force_rebuild)

    # Multi-band stitching
    sub_bands = compute_sub_bands(lam_min, lam_max, width, overlap)
    print(
        f"[fetch] Splitting {lam_min:.0f}-{lam_max:.0f} nm into "
        f"{len(sub_bands)} sub-bands of ~{width:.0f} nm "
        f"with {overlap:.0f} nm overlap"
    )
    for i, (b0, b1) in enumerate(sub_bands):
        print(f"  band {i+1}: {b0:.1f} - {b1:.1f} nm")

    sub_results = []
    for i, (b_start, b_end) in enumerate(sub_bands):
        print(f"\n[fetch] Sub-band {i+1}/{len(sub_bands)}: "
              f"{b_start:.1f}-{b_end:.1f} nm")

        psg_in = build_psg_input(
            am, surface_T, cfg_obj, h2o_abun,
            lam_min_nm=b_start, lam_max_nm=b_end,
        )
        cfg_hash = _stable_cfg_hash(psg_in)
        p  = _cache_path(cfg_obj.out_dir, am, h2o_abun, surface_T, cfg_hash)
        cp = _cols_path(p)

        arr, col_names = _load_or_call_psg(psg_in, cfg_obj, p, cp, force_rebuild)
        if arr is None:
            print(f"  Sub-band {i+1} failed — aborting stitch")
            return None, []
        sub_results.append((arr, col_names))

    return stitch_psg_responses(sub_results, sub_bands, overlap)


# ============================================================
# AVERAGE PSG MODEL — FITS EXPORT AND LOADER
# ============================================================

# Column names written to and read from the average PSG FITS file.
# Order matches the KPIC open_psg_allmol convention so the same
# scale_psg function works with both pipelines.
PSG_FITS_COLS = ("Wave/freq", "H2O", "CO2", "CH4", "CO", "O3", "N2O", "O2")


def save_average_psg_model(
    cfg_obj:   PipelineConfig,
    fits_path: str,
    am:        float = 1.0,
    surface_T: float = 280.0,
    h2o_abun:  float = 1.0,
    force_rebuild: bool = False,
) -> str:
    """
    Generate and save the average PSG telluric model as a FITS binary
    table.

    This is a one-time offline step.  The resulting file is loaded at
    runtime by load_average_psg_model() instead of calling PSG live,
    which is what the DRP primitive requires.

    The FITS table has one column per species in PSG_FITS_COLS.
    Wavelengths are stored in nm (ascending).  The per-species columns
    are the raw PSG transmission values at am=1.0 before any
    Beer-Lambert scaling — scale_psg applies the scaling at fit time.

    Parameters
    ----------
    cfg_obj   : PipelineConfig controlling wavelength range and PSG settings
    fits_path : output path for the FITS file
    am        : airmass for the PSG call (should be 1.0 for the base model)
    surface_T : surface temperature in K
    h2o_abun  : H2O abundance scaling (1.0 = template default)
    force_rebuild : if True, re-fetch from PSG even if fits_path exists

    Returns
    -------
    fits_path : path to the written FITS file
    """
    import astropy.io.fits as fits

    if not force_rebuild and os.path.exists(fits_path):
        print(f"[psg_fits] Average model already exists: {fits_path}")
        return fits_path

    print(f"[psg_fits] Generating average PSG model -> {fits_path}")
    arr, cols = fetch_stitched_spectrum(
        am=am,
        surface_T=surface_T,
        cfg_obj=cfg_obj,
        h2o_abun=h2o_abun,
        force_rebuild=force_rebuild,
    )
    if arr is None:
        raise RuntimeError("PSG fetch failed — cannot write average model FITS.")

    # Convert wavelength column to nm and sort ascending
    wave_idx = find_column(cols, "Wave") or 0
    lam_nm   = _wave_col_to_nm(arr[:, wave_idx])
    order    = np.argsort(lam_nm)
    lam_nm   = lam_nm[order]

    # Build one FITS column per species
    fits_columns = [
        fits.Column(name="Wave/freq", format="D", unit="nm", array=lam_nm)
    ]
    for label in PSG_MOL_LABELS:
        idx = find_column(cols, label)
        if idx is None:
            print(f"  [psg_fits] WARNING: {label} not found in PSG response, "
                  f"filling with ones")
            data = np.ones(len(lam_nm))
        else:
            data = np.nan_to_num(arr[order, idx].astype(float), nan=1.0)
        fits_columns.append(
            fits.Column(name=label, format="D", array=data)
        )

    hdu   = fits.BinTableHDU.from_columns(fits_columns)
    phdu  = fits.PrimaryHDU()

    # Store metadata in the primary header
    phdu.header["OBSDATE"]  = (cfg_obj.observation_date, "PSG observation date")
    phdu.header["SITE"]     = (cfg_obj.site.name,        "Observatory site")
    phdu.header["SITELAT"]  = (cfg_obj.site.latitude_deg,  "Site latitude deg N")
    phdu.header["SITELON"]  = (cfg_obj.site.longitude_deg, "Site longitude deg E")
    phdu.header["SITEALT"]  = (cfg_obj.site.altitude_km,   "Site altitude km")
    phdu.header["LAM_MIN"]  = (cfg_obj.wavelength_range_nm[0], "Wavelength min nm")
    phdu.header["LAM_MAX"]  = (cfg_obj.wavelength_range_nm[1], "Wavelength max nm")
    phdu.header["AIRMASS"]  = (am,        "Airmass of base PSG call")
    phdu.header["SURF_T"]   = (surface_T, "Surface temperature K")
    phdu.header["H2O_ABUN"] = (h2o_abun,  "H2O abundance scaling")
    phdu.header["NPTS"]     = (len(lam_nm), "Number of wavelength points")
    phdu.header["COMMENT"]  = "Average PSG telluric model for HISPEC/LIGER"
    phdu.header["COMMENT"]  = "Per-species columns are raw PSG transmission at am=1.0"
    phdu.header["COMMENT"]  = "Apply Beer-Lambert scaling via scale_psg()"

    hdulist = fits.HDUList([phdu, hdu])
    hdulist.writeto(fits_path, overwrite=True)
    print(
        f"[psg_fits] Wrote {fits_path}  "
        f"({len(lam_nm)} pts, "
        f"{cfg_obj.wavelength_range_nm[0]:.0f}-"
        f"{cfg_obj.wavelength_range_nm[1]:.0f} nm)"
    )
    return fits_path


def load_average_psg_model(
    fits_path: str,
    l0: Optional[float] = None,
    l1: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Load the average PSG telluric model from a FITS file.

    Equivalent to KPIC's open_psg_allmol but reads from the FITS
    format written by save_average_psg_model.

    Parameters
    ----------
    fits_path : path to the FITS file written by save_average_psg_model
    l0        : lower wavelength bound in nm (None = use full range)
    l1        : upper wavelength bound in nm (None = use full range)

    Returns
    -------
    lam_nm    : (N,) wavelength array in nm
    psg_tuple : (H2O, CO2, CH4, CO, O3, N2O, O2) transmission arrays
                ready to pass directly to scale_psg()
    """
    import astropy.io.fits as fits

    if not os.path.exists(fits_path):
        raise FileNotFoundError(
            f"Average PSG model not found: {fits_path}\n"
            "Run save_average_psg_model() to generate it."
        )

    with fits.open(fits_path) as hdulist:
        table = hdulist[1].data
        lam_nm = table["Wave/freq"].astype(float)
        mol_arrays = tuple(
            table[label].astype(float) for label in PSG_MOL_LABELS
        )

    # Trim to requested wavelength range
    mask = np.ones(len(lam_nm), bool)
    if l0 is not None:
        mask &= (lam_nm >= l0)
    if l1 is not None:
        mask &= (lam_nm <= l1)

    lam_nm    = lam_nm[mask]
    psg_tuple = tuple(arr[mask] for arr in mol_arrays)

    print(
        f"[psg_fits] Loaded {fits_path}  "
        f"({len(lam_nm)} pts, "
        f"{lam_nm.min():.1f}-{lam_nm.max():.1f} nm)"
    )
    return lam_nm, psg_tuple


# ============================================================
# GRID CACHING
# ============================================================

def _grid_hash(
    observation_date: str, site: SiteConfig,
    wavelength_range_nm: Tuple[float, float],
    airmass_grid: Sequence[float], temp_grid: Sequence[float],
) -> str:
    key = str((
        observation_date,
        (site.latitude_deg, site.longitude_deg, site.altitude_km),
        tuple(float(x) for x in wavelength_range_nm),
        tuple(sorted(float(x) for x in airmass_grid)),
        tuple(sorted(float(x) for x in temp_grid)),
    ))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _grid_is_valid(
    h5_filename: str, cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float], temp_grid: Sequence[float],
) -> bool:
    if not os.path.exists(h5_filename):
        return False
    expected = _grid_hash(
        cfg_obj.observation_date, cfg_obj.site,
        cfg_obj.wavelength_range_nm, airmass_grid, temp_grid,
    )
    try:
        with h5py.File(h5_filename, "r") as hf:
            return hf.attrs.get("grid_hash", "") == expected
    except Exception:
        return False


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
    Build the 3-D transmission grid: spectra[i_am, k_T, i_lambda].

    One PSG fetch (possibly multi-band stitched) per temperature node
    at am=1.0; scale_psg fills the airmass axis analytically.
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
    lam_min, lam_max = cfg_obj.wavelength_range_nm

    print(
        f"[grid] Building grid  date={cfg_obj.observation_date}  "
        f"site={cfg_obj.site.name}\n"
        f"       range={lam_min:.0f}-{lam_max:.0f} nm  "
        f"sub-band={cfg_obj.sub_band_width_nm:.0f} nm  "
        f"overlap={cfg_obj.sub_band_overlap_nm:.0f} nm\n"
        f"       airmasses={list(airmass_grid)}\n"
        f"       temps={list(temp_grid)}\n"
        f"       PSG fetches needed: {n_T}"
    )

    # Reference wavelength grid from first temperature node
    ref_arr, ref_cols = fetch_stitched_spectrum(
        1.0, temp_grid[0], cfg_obj, h2o_abun=1.0, force_rebuild=force_rebuild,
    )
    if ref_arr is None:
        raise RuntimeError("PSG returned no data for first temperature node")

    wave_idx = find_column(ref_cols, "Wave") or 0
    lam_ref  = _wave_col_to_nm(ref_arr[:, wave_idx])
    order    = np.argsort(lam_ref)
    lam_ref  = lam_ref[order]
    nlam     = len(lam_ref)

    grid_specs = np.full((n_am, n_T, nlam), np.nan, dtype=float)
    grid_h2o   = np.full_like(grid_specs, np.nan)

    for k, surface_T in enumerate(temp_grid):
        print(f"\n[grid] Temperature node {k+1}/{n_T}  T={surface_T:.1f} K")
        arr, cols = fetch_stitched_spectrum(
            1.0, surface_T, cfg_obj, h2o_abun=1.0, force_rebuild=force_rebuild,
        )
        if arr is None:
            raise RuntimeError(f"PSG returned no data for T={surface_T}")

        _, mol_tuple = extract_mol_columns(arr, cols, lam_ref)
        h2o_base     = mol_tuple[0]

        for i, am in enumerate(airmass_grid):
            scaled = scale_psg(mol_tuple, airmass=am, pwv=0.0)
            grid_specs[i, k, :] = scaled
            grid_h2o[i, k, :]   = np.clip(h2o_base, 1e-30, 1.0) ** float(am)

        print(f"  -> filled {n_am} airmass nodes  am={list(airmass_grid)}")

    _debug_airmass_variation(grid_specs, airmass_grid, temp_grid)

    gh = _grid_hash(
        cfg_obj.observation_date, cfg_obj.site,
        cfg_obj.wavelength_range_nm, airmass_grid, temp_grid,
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

    print(f"\n[grid] Wrote {cfg_obj.h5_filename}  "
          f"({nlam} wavelength pts, hash={gh})")
    return cfg_obj.h5_filename


def _debug_airmass_variation(grid_specs, airmass_grid, temp_grid):
    k_mid = len(temp_grid) // 2
    i_ref = len(airmass_grid) // 2
    ref   = grid_specs[i_ref, k_mid, :]
    print("\n[debug] Airmass variation (relative to middle node):")
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
    spectra, spectra_h2o, airmasses, temperatures, lam_grid = load_hdf5_grid(h5_filename)
    def _make(values):
        return RegularGridInterpolator(
            points=(airmasses, temperatures), values=values,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
    return _make(spectra), _make(spectra_h2o), lam_grid


def eval_interp(interp, am, surface_T):
    return interp([[float(am), float(surface_T)]])[0]


# ============================================================
# FORWARD MODEL
# ============================================================

def convolve_to_R(wave_nm, flux, R_inst):
    wave_nm = np.asarray(wave_nm, float)
    flux    = np.asarray(flux, float)
    lnw   = np.log(wave_nm)
    lnw_u = np.linspace(lnw.min(), lnw.max(), len(wave_nm))
    flux_u = np.interp(lnw_u, lnw, flux)
    sigma_lnw = (1.0 / R_inst) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dln       = lnw_u[1] - lnw_u[0]
    sigma_pix = sigma_lnw / dln
    flux_conv_u = gaussian_filter1d(flux_u, sigma_pix, mode="nearest")
    return np.interp(lnw, lnw_u, flux_conv_u)


def forward_model(T_base, dlam_nm, wave_obs, lam_grid, R_inst):
    T_conv       = convolve_to_R(lam_grid, T_base, R_inst)
    wave_shifted = np.asarray(wave_obs, float) - float(dlam_nm)
    return np.interp(wave_shifted, lam_grid, T_conv, left=np.nan, right=np.nan)


# ============================================================
# DIAGNOSTICS
# ============================================================

def check_forward_model_roundtrip(interp, lam_grid, R_inst,
                                   am=1.5, surface_T=280.0, sigma_ref=0.02):
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
        print("  WARNING: round-trip residual > 10% of sigma.")
    else:
        print("  OK -- forward model round-trip is clean.")
    return result


def check_fit_window(interp, lam_grid, fit_mask, am=1.5, surface_T=280.0,
                     min_mean=0.10, min_max=0.50):
    spec   = eval_interp(interp, am, surface_T)
    in_win = spec[fit_mask]
    t_min, t_max, t_mean = float(np.nanmin(in_win)), float(np.nanmax(in_win)), float(np.nanmean(in_win))
    n_pix = int(fit_mask.sum())
    print(f"\n[fit window]  N={n_pix}  min={t_min:.4f}  max={t_max:.4f}  mean={t_mean:.4f}")
    ok = True
    if t_mean < min_mean:
        print(f"  WARNING: mean transmission {t_mean:.4f} < {min_mean}"); ok = False
    if t_max < min_max:
        print(f"  WARNING: max transmission {t_max:.4f} < {min_max}"); ok = False
    if ok:
        print("  OK -- fit window has usable transmission structure.")
    return ok


# ============================================================
# FITTING
# ============================================================

def _solve_linear_continuum(T_model, wave_obs, flux_obs, sigma_obs, valid):
    lam  = wave_obs[valid]; lam0 = np.median(lam)
    w    = 1.0 / sigma_obs[valid] ** 2
    A    = np.vstack([T_model[valid], T_model[valid] * (lam - lam0)]).T
    Aw   = A * np.sqrt(w[:, None])
    yw   = flux_obs[valid] * np.sqrt(w)
    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    return float(coeffs[0]), float(coeffs[1])

def _valid_mask(T_model, flux_obs, sigma_obs, mask):
    return (np.asarray(mask, bool) & np.isfinite(T_model) & np.isfinite(flux_obs)
            & np.isfinite(sigma_obs) & (sigma_obs > 0) & (T_model > 0.0))

def chi2(params, interp, wave_obs, flux_obs, sigma_obs, mask,
         lam_grid, R_inst, T_fixed=None):
    if T_fixed is not None:
        am, dlam_nm = params; surface_T = T_fixed
    else:
        am, surface_T, dlam_nm = params
    T_base  = eval_interp(interp, am, surface_T)
    T_model = forward_model(T_base, dlam_nm, wave_obs, lam_grid, R_inst)
    valid   = _valid_mask(T_model, flux_obs, sigma_obs, mask)
    if valid.sum() < 50:
        return 1e99
    c0, c1 = _solve_linear_continuum(T_model, wave_obs, flux_obs, sigma_obs, valid)
    lam    = wave_obs[valid]; lam0 = np.median(lam)
    model  = (c0 + c1 * (lam - lam0)) * T_model[valid]
    resid  = flux_obs[valid] - model
    w      = 1.0 / sigma_obs[valid] ** 2
    return float(np.sum(resid ** 2 * w))


def fit_telluric(interp, wave_obs, flux_obs, sigma_obs, mask, lam_grid,
                 R_inst, T_fixed=280.0, x0_am=1.5, x0_dlam=0.0, x0_T=280.0,
                 am_bounds=(1.0, 2.5), dlam_bounds=(-0.5, 0.5),
                 T_bounds=(250.0, 320.0), optimizer="de"):
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
        print("[fit] Running Differential Evolution...")
        res = differential_evolution(objective, bounds=bounds, seed=42,
                                     popsize=20, tol=1e-6, polish=True, workers=1)
        print(f"[fit] DE converged: {res.success}  nfev={res.nfev}  chi2={res.fun:.4f}")
    elif optimizer == "lbfgsb":
        x0 = [x0_am, x0_dlam] if T_fixed is not None else [x0_am, x0_T, x0_dlam]
        print("[fit] Running L-BFGS-B...")
        res = minimize(objective, x0=np.array(x0, float), bounds=bounds, method="L-BFGS-B")
        print(f"[fit] L-BFGS-B converged: {res.success}  nfev={res.nfev}  chi2={res.fun:.4f}")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    if T_fixed is not None:
        am_fit, dlam_fit = res.x; T_fit = T_fixed
    else:
        am_fit, T_fit, dlam_fit = res.x

    best = {"airmass": float(am_fit), "surface_T": float(T_fit), "dlam_nm": float(dlam_fit)}
    return best, float(res.fun), res


def apply_telluric_correction(best_params, interp, wave_obs, flux_obs,
                               sigma_obs, mask, lam_grid, R_inst,
                               telluric_floor=1e-6):
    am, surface_T, dlam_nm = best_params["airmass"], best_params["surface_T"], best_params["dlam_nm"]
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
        "T_base": T_base, "T_model": T_model, "continuum": continuum,
        "corrected_flux": corrected_flux, "corrected_sigma": corrected_sigma,
    }


# ============================================================
# HIGH-LEVEL DRIVER
# ============================================================

# Build (or load from cache) the PSG transmission grid and return
# interpolators for total and H2O transmission on the native wavelength axis.
def prepare_pipeline(cfg_obj, airmass_grid, temp_grid, force_rebuild=False):
    build_grid_hdf5(cfg_obj=cfg_obj, airmass_grid=airmass_grid,
                    temp_grid=temp_grid, force_rebuild=force_rebuild)
    return make_telluric_interp(cfg_obj.h5_filename)




# ============================================================
# TELLURIC PRIMITIVE — DRP-FACING APPLY FUNCTION
# ============================================================

# Known FITS header keywords for airmass across Keck instruments.
# The list is checked in order; the first match wins.
AIRMASS_HEADER_KEYS = ("AIRMASS", "AMSTART", "AMEND", "MEAN_AM", "HIERARCH ESO TEL AIRM START")


def _read_airmass_from_header(header: dict) -> Optional[float]:
    """
    Try to read airmass from a FITS header dict.

    Checks AIRMASS_HEADER_KEYS in order and returns the first valid
    positive float found.  Returns None if no valid airmass keyword
    is present.
    """
    for key in AIRMASS_HEADER_KEYS:
        val = header.get(key)
        if val is not None:
            try:
                am = float(val)
                if am > 0:
                    return am
            except (TypeError, ValueError):
                continue
    return None


def apply_telluric_primitive(
    wave_obs:       np.ndarray,
    flux_obs:       np.ndarray,
    sigma_obs:      np.ndarray,
    model_fits:     str,
    output_fits:    str,
    header:         Optional[dict] = None,
    airmass:        Optional[float] = None,
    pwv:            float = 0.0,
    fit_airmass:    bool = False,
    fit_mask:       Optional[np.ndarray] = None,
    am_bounds:      Tuple[float, float] = (1.0, 2.5),
    l0:             Optional[float] = None,
    l1:             Optional[float] = None,
    telluric_floor: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Apply telluric correction using a pre-generated average PSG model.

    This is the DRP-facing function.  It loads the static average PSG
    FITS model, scales it to the observation airmass, divides it from
    the observed spectrum, and writes the corrected spectrum plus the
    telluric model to a FITS output file.

    Airmass resolution
    ------------------
    If fit_airmass=False (default):
        1.  FITS header  (if header is provided and contains a valid AIRMASS key)
        2.  airmass parameter  (explicit override)
        3.  Raises ValueError if neither is available

    If fit_airmass=True:
        The header or airmass parameter is used as the starting point
        for a Differential Evolution search over am_bounds.  The fitted
        value replaces the header value.  PWV is kept fixed (not fitted)
        to avoid the airmass-PWV degeneracy.

    Parameters
    ----------
    wave_obs       : (N,) wavelength array in nm
    flux_obs       : (N,) observed flux array
    sigma_obs      : (N,) per-pixel uncertainty array
    model_fits     : path to the average PSG model FITS file produced
                     by save_average_psg_model()
    output_fits    : path for the output FITS file
    header         : FITS header dict from the input observation
                     (used to read airmass and copy metadata)
    airmass        : starting airmass; used directly when fit_airmass=False,
                     or as a fallback x0 when fit_airmass=True and no
                     valid header airmass is found
    pwv            : PWV perturbation passed to scale_psg()
                     (0.0 = unperturbed template column, never fitted)
    fit_airmass    : if True, optimize airmass against the spectral data
                     rather than using the header value directly
    fit_mask       : boolean array selecting pixels used for fitting
                     (None = use all finite pixels in [l0, l1])
    am_bounds      : (min, max) search bounds for airmass when fitting
    l0             : lower wavelength trim bound in nm (None = no trim)
    l1             : upper wavelength trim bound in nm (None = no trim)
    telluric_floor : minimum telluric transmission before division to
                     avoid blowing up near saturated lines

    Returns
    -------
    dict with keys:
        corrected_flux   : telluric-corrected flux
        corrected_sigma  : propagated uncertainties
        telluric_model   : telluric transmission spectrum on wave_obs grid
        airmass_used     : airmass value that was applied
        airmass_fitted   : True if airmass was fitted, False if taken from header
    """
    import astropy.io.fits as fits

    # ----------------------------------------------------------
    # 1.  Resolve starting airmass
    # ----------------------------------------------------------
    am_init = None
    if header is not None:
        am_init = _read_airmass_from_header(header)
        if am_init is not None:
            print(f"[primitive] Airmass from FITS header: {am_init:.4f}")

    if am_init is None and airmass is not None:
        am_init = float(airmass)
        print(f"[primitive] Airmass from parameter: {am_init:.4f}")

    if am_init is None and not fit_airmass:
        raise ValueError(
            "No airmass available.  Provide a FITS header with an AIRMASS "
            "keyword, pass airmass= explicitly, or set fit_airmass=True."
        )

    # If fitting without a starting point, use the midpoint of am_bounds
    if am_init is None:
        am_init = float(np.mean(am_bounds))
        print(f"[primitive] No header airmass — using am_bounds midpoint "
              f"as x0: {am_init:.4f}")

    # ----------------------------------------------------------
    # 2.  Load average PSG model and trim to requested range
    # ----------------------------------------------------------
    lam_model, psg_tuple = load_average_psg_model(
        fits_path = model_fits,
        l0        = l0,
        l1        = l1,
    )

    # ----------------------------------------------------------
    # 3.  Fit airmass if requested
    #     PWV is held fixed — fitting both simultaneously is
    #     unreliable due to the airmass-PWV degeneracy.
    # ----------------------------------------------------------
    airmass_fitted = False

    if fit_airmass:
        # Build the fit mask from finite pixels in the trim range
        if fit_mask is None:
            fit_mask_use = np.isfinite(flux_obs) & np.isfinite(sigma_obs)
            if l0 is not None:
                fit_mask_use &= (wave_obs >= l0)
            if l1 is not None:
                fit_mask_use &= (wave_obs <= l1)
        else:
            fit_mask_use = np.asarray(fit_mask, bool)

        def _chi2_am(params):
            """Chi-square over airmass only; PWV fixed."""
            am_try = float(params[0])
            tel_native = scale_psg(psg_tuple, airmass=am_try, pwv=pwv)
            tel_on_obs = np.interp(
                wave_obs, lam_model, tel_native,
                left=np.nan, right=np.nan,
            )
            valid = (
                fit_mask_use
                & np.isfinite(tel_on_obs)
                & np.isfinite(flux_obs)
                & np.isfinite(sigma_obs)
                & (sigma_obs > 0)
                & (tel_on_obs > 0)
            )
            if valid.sum() < 50:
                return 1e99

            # Solve for a linear continuum scale analytically
            T  = tel_on_obs[valid]
            d  = flux_obs[valid]
            w  = 1.0 / sigma_obs[valid] ** 2
            # scalar continuum: flux = c * T  =>  c = sum(w*d*T) / sum(w*T^2)
            c  = np.sum(w * d * T) / np.sum(w * T ** 2)
            resid = d - c * T
            return float(np.sum(resid ** 2 * w))

        print(f"[primitive] Fitting airmass in {am_bounds}  "
              f"(PWV fixed at {pwv:.3f})...")
        from scipy.optimize import differential_evolution
        res = differential_evolution(
            _chi2_am,
            bounds=[am_bounds],
            seed=42,
            popsize=15,
            tol=1e-6,
            polish=True,
            workers=1,
        )
        am_used = float(res.x[0])
        airmass_fitted = True
        print(f"[primitive] Fitted airmass: {am_used:.4f}  "
              f"(header/param was {am_init:.4f})  chi2={res.fun:.4f}")
    else:
        am_used = am_init

    # ----------------------------------------------------------
    # 4.  Scale telluric model to the resolved airmass
    # ----------------------------------------------------------
    telluric_native = scale_psg(psg_tuple, airmass=am_used, pwv=pwv)

    telluric_on_obs = np.interp(
        wave_obs, lam_model, telluric_native,
        left=np.nan, right=np.nan,
    )

    # ----------------------------------------------------------
    # 5.  Divide out telluric model
    # ----------------------------------------------------------
    T_safe          = np.clip(telluric_on_obs, telluric_floor, np.inf)
    corrected_flux  = flux_obs  / T_safe
    corrected_sigma = sigma_obs / T_safe

    # ----------------------------------------------------------
    # 6.  Write FITS output
    #     Extension 0 (Primary) : metadata header
    #     Extension 1           : corrected flux + sigma
    #     Extension 2           : telluric model on obs grid
    # ----------------------------------------------------------
    phdu = fits.PrimaryHDU()

    # Copy input observation metadata if available
    if header is not None:
        for key, val in header.items():
            if key in ("SIMPLE", "EXTEND", "NAXIS", "NAXIS1",
                       "BITPIX", "", "END"):
                continue
            try:
                phdu.header[key] = val
            except Exception:
                pass

    # Record what was applied
    phdu.header["TCORR"]   = (True,             "Telluric correction applied")
    phdu.header["TCMODEL"] = (os.path.basename(model_fits), "Telluric model file")
    phdu.header["TCAIRM"]  = (am_used,           "Airmass used for telluric scaling")
    phdu.header["TCAMFIT"] = (airmass_fitted,     "True if airmass was fitted to data")
    phdu.header["TCPWV"]   = (pwv,               "PWV perturbation applied (fixed)")
    phdu.header["TCFLOOR"] = (telluric_floor,     "Telluric floor before division")
    if l0 is not None:
        phdu.header["TCL0"] = (l0, "Trim wavelength min nm")
    if l1 is not None:
        phdu.header["TCL1"] = (l1, "Trim wavelength max nm")

    # Extension 1: corrected spectrum
    spec_hdu = fits.ImageHDU(
        data = np.vstack([wave_obs, corrected_flux, corrected_sigma]),
        name = "CORRECTED",
    )
    spec_hdu.header["ROW0"] = "wavelength_nm"
    spec_hdu.header["ROW1"] = "corrected_flux"
    spec_hdu.header["ROW2"] = "corrected_sigma"

    # Extension 2: telluric model so the science team can inspect it
    tel_hdu = fits.ImageHDU(
        data = np.vstack([wave_obs, telluric_on_obs]),
        name = "TELLURIC_MODEL",
    )
    tel_hdu.header["ROW0"] = "wavelength_nm"
    tel_hdu.header["ROW1"] = "telluric_transmission"

    hdulist = fits.HDUList([phdu, spec_hdu, tel_hdu])
    hdulist.writeto(output_fits, overwrite=True)
    print(
        f"[primitive] Wrote {output_fits}  "
        f"airmass={am_used:.4f} (fitted={airmass_fitted})  "
        f"pwv={pwv:.3f}  floor={telluric_floor:.1e}"
    )

    return {
        "corrected_flux":  corrected_flux,
        "corrected_sigma": corrected_sigma,
        "telluric_model":  telluric_on_obs,
        "airmass_used":    am_used,
        "airmass_fitted":  airmass_fitted,
    }


# ============================================================
# FITS SPECTRUM READER
# ============================================================

# Extension names to try when looking for spectrum data, in order.
# HISPEC/LIGER names will be added here once the DRP format is finalised.
SPECTRUM_EXTNAMES = (
    "SPECTRUM", "SCI", "SCIENCE", "FLUX", "DATA", "PRIMARY",
)

# Header keywords to try for the wavelength start/step (WCS style)
WCS_CRVAL = ("CRVAL1",)
WCS_CDELT = ("CDELT1", "CD1_1")
WCS_CRPIX = ("CRPIX1",)


def read_spectrum_fits(
    fits_path:      str,
    wave_ext:       Optional[str] = None,
    flux_ext:       Optional[str] = None,
    sigma_ext:      Optional[str] = None,
    wave_keyword:   Optional[str] = None,
    flux_keyword:   Optional[str] = None,
    sigma_keyword:  Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Read a 1-D spectrum from a FITS file and return arrays ready
    to pass to apply_telluric_primitive().

    The reader tries several common layouts in order and returns
    the first one that works.  Once the HISPEC/LIGER DRP format is
    finalised, explicit ext/keyword overrides should be passed in
    to skip the auto-detection.

    Layout detection order
    ----------------------
    1.  Caller-supplied extension names / column keywords (highest priority)
    2.  Binary table with named columns (WAVE/FLUX/SIGMA or similar)
    3.  Separate image extensions for flux and sigma
    4.  Single image extension with WCS wavelength axis

    Parameters
    ----------
    fits_path     : path to the input FITS file
    wave_ext      : extension name or index containing the wavelength array
    flux_ext      : extension name or index containing the flux array
    sigma_ext     : extension name or index containing the uncertainty array
                    (None = fill with ones if not found)
    wave_keyword  : binary-table column name for wavelength
    flux_keyword  : binary-table column name for flux
    sigma_keyword : binary-table column name for uncertainty

    Returns
    -------
    wave_obs  : (N,) wavelength array in nm
    flux_obs  : (N,) flux array
    sigma_obs : (N,) uncertainty array (ones if not found)
    header    : combined primary + data extension header dict
    """
    import astropy.io.fits as fits

    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"Spectrum FITS not found: {fits_path}")

    with fits.open(fits_path) as hdulist:
        # Merge primary header with first data extension header
        primary_header = dict(hdulist[0].header)

        # -------------------------------------------------------
        # Strategy 1: Caller provided explicit extension names
        # -------------------------------------------------------
        if flux_ext is not None:
            flux_hdu = hdulist[flux_ext]
            header   = {**primary_header, **dict(flux_hdu.header)}

            # Try to get wavelength
            if wave_ext is not None:
                wave_nm = np.asarray(hdulist[wave_ext].data, float).ravel()
            else:
                wave_nm = _wcs_wavelength(flux_hdu.header, len(flux_hdu.data.ravel()))

            flux_obs  = np.asarray(flux_hdu.data, float).ravel()
            sigma_obs = (
                np.asarray(hdulist[sigma_ext].data, float).ravel()
                if sigma_ext is not None else np.ones_like(flux_obs)
            )
            print(f"[reader] Loaded via explicit extensions from {fits_path}")
            return wave_nm, flux_obs, sigma_obs, header

        # -------------------------------------------------------
        # Strategy 2: Binary table with named columns
        # -------------------------------------------------------
        for extname in SPECTRUM_EXTNAMES:
            try:
                hdu = hdulist[extname]
            except (KeyError, IndexError):
                continue
            if hdu.data is None:
                continue
            try:
                cols = hdu.columns.names
            except AttributeError:
                continue   # not a table

            header = {**primary_header, **dict(hdu.header)}

            wave_col  = wave_keyword  or _find_wave_col(cols)
            flux_col  = flux_keyword  or _find_flux_col(cols)
            sigma_col = sigma_keyword or _find_sigma_col(cols)

            if flux_col is None:
                continue

            flux_obs  = np.asarray(hdu.data[flux_col],  float).ravel()
            sigma_obs = (
                np.asarray(hdu.data[sigma_col], float).ravel()
                if sigma_col is not None else np.ones_like(flux_obs)
            )

            if wave_col is not None:
                wave_nm = np.asarray(hdu.data[wave_col], float).ravel()
            else:
                wave_nm = _wcs_wavelength(hdu.header, len(flux_obs))

            print(f"[reader] Loaded binary table ext='{extname}' "
                  f"wave='{wave_col}' flux='{flux_col}' "
                  f"sigma='{sigma_col}' from {fits_path}")
            return wave_nm, flux_obs, sigma_obs, header

        # -------------------------------------------------------
        # Strategy 3: Separate image extensions
        # -------------------------------------------------------
        flux_hdu  = None
        sigma_hdu = None

        for extname in SPECTRUM_EXTNAMES:
            try:
                hdu = hdulist[extname]
                if hdu.data is not None and hdu.data.ndim >= 1:
                    flux_hdu = hdu
                    break
            except (KeyError, IndexError):
                continue

        if flux_hdu is None and hdulist[0].data is not None:
            flux_hdu = hdulist[0]

        if flux_hdu is not None:
            header    = {**primary_header, **dict(flux_hdu.header)}
            flux_obs  = np.asarray(flux_hdu.data, float).ravel()
            wave_nm   = _wcs_wavelength(flux_hdu.header, len(flux_obs))

            # Look for a sigma/error extension
            for err_name in ("SIGMA", "ERR", "ERROR", "NOISE", "IVAR", "VAR"):
                try:
                    ehdu = hdulist[err_name]
                    if ehdu.data is not None:
                        raw = np.asarray(ehdu.data, float).ravel()
                        # Convert inverse variance to sigma if needed
                        if "IVAR" in err_name or "VAR" in err_name:
                            with np.errstate(divide="ignore", invalid="ignore"):
                                sigma_obs = np.where(raw > 0, 1.0 / np.sqrt(raw), np.inf)
                        else:
                            sigma_obs = raw
                        print(f"[reader] Loaded image extensions, "
                              f"sigma from '{err_name}' in {fits_path}")
                        return wave_nm, flux_obs, sigma_obs, header
                except (KeyError, IndexError):
                    continue

            sigma_obs = np.ones_like(flux_obs)
            print(f"[reader] Loaded image extension; no sigma found, "
                  f"using ones. From {fits_path}")
            return wave_nm, flux_obs, sigma_obs, header

    raise ValueError(
        f"Could not read spectrum from {fits_path}.\n"
        "Try passing explicit wave_ext, flux_ext, sigma_ext."
    )


def _wcs_wavelength(header, n: int) -> np.ndarray:
    """
    Reconstruct a wavelength axis from WCS keywords in a FITS header.
    Returns wavelength in nm (converts from Angstroms if CUNIT1 is set).
    Falls back to pixel indices if no WCS keywords are found.
    """
    crval = None
    cdelt = None
    crpix = 1.0

    for key in WCS_CRVAL:
        if key in header:
            crval = float(header[key]); break
    for key in WCS_CDELT:
        if key in header:
            cdelt = float(header[key]); break
    for key in WCS_CRPIX:
        if key in header:
            crpix = float(header[key]); break

    if crval is None or cdelt is None:
        print("  [reader] No WCS keywords found — using pixel indices as wavelength")
        return np.arange(n, dtype=float)

    pix  = np.arange(1, n + 1, dtype=float)
    wave = crval + cdelt * (pix - crpix)

    # Convert Angstroms to nm if needed
    cunit = str(header.get("CUNIT1", "")).strip().lower()
    if cunit in ("angstrom", "angstroms", "a"):
        wave = wave / 10.0
    elif cunit in ("um", "micron", "microns"):
        wave = wave * 1000.0
    elif cunit in ("m", "meter", "meters"):
        wave = wave * 1e9

    return wave


def _find_wave_col(cols: list) -> Optional[str]:
    for name in ("WAVE", "WAVELENGTH", "LAMBDA", "WAV", "WVLG"):
        for c in cols:
            if c.upper() == name:
                return c
    return None


def _find_flux_col(cols: list) -> Optional[str]:
    for name in ("FLUX", "SCI", "SCIENCE", "DATA", "SPEC", "INTENSITY"):
        for c in cols:
            if c.upper() == name:
                return c
    return None


def _find_sigma_col(cols: list) -> Optional[str]:
    for name in ("SIGMA", "ERR", "ERROR", "NOISE", "UNC", "UNCERTAINTY"):
        for c in cols:
            if c.upper() == name:
                return c
    return None


# ============================================================
# SPECTRUM TRIMMING UTILITY
# ============================================================

# HISPEC channel boundaries in nm.
# BSPEC covers y+J band (silica fiber),  RSPEC covers H+K band (ZBLAN fiber).
# The dichroic split is at approximately 1400 nm based on the fiber material
# change documented in the HISPEC fiber delivery subsystem paper (Jovanovic
# et al. 2024, SPIE 13096).  Confirm the exact value before
# using these in production — the instrument is pre-commissioning and the
# boundary may shift slightly in the final design.
HISPEC_BSPEC_LAM_MIN =  980.0   # nm  (y-band blue edge)
HISPEC_BSPEC_LAM_MAX = 1400.0   # nm  (yJ / HK dichroic, approx.)
HISPEC_RSPEC_LAM_MIN = 1400.0   # nm  (HK red edge of dichroic, approx.)
HISPEC_RSPEC_LAM_MAX = 2500.0   # nm  (K-band red edge)


def trim_spectrum(
    lam:     np.ndarray,
    flux:    np.ndarray,
    l0:      Optional[float] = None,
    l1:      Optional[float] = None,
    channel: Optional[str]  = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim a spectrum to a wavelength range.

    Parameters
    ----------
    lam     : wavelength array in nm
    flux    : flux array (same length as lam)
    l0      : lower bound in nm  (None = no lower trim)
    l1      : upper bound in nm  (None = no upper trim)
    channel : 'bspec' or 'rspec' — sets l0/l1 to the HISPEC channel
              boundaries automatically.  Explicit l0/l1 override the
              channel defaults if provided alongside channel.

              NOTE: the bspec/rspec boundaries are approximate (see
              HISPEC_BSPEC_LAM_MAX / HISPEC_RSPEC_LAM_MIN constants).
              Confirm with your advisor before using in production.

    Returns
    -------
    lam_trimmed, flux_trimmed
    """
    # Apply channel defaults first, then let explicit l0/l1 override
    if channel is not None:
        ch = channel.lower().strip()
        if ch == "bspec":
            if l0 is None:
                l0 = HISPEC_BSPEC_LAM_MIN
            if l1 is None:
                l1 = HISPEC_BSPEC_LAM_MAX
        elif ch == "rspec":
            if l0 is None:
                l0 = HISPEC_RSPEC_LAM_MIN
            if l1 is None:
                l1 = HISPEC_RSPEC_LAM_MAX
        else:
            raise ValueError(
                f"Unknown channel '{channel}'. Choose 'bspec' or 'rspec'."
            )

    mask = np.ones(len(lam), bool)
    if l0 is not None:
        mask &= (lam >= l0)
    if l1 is not None:
        mask &= (lam <= l1)
    return lam[mask], flux[mask]


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def _lam_mask(lam_grid, lam_range):
    if lam_range is None:
        return np.ones(len(lam_grid), bool)
    return (lam_grid >= lam_range[0]) & (lam_grid <= lam_range[1])

def _finish_plot(title, lam_range):
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Transmission")
    plt.title(title)
    if lam_range is not None: plt.xlim(*lam_range)
    plt.ylim(0, 1.05); plt.legend(); plt.tight_layout(); plt.show()

def plot_vs_airmass(interp, lam_grid, surface_T=280.0,
                    am_list=(1.0, 1.2, 1.5, 2.0), lam_range=None):
    mask = _lam_mask(lam_grid, lam_range)
    plt.figure(figsize=(12, 5))
    for am in am_list:
        spec = eval_interp(interp, am, surface_T)
        plt.plot(lam_grid[mask], spec[mask], label=f"AM={am:.2f}")
    _finish_plot(f"Transmission vs Airmass  (T={surface_T:.1f} K)", lam_range)

def plot_best_fit(wave_obs, flux_obs, sigma_obs, interp, lam_grid, best_params, R_inst):
    products   = apply_telluric_correction(
        best_params, interp, wave_obs, flux_obs, sigma_obs,
        np.ones(len(wave_obs), bool), lam_grid, R_inst)
    model_full = products["continuum"] * products["T_model"]
    plt.figure(figsize=(12, 5))
    plt.plot(wave_obs, flux_obs, lw=1, label="Observed")
    plt.plot(wave_obs, model_full, lw=1.2, label="Best-fit model")
    plt.fill_between(wave_obs, flux_obs - sigma_obs, flux_obs + sigma_obs,
                     alpha=0.25, label="1sigma")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Flux")
    plt.title(f"Best-fit  AM={best_params['airmass']:.3f}  "
              f"dlam={best_params['dlam_nm']:.4f} nm")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_corrected_spectrum(wave_obs, products):
    plt.figure(figsize=(12, 5))
    plt.plot(wave_obs, products["corrected_flux"], lw=1, label="Corrected flux")
    plt.fill_between(wave_obs,
                     products["corrected_flux"] - products["corrected_sigma"],
                     products["corrected_flux"] + products["corrected_sigma"],
                     alpha=0.25, label="1sigma")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Corrected Flux")
    plt.title("Telluric-corrected Spectrum")
    plt.legend(); plt.tight_layout(); plt.show()

# def plot_am_dlam_heatmap(interp, wave_obs, flux_obs, sigma_obs, mask, lam_grid,
#                           R_inst, T_fixed=280.0, am_bounds=(1.0, 2.5),
#                           dlam_bounds=(-0.5, 0.5), n_am=80, n_dlam=80):
#     am_vals   = np.linspace(*am_bounds, n_am)
#     dlam_vals = np.linspace(*dlam_bounds, n_dlam)
#     chi2_map  = np.full((n_dlam, n_am), np.nan)
#     for j, dlam in enumerate(dlam_vals):
#         for i, am in enumerate(am_vals):
#             chi2_map[j, i] = chi2([am, dlam], interp, wave_obs, flux_obs,
#                                   sigma_obs, mask, lam_grid, R_inst, T_fixed=T_fixed)
#     finite = np.isfinite(chi2_map) & (chi2_map < 1e98)
#     dchi2_map = np.full_like(chi2_map, np.nan)
#     dchi2_map[finite] = chi2_map[finite] - np.nanmin(chi2_map[finite])
#     min_idx   = np.unravel_index(np.nanargmin(chi2_map), chi2_map.shape)
#     best_am, best_dlam = am_vals[min_idx[1]], dlam_vals[min_idx[0]]
#     plt.figure(figsize=(9, 7))
#     im = plt.imshow(dchi2_map, origin="lower", aspect="auto",
#                     extent=[am_vals[0], am_vals[-1], dlam_vals[0], dlam_vals[-1]])
#     plt.colorbar(im, label=r"$\Delta\chi^2$")
#     plt.plot(best_am, best_dlam, "wx", ms=10, mew=2,
#              label=f"Best: AM={best_am:.3f}, dlam={best_dlam:.4f} nm")
#     plt.xlabel("Airmass"); plt.ylabel(r"$\delta\lambda$ (nm)")
#     plt.title(f"$\\Delta\\chi^2$ Heat Map  (T={T_fixed:.1f} K)")
#     plt.legend(); plt.tight_layout(); plt.show()
#     print(f"\n[heatmap] Best: AM={best_am:.4f}  dlam={best_dlam:.4f} nm  "
#           f"chi2={chi2_map[min_idx]:.4f}")
#     return am_vals, dlam_vals, chi2_map


# ============================================================
# SYNTHETIC OBSERVATION
# ============================================================

def make_synthetic_observation(interp, lam_grid, R_inst, true_am=1.5,
                                true_T=280.0, true_dlam=0.015,
                                sigma=0.02, seed=42):
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

def plot_raw_psg_response(cfg_obj, am=1.5, surface_T=280.0,
                           R_mid=25_000.0, R_inst=8_000.0, lam_range=None):
    print(f"[plot_raw_psg_response] Fetching: am={am}, T={surface_T}")
    arr, cols = fetch_stitched_spectrum(am, surface_T, cfg_obj, h2o_abun=1.0)
    if arr is None:
        print("PSG fetch failed."); return

    wave_idx  = find_column(cols, "Wave")  or 0
    total_idx = find_column(cols, PSG_TOTAL_LABEL) or 1
    h2o_idx   = find_column(cols, PSG_H2O_LABEL)

    lam_nm = _wave_col_to_nm(arr[:, wave_idx])
    order  = np.argsort(lam_nm); lam_nm = lam_nm[order]
    total  = arr[order, total_idx]
    h2o    = arr[order, h2o_idx] if h2o_idx is not None else total

    if lam_range is None:
        lam_range = cfg_obj.wavelength_range_nm
    mask = (lam_nm >= lam_range[0]) & (lam_nm <= lam_range[1])
    lam  = lam_nm[mask]

    total_raw  = total[mask]
    total_mid  = convolve_to_R(lam, total_raw, R_mid)
    total_inst = convolve_to_R(lam, total_raw, R_inst)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"PSG Telluric  AM={am:.2f}  T={surface_T:.0f} K", fontsize=13)
    axes[0].plot(lam, total_raw,  lw=0.4, color="steelblue")
    axes[0].set_ylabel("Transmittance"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Raw PSG H2O  (native resolution)")
    axes[1].plot(lam, total_mid,  lw=0.7, color="steelblue")
    axes[1].set_ylabel("Transmittance"); axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(f"Convolved H2O at R={int(R_mid):,}")
    axes[2].plot(lam, total_inst, lw=0.8, color="steelblue")
    axes[2].set_ylabel("Transmittance"); axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title(f"Convolved Total at R={int(R_inst):,}")
    axes[2].set_xlabel("Wavelength (nm)")
    plt.tight_layout(); plt.show()

plot_raw_psg_three_panel = plot_raw_psg_response


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    cfg = PipelineConfig(
        observation_date    = "2024/01/15 05:00",
        site                = SiteConfig(),
        wavelength_range_nm = (960.0, 2500.0),
        sub_band_width_nm   = 500.0,
        sub_band_overlap_nm = 10.0,
        h5_filename         = "telluric_grid.h5",
    )

    airmass_grid = [1.0, 1.2, 1.5, 2.0]
    temp_grid    = [280.0]

    interp, interp_h2o, lam_grid = prepare_pipeline(
        cfg_obj=cfg, airmass_grid=airmass_grid,
        temp_grid=temp_grid, force_rebuild=True,
    )

    # ----------------------------------------------------------
    # 2b. Generate the average PSG model FITS file
    #     One-time offline step: PSG called at am=1.0 with typical
    #     Maunakea conditions and saved as a FITS binary table.
    #     The DRP primitive loads this file at runtime instead of
    #     calling PSG live.
    # ----------------------------------------------------------
    avg_fits = save_average_psg_model(
        cfg_obj   = cfg,
        fits_path = "average_psg_model.fits",
        am        = 1.0,
        surface_T = 280.0,
        h2o_abun  = 1.0,
        force_rebuild = False,
    )

    # Verify the loader round-trips correctly
    lam_avg, psg_tuple_avg = load_average_psg_model(
        fits_path = avg_fits,
        l0        = 960.0,
        l1        = 2500.0,
    )
    print(f"[verify] Loaded average model: {len(lam_avg)} pts  "
          f"H2O range: {psg_tuple_avg[0].min():.4f} - {psg_tuple_avg[0].max():.4f}")

    T_FIXED = 280.0
    R_INST  = 8_000.0

    roundtrip = check_forward_model_roundtrip(
        interp, lam_grid, R_INST, am=1.5, surface_T=T_FIXED, sigma_ref=0.02)

    plot_raw_psg_response(cfg, am=1.5, surface_T=T_FIXED,
                          R_mid=25_000.0, R_inst=R_INST)

    _arr_mol, _cols_mol = fetch_stitched_spectrum(1.5, T_FIXED, cfg, h2o_abun=1.0)
    if _arr_mol is not None:
        _, _mol_tuple = extract_mol_columns(_arr_mol, _cols_mol, lam_grid)
        plot_molecular_species(lam_grid, _mol_tuple, R_inst=R_INST,
                               lam_range=(960.0, 2600.0),
                               title=f"Molecular Species  AM=1.5  T={T_FIXED:.0f} K")

    true_am, true_T, true_dlam = 1.5, T_FIXED, 0.10
    wave_obs, flux_obs, sigma_obs = make_synthetic_observation(
        interp, lam_grid, R_INST, true_am=true_am,
        true_T=true_T, true_dlam=true_dlam, sigma=0.02)

    fit_mask = np.isfinite(wave_obs) & (wave_obs >= 1550.0) & (wave_obs <= 1750.0)
    check_fit_window(interp, lam_grid, fit_mask, am=true_am, surface_T=T_FIXED)

    best_params, chi2_val, res = fit_telluric(
        interp, wave_obs, flux_obs, sigma_obs, fit_mask, lam_grid, R_INST,
        T_fixed=T_FIXED, am_bounds=(1.0, 2.5), dlam_bounds=(-0.5, 0.5),
        optimizer="de")

    n_pix = int(fit_mask.sum())
    print(f"\n--- Smoke test ---")
    print(f"  True:      am={true_am}  dlam={true_dlam}  T={true_T}")
    print(f"  Recovered: am={best_params['airmass']:.4f}  "
          f"dlam={best_params['dlam_nm']:.4f}  T={best_params['surface_T']:.1f}")
    print(f"  chi2={chi2_val:.4f}  N={n_pix}  chi2/N={chi2_val/n_pix:.3f}")

    products = apply_telluric_correction(
        best_params, interp, wave_obs, flux_obs, sigma_obs,
        fit_mask, lam_grid, R_INST)

    plot_vs_airmass(interp, lam_grid, surface_T=T_FIXED)
    plot_vs_airmass(interp, lam_grid, surface_T=T_FIXED, lam_range=(1550.0, 1750.0))
    plot_best_fit(wave_obs, flux_obs, sigma_obs, interp, lam_grid, best_params, R_INST)
    plot_corrected_spectrum(wave_obs, products)

    print("\nPipeline complete.")