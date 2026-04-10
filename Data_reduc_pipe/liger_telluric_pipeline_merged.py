"""
Telluric correction pipeline for Keck LIGER

This version merges:
- the scientifically safer PWV handling from the calibrated/user-controlled
  atmosphere version
- the cleaner software structure from the later rewrite

Design choices
--------------
1. User-controlled atmosphere mode is enabled by default to avoid PSG/MERRA-2
   silently overriding H2O and making the PWV axis meaningless.
2. PWV scaling uses an empirically calibrated TEMPLATE_PWV_MM, not an analytic
   layer integral.
3. PSG column names are parsed from headers whenever possible.
4. Grid caching is keyed to wavelength config, atmospheric axes, and
   TEMPLATE_PWV_MM so recalibration invalidates stale grids.
5. Temperature is supported as a grid axis, but it is explicitly validated.
   If PSG produces identical spectra across the temperature axis, you should
   collapse temp_grid to a single value in production.

Paste this file directly into VS Code as a single .py module.
"""

from __future__ import annotations

import io
import os
import re
import time
import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any
import matplotlib.pyplot as plt
import h5py
import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


# ============================================================
# CONFIG
# ============================================================

PSG_API = "https://psg.gsfc.nasa.gov/api.php"
TEMPLATE_PATH = "earth_cfg.txt"
OUT_DIR = "telluric_grid"

PSG_TOTAL_LABEL = "Total"
PSG_H2O_LABEL = "H2O"

# User-controlled atmosphere mode:
# Sets a non-MERRA date and explicit gas list so PSG respects the user layers.
USE_OPTION_B_USER_ATMOSPHERE = True

# How to scale H2O once user atmosphere mode is active.
# "layers" directly patches ATMOSPHERE-LAYER-i H2O VMR values.
# "abun" scales the H2O entry of ATMOSPHERE-ABUN.
H2O_SCALING_MODE = "abun"

# Empirically calibrated PWV of the unscaled template in mm.
# Re-run calibration if earth_cfg.txt changes.
TEMPLATE_PWV_MM = 20.0935

# Grid-build resolution. Lower than 300k is usually much more practical.
DEFAULT_WAVELENGTH_CFG_NM = (1500.0, 2500.0, 50000.0)


@dataclass(frozen=True)
class PipelineConfig:
    template_path: str = TEMPLATE_PATH
    out_dir: str = OUT_DIR
    h5_filename: str = "telluric_grid.h5"
    psg_api: str = PSG_API
    use_option_b_user_atmosphere: bool = USE_OPTION_B_USER_ATMOSPHERE
    h2o_scaling_mode: str = H2O_SCALING_MODE
    template_pwv_mm: float = TEMPLATE_PWV_MM
    wavelength_cfg_nm: Tuple[float, float, float] = DEFAULT_WAVELENGTH_CFG_NM
    psg_timeout_s: int = 180
    psg_max_tries: int = 10
    debug_dump: bool = True


# ============================================================
# BASIC HELPERS
# ============================================================

def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_psg_template(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PSG template not found: {path}. Download it manually from PSG."
        )
    with open(path, "r") as f:
        cfg = f.read()
    if "<ATMOSPHERE-LAYER" not in cfg:
        raise RuntimeError("Invalid PSG template: no ATMOSPHERE-LAYER entries found.")
    return cfg


def _get_tag(cfg: str, tag: str) -> Optional[str]:
    m = re.search(rf"^<{re.escape(str(tag))}>\s*(.*)\s*$", cfg, flags=re.M)
    return None if m is None else m.group(1).strip()


def set_parameter(cfg: str, tag: str, value: Any) -> str:
    """
    Replace or append a <TAG> line in the config.
    """
    val = str(value)
    pat = rf"(^<{re.escape(tag)}>\s*).*$"
    new, n = re.subn(pat, rf"\g<1>{val}", cfg, flags=re.MULTILINE)
    if n == 0:
        if not cfg.endswith("\n"):
            cfg += "\n"
        new = cfg + f"<{tag}> {val}\n"
    return new


def am_to_zenith_deg(am: float) -> float:
    am = max(float(am), 1.0)
    return float(np.degrees(np.arccos(np.clip(1.0 / am, 0.0, 1.0))))


def _stable_cfg_hash(cfg_text: str) -> str:
    lines = [ln.rstrip() for ln in cfg_text.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    normalized = "\n".join(lines) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def force_user_atmosphere_mode(cfg: str) -> str:
    """
    Force PSG to use a user-controlled atmosphere rather than reanalysis-driven
    Earth atmosphere.
    """
    cfg = set_parameter(cfg, "OBJECT-DATE", "1970/01/01 00:00")
    cfg = set_parameter(cfg, "ATMOSPHERE-NGAS", "8")
    cfg = set_parameter(
        cfg,
        "ATMOSPHERE-TYPE",
        "HIT[1],HIT[2],HIT[3],HIT[4],HIT[5],HIT[6],HIT[7],HIT[22]",
    )
    return cfg


def _apply_geometry_and_range(cfg: str, am: float, wavelength_cfg_nm) -> str:
    lam_min_nm, lam_max_nm, rp = wavelength_cfg_nm
    zen = am_to_zenith_deg(am)

    cfg = set_parameter(cfg, "GEOMETRY", "Lookingup")
    cfg = set_parameter(cfg, "GEOMETRY-USER-PARAM", f"{zen:.6f}")
    cfg = set_parameter(cfg, "GEOMETRY-OBS-ANGLE", f"{zen:.6f}")

    wn1 = 1e7 / lam_max_nm
    wn2 = 1e7 / lam_min_nm
    cfg = set_parameter(cfg, "GENERATOR-RANGEUNIT", "cm-1")
    cfg = set_parameter(cfg, "GENERATOR-RANGE1", f"{wn1:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RANGE2", f"{wn2:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTION", f"{rp}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTIONUNIT", "RP")
    return cfg


# ============================================================
# H2O / TEMPERATURE CONTROL
# ============================================================

def scale_h2o_in_layers(cfg: str, scale: float) -> str:
    """
    Scale H2O VMR (index 2) in every ATMOSPHERE-LAYER-i line.
    """
    def _scale(match):
        tag = match.group(1)
        values = [v.strip() for v in match.group(2).split(",")]
        values[2] = f"{float(values[2]) * float(scale):.6e}"
        return f"<{tag}>{','.join(values)}"

    return re.sub(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", _scale, cfg)


def _set_h2o_abundance(cfg: str, h2o_scale: float) -> str:
    """
    Set the H2O scale factor in ATMOSPHERE-ABUN while leaving the other gases at 1.
    """
    gas_str = _get_tag(cfg, "ATMOSPHERE-GAS")
    if gas_str is None:
        raise RuntimeError("ATMOSPHERE-GAS tag not found in template.")
    n_gases = len(gas_str.split(","))
    abun_vals = ["1"] * n_gases
    abun_vals[0] = f"{float(h2o_scale):.8f}"
    return set_parameter(cfg, "ATMOSPHERE-ABUN", ",".join(abun_vals))


def apply_h2o_scaling(cfg: str, h2o_scale: float, mode: str = H2O_SCALING_MODE) -> str:
    mode = mode.lower().strip()
    if mode == "layers":
        return scale_h2o_in_layers(cfg, h2o_scale)
    if mode == "abun":
        return _set_h2o_abundance(cfg, h2o_scale)
    raise ValueError(f"Unknown H2O scaling mode: {mode!r}")


def shift_temperature_in_layers(cfg: str, target_surface_T: float) -> str:
    """
    Shift all layer temperatures by a constant offset so the first layer matches
    target_surface_T. This is only scientifically useful if PSG actually responds
    to temperature changes under the chosen atmosphere mode.
    """
    layers = re.findall(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", cfg)
    if not layers:
        raise RuntimeError("No ATMOSPHERE-LAYER entries found in template.")
    template_surface_T = float(layers[0][1].split(",")[1].strip())
    dT = float(target_surface_T) - template_surface_T

    def _shift(match):
        tag = match.group(1)
        values = [v.strip() for v in match.group(2).split(",")]
        values[1] = f"{float(values[1]) + dT:.6f}"
        return f"<{tag}>{','.join(values)}"

    return re.sub(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", _shift, cfg)


def build_psg_input(
    am: float,
    pwv_mm: float,
    surface_T: float,
    cfg_obj: PipelineConfig,
) -> str:
    """
    Build a PSG config for the requested (airmass, PWV, surface temperature).
    """
    if cfg_obj.template_pwv_mm is None or cfg_obj.template_pwv_mm <= 0:
        raise RuntimeError(
            "template_pwv_mm must be set to a positive empirically calibrated value."
        )

    cfg = load_psg_template(cfg_obj.template_path)

    if cfg_obj.use_option_b_user_atmosphere:
        cfg = force_user_atmosphere_mode(cfg)

    cfg = _apply_geometry_and_range(cfg, am, cfg_obj.wavelength_cfg_nm)

    h2o_scale = float(pwv_mm) / float(cfg_obj.template_pwv_mm)
    cfg = apply_h2o_scaling(cfg, h2o_scale, mode=cfg_obj.h2o_scaling_mode)

    # Temperature handling:
    # 1) set SURFACE-TEMPERATURE
    # 2) optionally shift layer temperatures so the atmosphere responds more
    #    consistently if PSG honors the layer values
    cfg = shift_temperature_in_layers(cfg, surface_T)
    cfg = set_parameter(cfg, "SURFACE-TEMPERATURE", f"{float(surface_T):.6f}")

    print(
        f"[PSG INPUT] am={am:.4f}  pwv={pwv_mm:.4f} mm  T={surface_T:.2f} K  "
        f"h2o_scale={h2o_scale:.8f}  mode={cfg_obj.h2o_scaling_mode}"
    )
    return cfg


# ============================================================
# PSG PARSING / API
# ============================================================

def parse_psg_header(lines: Sequence[str]) -> list[str]:
    """
    Extract column names from PSG comment lines.
    """
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
) -> Tuple[Optional[np.ndarray], list[str]]:
    """
    POST a PSG config and return (arr, col_names).
    """
    wait_s = 5

    for attempt in range(cfg_obj.psg_max_tries):
        print(f"\nAttempt {attempt+1}/{cfg_obj.psg_max_tries} — PSG (type={output_type})")

        try:
            response = requests.post(
                cfg_obj.psg_api,
                data={"file": psg_config_text, "type": output_type},
                timeout=cfg_obj.psg_timeout_s,
            )

            print("HTTP status:", response.status_code)

            if response.status_code != 200:
                if cfg_obj.debug_dump:
                    with open("psg_http_error.txt", "w") as f:
                        f.write(response.text)
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            text = response.text.strip()

            if cfg_obj.debug_dump:
                with open("psg_last_raw_response.txt", "w") as f:
                    f.write(text)

            busy_phrases = [
                "other api call is still running",
                "please let it finish",
                "please wait",
                "busy",
                "wait 10 minutes",
            ]
            if any(b in text.lower() for b in busy_phrases):
                print("PSG reports busy server. Waiting...")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            if len(text) < 100:
                print("PSG response suspiciously short.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            lines = text.splitlines()
            col_names = parse_psg_header(lines)
            data_lines = [ln for ln in lines if not ln.strip().startswith("#") and ln.strip()]

            if not data_lines:
                print("No numeric data lines found.")
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
                print("All PSG array values are non-finite.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            print(f"PSG returned {arr.shape[0]} points, {arr.shape[1]} columns")
            if col_names:
                print("Column names:", col_names)

            total_idx = find_column(col_names, PSG_TOTAL_LABEL)
            if total_idx is None:
                total_idx = 1
            print(
                "Transmission min/max:",
                np.nanmin(arr[:, total_idx]),
                np.nanmax(arr[:, total_idx]),
            )
            return arr, col_names

        except requests.exceptions.RequestException as e:
            print("PSG network error:", repr(e))
            time.sleep(wait_s)
            wait_s = min(wait_s * 1.5, 120)

    print("PSG failed after all retries.")
    return None, []


# ============================================================
# SCIENTIFIC DIAGNOSTICS
# ============================================================

def check_h2o_column(cfg_obj: PipelineConfig) -> Dict[str, Any]:
    """
    Identify which PSG column actually responds most strongly to H2O scaling.
    """
    print("\n[h2o_check] Running 2 PSG calls to identify the real H2O column.")
    cfg_full = load_psg_template(cfg_obj.template_path)
    if cfg_obj.use_option_b_user_atmosphere:
        cfg_full = force_user_atmosphere_mode(cfg_full)
    cfg_full = _apply_geometry_and_range(cfg_full, 1.0, cfg_obj.wavelength_cfg_nm)
    arr_full, col_names = run_psg(cfg_full, cfg_obj)

    cfg_zero = load_psg_template(cfg_obj.template_path)
    if cfg_obj.use_option_b_user_atmosphere:
        cfg_zero = force_user_atmosphere_mode(cfg_zero)
    cfg_zero = _apply_geometry_and_range(cfg_zero, 1.0, cfg_obj.wavelength_cfg_nm)
    cfg_zero = apply_h2o_scaling(cfg_zero, 0.0001, mode=cfg_obj.h2o_scaling_mode)
    arr_zero, _ = run_psg(cfg_zero, cfg_obj)

    if arr_full is None or arr_zero is None:
        raise RuntimeError("H2O column check failed: PSG did not return valid data.")

    lam_nm = 1e7 / arr_full[:, 0]
    band = (lam_nm >= 1850.0) & (lam_nm <= 1950.0)

    best_idx = None
    best_delta = -np.inf
    rows = []

    for i in range(1, arr_full.shape[1]):
        t_full = np.clip(arr_full[band, i], 1e-10, 1.0)
        t_zero = np.clip(arr_zero[band, i], 1e-10, 1.0)
        tau_full = np.median(-np.log(t_full))
        tau_zero = np.median(-np.log(t_zero))
        delta = abs(tau_full - tau_zero)
        name = col_names[i] if i < len(col_names) else f"col{i}"
        rows.append((i, name, tau_full, tau_zero, delta))
        if delta > best_delta:
            best_delta = delta
            best_idx = i

    for i, name, tau_full, tau_zero, delta in rows:
        marker = "  <-- strongest H2O response" if i == best_idx else ""
        print(
            f"[h2o_check] col={i:2d}  name={name:>10s}  "
            f"tau_full={tau_full:.6f}  tau_zero={tau_zero:.6f}  "
            f"delta={delta:.6f}{marker}"
        )

    best_name = col_names[best_idx] if (best_idx is not None and best_idx < len(col_names)) else f"col{best_idx}"
    return {"best_idx": best_idx, "best_name": best_name, "delta_tau": best_delta}


def calibrate_template_pwv(
    cfg_obj: PipelineConfig,
    test_pwv_mm: float = 5.0,
    rough_guess_mm: float = 20.0,
) -> float:
    """
    Empirically determine TEMPLATE_PWV_MM using Beer-Lambert scaling in a
    strong H2O band.
    """
    print("\n[calibration] Starting template PWV calibration — 3 PSG calls.")

    # Call 1: scale = 1.0
    cfg1 = load_psg_template(cfg_obj.template_path)
    if cfg_obj.use_option_b_user_atmosphere:
        cfg1 = force_user_atmosphere_mode(cfg1)
    cfg1 = _apply_geometry_and_range(cfg1, 1.0, cfg_obj.wavelength_cfg_nm)
    arr1, col_names1 = run_psg(cfg1, cfg_obj)
    if arr1 is None:
        raise RuntimeError("Calibration call 1 failed.")

    # Call 2: scale = 0.5
    cfg2 = load_psg_template(cfg_obj.template_path)
    if cfg_obj.use_option_b_user_atmosphere:
        cfg2 = force_user_atmosphere_mode(cfg2)
    cfg2 = _apply_geometry_and_range(cfg2, 1.0, cfg_obj.wavelength_cfg_nm)
    cfg2 = apply_h2o_scaling(cfg2, 0.5, mode=cfg_obj.h2o_scaling_mode)
    arr2, col_names2 = run_psg(cfg2, cfg_obj)
    if arr2 is None:
        raise RuntimeError("Calibration call 2 failed.")

    h2o_idx1 = find_column(col_names1, PSG_H2O_LABEL)
    h2o_idx2 = find_column(col_names2, PSG_H2O_LABEL)
    if h2o_idx1 is None:
        h2o_idx1 = 2
    if h2o_idx2 is None:
        h2o_idx2 = 2

    lam1 = 1e7 / arr1[:, 0]
    band = (lam1 >= 1850.0) & (lam1 <= 1950.0)

    h2o_full = np.clip(arr1[band, h2o_idx1], 1e-10, 1.0)
    h2o_half = np.clip(arr2[band, h2o_idx2], 1e-10, 1.0)

    tau_full = -np.log(h2o_full)
    tau_half = -np.log(h2o_half)

    valid = (tau_full > 0.01) & np.isfinite(tau_full) & np.isfinite(tau_half)
    if valid.sum() < 5:
        raise RuntimeError(
            "Calibration failed: too few pixels in the H2O band with meaningful optical depth."
        )

    ratio_half = np.median(tau_half[valid] / tau_full[valid])
    print(f"[calibration] tau(0.5)/tau(1.0) = {ratio_half:.4f} (expect ~0.5)")

    # Call 3: a known test scale
    scale_test = float(test_pwv_mm) / float(rough_guess_mm)

    cfg3 = load_psg_template(cfg_obj.template_path)
    if cfg_obj.use_option_b_user_atmosphere:
        cfg3 = force_user_atmosphere_mode(cfg3)
    cfg3 = _apply_geometry_and_range(cfg3, 1.0, cfg_obj.wavelength_cfg_nm)
    cfg3 = apply_h2o_scaling(cfg3, scale_test, mode=cfg_obj.h2o_scaling_mode)
    arr3, col_names3 = run_psg(cfg3, cfg_obj)
    if arr3 is None:
        raise RuntimeError("Calibration call 3 failed.")

    h2o_idx3 = find_column(col_names3, PSG_H2O_LABEL)
    if h2o_idx3 is None:
        h2o_idx3 = 2

    h2o_test = np.clip(arr3[band, h2o_idx3], 1e-10, 1.0)
    tau_test = -np.log(h2o_test)
    ratio_test = np.median(tau_test[valid] / tau_full[valid])

    template_pwv_mm = float(test_pwv_mm) / float(ratio_test)

    print(
        f"[calibration] scale_test={scale_test:.6f}  tau_ratio={ratio_test:.6f}  "
        f"=> TEMPLATE_PWV_MM = {template_pwv_mm:.4f} mm"
    )
    print(f"[calibration] Update PipelineConfig(template_pwv_mm={template_pwv_mm:.4f})")
    return template_pwv_mm


def validate_temperature_axis(
    interp: RegularGridInterpolator,
    lam_grid: np.ndarray,
    am: float,
    pwv_mm: float,
    t1: float,
    t2: float,
) -> Dict[str, Any]:
    """
    Compare two spectra at the same (airmass, PWV) but different temperatures.
    """
    spec1 = eval_interp(interp, am, pwv_mm, t1)
    spec2 = eval_interp(interp, am, pwv_mm, t2)
    diff = np.abs(spec1 - spec2)
    out = {
        "max_abs_diff": float(np.nanmax(diff)),
        "median_abs_diff": float(np.nanmedian(diff)),
        "t1": float(t1),
        "t2": float(t2),
        "am": float(am),
        "pwv_mm": float(pwv_mm),
    }
    print(
        f"[temperature_check] am={am:.3f}  pwv={pwv_mm:.3f}  "
        f"T1={t1:.2f}  T2={t2:.2f}  "
        f"max|dT|={out['max_abs_diff']:.6e}  median|dT|={out['median_abs_diff']:.6e}"
    )
    return out


# ============================================================
# GRID HASHING / HDF5
# ============================================================

def _grid_hash(
    wavelength_cfg_nm,
    airmass_grid: Sequence[float],
    pwv_grid: Sequence[float],
    temp_grid: Sequence[float],
    template_pwv_mm: float,
    use_option_b_user_atmosphere: bool,
    h2o_scaling_mode: str,
) -> str:
    key = str(
        (
            tuple(float(x) for x in wavelength_cfg_nm),
            tuple(sorted(float(x) for x in airmass_grid)),
            tuple(sorted(float(x) for x in pwv_grid)),
            tuple(sorted(float(x) for x in temp_grid)),
            float(template_pwv_mm),
            bool(use_option_b_user_atmosphere),
            str(h2o_scaling_mode),
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def grid_is_valid(
    h5_filename: str,
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    pwv_grid: Sequence[float],
    temp_grid: Sequence[float],
) -> bool:
    if not os.path.exists(h5_filename):
        return False

    expected = _grid_hash(
        cfg_obj.wavelength_cfg_nm,
        airmass_grid,
        pwv_grid,
        temp_grid,
        cfg_obj.template_pwv_mm,
        cfg_obj.use_option_b_user_atmosphere,
        cfg_obj.h2o_scaling_mode,
    )
    try:
        with h5py.File(h5_filename, "r") as hf:
            stored = hf.attrs.get("grid_hash", "")
        return stored == expected
    except Exception:
        return False


def load_hdf5_grid(h5_filename: str):
    with h5py.File(h5_filename, "r") as hf:
        spectra = hf["spectra"][:]
        airmasses = hf["airmasses"][:]
        pwv_values = hf["pwv_values"][:]
        temperatures = hf["temperatures"][:]
        wavelengths_nm = hf["wavelengths_nm"][:]
    return spectra, airmasses, pwv_values, temperatures, wavelengths_nm


def make_telluric_interp(h5_filename: str):
    spectra_grid, airmasses, pwv_values, temperatures, lam_grid = load_hdf5_grid(h5_filename)
    interp = RegularGridInterpolator(
        points=(airmasses, pwv_values, temperatures),
        values=spectra_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp, lam_grid


def eval_interp(interp: RegularGridInterpolator, am: float, pwv_mm: float, surface_T: float) -> np.ndarray:
    """
    Centralized wrapper for 3D interpolation.
    """
    return interp([[float(am), float(pwv_mm), float(surface_T)]])[0]


def build_grid_hdf5(
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    pwv_grid: Sequence[float],
    temp_grid: Sequence[float],
    force_rebuild: bool = False,
) -> str:
    """
    Build and cache the 4D stored array:
        spectra[i_am, j_pwv, k_T, i_lambda]
    """
    create_directory(cfg_obj.out_dir)

    airmass_grid = np.array(sorted(float(x) for x in airmass_grid), dtype=float)
    pwv_grid = np.array(sorted(float(x) for x in pwv_grid), dtype=float)
    temp_grid = np.array(sorted(float(x) for x in temp_grid), dtype=float)

    if (not force_rebuild) and grid_is_valid(
        cfg_obj.h5_filename, cfg_obj, airmass_grid, pwv_grid, temp_grid
    ):
        print(f"[grid] Grid is current — loading: {cfg_obj.h5_filename}")
        return cfg_obj.h5_filename

    lam_min_nm, lam_max_nm, rp = cfg_obj.wavelength_cfg_nm

    def cache_path(am: float, pwv_mm: float, surface_T: float, cfg_hash: str) -> str:
        return os.path.join(
            cfg_obj.out_dir,
            f"am{am:.3f}_pwv{pwv_mm:.3f}_T{surface_T:.1f}_"
            f"lam{lam_min_nm:.0f}-{lam_max_nm:.0f}_rp{int(rp)}_cfg{cfg_hash}.dat",
        )

    def load_cached(path: str) -> np.ndarray:
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Cache file is not valid 2-D numeric data.")
        return arr

    def compute_or_load(am: float, pwv_mm: float, surface_T: float) -> Tuple[np.ndarray, list[str], bool]:
        psg_in = build_psg_input(am, pwv_mm, surface_T, cfg_obj)
        cfg_hash = _stable_cfg_hash(psg_in)
        p = cache_path(am, pwv_mm, surface_T, cfg_hash)

        print(f"[grid] am={am:.3f}  pwv={pwv_mm:.3f}  T={surface_T:.1f}  hash={cfg_hash}")

        if (not force_rebuild) and os.path.exists(p) and os.path.getsize(p) > 0:
            print("  -> USING CACHE:", p)
            try:
                return load_cached(p), [], True
            except Exception as e:
                print("  -> cache read failed, recomputing:", repr(e))

        arr, col_names = run_psg(psg_in, cfg_obj)
        if arr is None:
            raise RuntimeError(f"PSG returned no valid data for am={am}, pwv={pwv_mm}, T={surface_T}")
        np.savetxt(p, arr)
        return arr, col_names, False

    # Reference wavelength grid from first point
    ref_arr, ref_cols, _ = compute_or_load(airmass_grid[0], pwv_grid[0], temp_grid[0])
    wave_idx = find_column(ref_cols, "Wave")
    if wave_idx is None:
        wave_idx = 0
    total_idx = find_column(ref_cols, PSG_TOTAL_LABEL)
    if total_idx is None:
        total_idx = 1

    lam_ref = 1e7 / ref_arr[:, wave_idx]
    order = np.argsort(lam_ref)
    lam_ref = lam_ref[order]
    nlam = len(lam_ref)

    grid_specs = np.full((len(airmass_grid), len(pwv_grid), len(temp_grid), nlam), np.nan, dtype=float)

    for i, am in enumerate(airmass_grid):
        for j, pwv_mm in enumerate(pwv_grid):
            for k, surface_T in enumerate(temp_grid):
                arr, cols, _ = compute_or_load(am, pwv_mm, surface_T)

                wave_idx = find_column(cols, "Wave")
                if wave_idx is None:
                    wave_idx = 0
                total_idx = find_column(cols, PSG_TOTAL_LABEL)
                if total_idx is None:
                    total_idx = 1

                lam_nm = 1e7 / arr[:, wave_idx]
                trans = arr[:, total_idx].astype(float)

                ord_ = np.argsort(lam_nm)
                lam_nm = lam_nm[ord_]
                trans = np.nan_to_num(trans[ord_], nan=0.0)

                # Force all models onto the same wavelength support
                grid_specs[i, j, k, :] = np.interp(lam_ref, lam_nm, trans, left=np.nan, right=np.nan)

    # --- Debug: inspect airmass dependence directly from grid_specs ---
    try:
        j_pwv = np.where(np.isclose(pwv_grid, 2.0))[0][0]
        k_temp = np.where(np.isclose(temp_grid, 280.0))[0][0]

        ref_idx = np.where(np.isclose(airmass_grid, 1.5))[0][0]
        ref_spec = grid_specs[ref_idx, j_pwv, k_temp, :]

        for am in [1.0, 1.2, 1.5, 2.0]:
            i_am = np.where(np.isclose(airmass_grid, am))[0][0]
            spec = grid_specs[i_am, j_pwv, k_temp, :]
            diff = spec - ref_spec

            print(
                f"[debug airmass] AM={am:.2f}  "
                f"min={np.nanmin(spec):.6e}  "
                f"max={np.nanmax(spec):.6e}  "
                f"mean={np.nanmean(spec):.6e}  "
                f"max|diff vs 1.5|={np.nanmax(np.abs(diff)):.6e}"
            )
    except Exception as e:
        print("[debug airmass] failed:", repr(e))

    gh = _grid_hash(
        cfg_obj.wavelength_cfg_nm,
        airmass_grid,
        pwv_grid,
        temp_grid,
        cfg_obj.template_pwv_mm,
        cfg_obj.use_option_b_user_atmosphere,
        cfg_obj.h2o_scaling_mode,
    )

    with h5py.File(cfg_obj.h5_filename, "w") as hf:
        hf.create_dataset("spectra", data=grid_specs)
        hf.create_dataset("airmasses", data=airmass_grid)
        hf.create_dataset("pwv_values", data=pwv_grid)
        hf.create_dataset("temperatures", data=temp_grid)
        hf.create_dataset("wavelengths_nm", data=lam_ref)
        hf.attrs["grid_hash"] = gh
        hf.attrs["template_pwv_mm"] = float(cfg_obj.template_pwv_mm)
        hf.attrs["use_option_b_user_atmosphere"] = bool(cfg_obj.use_option_b_user_atmosphere)
        hf.attrs["h2o_scaling_mode"] = str(cfg_obj.h2o_scaling_mode)

    print(f"[grid] Wrote {cfg_obj.h5_filename}  (hash={gh})")

    # --- Debug: inspect raw PSG outputs directly ---
    for am in [1.0, 1.2, 1.5, 2.0]:
        cfg_text = build_psg_input(am, 2.0, 280.0, cfg_obj)
        arr, cols = run_psg(cfg_text, cfg_obj)

        if arr is None:
            print(f"[raw PSG debug] AM={am:.2f} failed")
            continue

        total_idx = find_column(cols, "Total")
        if total_idx is None:
            total_idx = 1

        total = arr[:, total_idx]
        print(
            f"[raw PSG debug] AM={am:.2f}  "
            f"min={np.nanmin(total):.6e}  "
            f"max={np.nanmax(total):.6e}  "
            f"mean={np.nanmean(total):.6e}"
        )
    return cfg_obj.h5_filename



# ============================================================
# FORWARD MODEL
# ============================================================

def convolve_to_R(wave_nm: np.ndarray, flux: np.ndarray, R_inst: float) -> np.ndarray:
    """
    Convolve flux(wave) with a Gaussian LSF at resolving power R_inst in log-lambda space.
    """
    wave_nm = np.asarray(wave_nm, float)
    flux = np.asarray(flux, float)

    lnw = np.log(wave_nm)
    lnw_u = np.linspace(lnw.min(), lnw.max(), len(wave_nm))
    flux_u = np.interp(lnw_u, lnw, flux)

    sigma_lnw = (1.0 / R_inst) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dln = lnw_u[1] - lnw_u[0]
    sigma_pix = sigma_lnw / dln

    flux_conv_u = gaussian_filter1d(flux_u, sigma_pix, mode="nearest")
    return np.interp(lnw, lnw_u, flux_conv_u)


def forward_model_from_base(
    T_base: np.ndarray,
    dlam_nm: float,
    wave_obs: np.ndarray,
    lam_grid: np.ndarray,
    R_inst: float,
) -> np.ndarray:
    """
    Convolve, wavelength-shift, then resample onto the observed wavelength grid.
    """
    T_conv = convolve_to_R(lam_grid, T_base, R_inst)
    wave_shifted = np.asarray(wave_obs, float) - float(dlam_nm)
    return np.interp(wave_shifted, lam_grid, T_conv, left=np.nan, right=np.nan)


# ============================================================
# FITTING
# ============================================================

def _solve_linear_continuum(
    T_model: np.ndarray,
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    sigma_obs: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float]:
    lam = wave_obs[valid]
    lam0 = np.median(lam)
    w = 1.0 / sigma_obs[valid] ** 2

    A = np.vstack([T_model[valid], T_model[valid] * (lam - lam0)]).T
    Aw = A * np.sqrt(w[:, None])
    yw = flux_obs[valid] * np.sqrt(w)

    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    c0, c1 = coeffs
    return float(c0), float(c1)


def chi2_full(
    params: Sequence[float],
    interp: RegularGridInterpolator,
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    sigma_obs: np.ndarray,
    mask: np.ndarray,
    lam_grid: np.ndarray,
    R_inst: float,
) -> float:
    """
    Full 4-parameter model:
        params = [airmass, pwv_mm, surface_T, dlam_nm]
    """
    am, pwv_mm, surface_T, dlam_nm = params

    T_base = eval_interp(interp, am, pwv_mm, surface_T)
    T_model = forward_model_from_base(T_base, dlam_nm, wave_obs, lam_grid, R_inst)

    valid = (
        np.asarray(mask, bool)
        & np.isfinite(T_model)
        & np.isfinite(flux_obs)
        & np.isfinite(sigma_obs)
        & (sigma_obs > 0)
        & (T_model > 0.0)
    )

    if valid.sum() < 50:
        return 1e99

    c0, c1 = _solve_linear_continuum(T_model, wave_obs, flux_obs, sigma_obs, valid)

    lam = wave_obs[valid]
    lam0 = np.median(lam)
    model = (c0 + c1 * (lam - lam0)) * T_model[valid]
    resid = flux_obs[valid] - model
    w = 1.0 / sigma_obs[valid] ** 2
    return float(np.sum(resid ** 2 * w))


def fit_am_pwv_T_dlam(
    interp: RegularGridInterpolator,
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    sigma_obs: np.ndarray,
    mask: np.ndarray,
    lam_grid: np.ndarray,
    R_inst: float,
    x0: Sequence[float] = (1.5, 2.0, 280.0, 0.0),
    bounds: Sequence[Tuple[float, float]] = ((1.0, 2.5), (0.1, 10.0), (250.0, 320.0), (-0.05, 0.05)),
):
    """
    L-BFGS-B fit for (airmass, PWV, temperature, wavelength shift).
    """
    res = minimize(
        chi2_full,
        x0=np.asarray(x0, float),
        args=(interp, wave_obs, flux_obs, sigma_obs, mask, lam_grid, R_inst),
        bounds=bounds,
        method="L-BFGS-B",
    )
    return res.x, res.fun, res


def fit_am_pwv_dlam_fixed_T(
    interp: RegularGridInterpolator,
    T_fixed: float,
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    sigma_obs: np.ndarray,
    mask: np.ndarray,
    lam_grid: np.ndarray,
    R_inst: float,
    x0: Sequence[float] = (1.5, 2.0, 0.0),
    bounds: Sequence[Tuple[float, float]] = ((1.0, 2.5), (0.1, 10.0), (-0.05, 0.05)),
):
    """
    Reduced fit with temperature held fixed. Useful when the temperature axis is
    known to be inactive or poorly constrained.
    """
    def objective(p):
        am, pwv_mm, dlam_nm = p
        return chi2_full(
            [am, pwv_mm, float(T_fixed), dlam_nm],
            interp, wave_obs, flux_obs, sigma_obs, mask, lam_grid, R_inst
        )

    res = minimize(
        objective,
        x0=np.asarray(x0, float),
        bounds=bounds,
        method="L-BFGS-B",
    )
    return res.x, res.fun, res


def build_best_model_and_correct(
    best_params_4d: Sequence[float],
    interp: RegularGridInterpolator,
    wave_obs: np.ndarray,
    flux_obs: np.ndarray,
    sigma_obs: np.ndarray,
    mask: np.ndarray,
    lam_grid: np.ndarray,
    R_inst: float,
    telluric_floor: float = 1e-6,
) -> Dict[str, np.ndarray]:
    am, pwv_mm, surface_T, dlam_nm = best_params_4d

    T_base = eval_interp(interp, am, pwv_mm, surface_T)
    T_model = forward_model_from_base(T_base, dlam_nm, wave_obs, lam_grid, R_inst)

    valid = (
        np.asarray(mask, bool)
        & np.isfinite(T_model)
        & np.isfinite(flux_obs)
        & np.isfinite(sigma_obs)
        & (sigma_obs > 0)
        & (T_model > 0.0)
    )

    c0, c1 = _solve_linear_continuum(T_model, wave_obs, flux_obs, sigma_obs, valid)

    lam = wave_obs
    lam0 = np.median(wave_obs[valid])
    continuum = c0 + c1 * (lam - lam0)

    telluric_safe = np.clip(T_model, telluric_floor, np.inf)
    corrected_flux = flux_obs / telluric_safe
    corrected_uncertainty = sigma_obs / telluric_safe

    return {
        "T_base": T_base,
        "T_model": T_model,
        "continuum": continuum,
        "corrected_flux": corrected_flux,
        "corrected_uncertainty": corrected_uncertainty,
    }


# ============================================================
# HIGH-LEVEL DRIVER
# ============================================================

def prepare_pipeline(
    cfg_obj: PipelineConfig,
    airmass_grid: Sequence[float],
    pwv_grid: Sequence[float],
    temp_grid: Sequence[float],
    force_rebuild: bool = False,
):
    """
    Build/load the HDF5 grid and return (interp, lam_grid).
    """
    build_grid_hdf5(
        cfg_obj=cfg_obj,
        airmass_grid=airmass_grid,
        pwv_grid=pwv_grid,
        temp_grid=temp_grid,
        force_rebuild=force_rebuild,
    )
    interp, lam_grid = make_telluric_interp(cfg_obj.h5_filename)
    return interp, lam_grid


def plot_grid_spectra_vs_airmass(interp, lam_grid, pwv_mm=2.0, surface_T=280.0,
                                 airmass_list=(1.0, 1.2, 1.5, 2.0),
                                 title=None):
    plt.figure(figsize=(12, 6))
    for am in airmass_list:
        spec = eval_interp(interp, am, pwv_mm, surface_T)
        plt.plot(lam_grid, spec, label=f"AM={am:.2f}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(title or f"Transmission vs Airmass (PWV={pwv_mm:.2f} mm, T={surface_T:.1f} K)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grid_spectra_vs_pwv(interp, lam_grid, am=1.5, surface_T=280.0,
                             pwv_list=(0.5, 1.0, 2.0, 5.0, 8.0),
                             title=None):
    plt.figure(figsize=(12, 6))
    for pwv_mm in pwv_list:
        spec = eval_interp(interp, am, pwv_mm, surface_T)
        plt.plot(lam_grid, spec, label=f"PWV={pwv_mm:.2f} mm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(title or f"Transmission vs PWV (AM={am:.2f}, T={surface_T:.1f} K)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grid_spectra_vs_temperature(interp, lam_grid, am=1.5, pwv_mm=2.0,
                                     temp_list=(270.0, 280.0, 290.0),
                                     title=None):
    plt.figure(figsize=(12, 6))
    for surface_T in temp_list:
        spec = eval_interp(interp, am, pwv_mm, surface_T)
        plt.plot(lam_grid, spec, label=f"T={surface_T:.1f} K")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(title or f"Transmission vs Temperature (AM={am:.2f}, PWV={pwv_mm:.2f} mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_zoom_region(interp, lam_grid, am_list=None, pwv_list=None, temp_list=None,
                     fixed_am=1.5, fixed_pwv=2.0, fixed_T=280.0,
                     lam_min=1850.0, lam_max=1950.0,
                     mode="pwv"):
    mask = (lam_grid >= lam_min) & (lam_grid <= lam_max)
    lam = lam_grid[mask]

    plt.figure(figsize=(12, 6))

    if mode == "airmass":
        if am_list is None:
            am_list = [1.0, 1.2, 1.5, 2.0]
        for am in am_list:
            spec = eval_interp(interp, am, fixed_pwv, fixed_T)
            plt.plot(lam, spec[mask], label=f"AM={am:.2f}")

    elif mode == "pwv":
        if pwv_list is None:
            pwv_list = [0.5, 1.0, 2.0, 5.0, 8.0]
        for pwv_mm in pwv_list:
            spec = eval_interp(interp, fixed_am, pwv_mm, fixed_T)
            plt.plot(lam, spec[mask], label=f"PWV={pwv_mm:.2f} mm")

    elif mode == "temperature":
        if temp_list is None:
            temp_list = [270.0, 280.0, 290.0]
        for surface_T in temp_list:
            spec = eval_interp(interp, fixed_am, fixed_pwv, surface_T)
            plt.plot(lam, spec[mask], label=f"T={surface_T:.1f} K")

    else:
        raise ValueError("mode must be 'airmass', 'pwv', or 'temperature'")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(f"Zoomed Telluric Region: {lam_min:.0f}-{lam_max:.0f} nm ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_best_fit_model(wave_obs, flux_obs, sigma_obs, interp, lam_grid, best_params, R_inst):
    am, pwv_mm, surface_T, dlam_nm = best_params

    T_base = eval_interp(interp, am, pwv_mm, surface_T)
    T_model = forward_model_from_base(T_base, dlam_nm, wave_obs, lam_grid, R_inst)

    valid = np.isfinite(T_model) & np.isfinite(flux_obs) & np.isfinite(sigma_obs) & (sigma_obs > 0) & (T_model > 0)

    lam = wave_obs[valid]
    lam0 = np.median(lam)
    w = 1.0 / sigma_obs[valid] ** 2

    A = np.vstack([T_model[valid], T_model[valid] * (lam - lam0)]).T
    Aw = A * np.sqrt(w[:, None])
    yw = flux_obs[valid] * np.sqrt(w)

    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    c0, c1 = coeffs

    model_full = np.full_like(flux_obs, np.nan, dtype=float)
    model_full[valid] = (c0 + c1 * (lam - lam0)) * T_model[valid]

    plt.figure(figsize=(12, 6))
    plt.plot(wave_obs, flux_obs, label="Observed", linewidth=1)
    plt.plot(wave_obs, model_full, label="Best-fit model", linewidth=1.2)
    plt.fill_between(wave_obs, flux_obs - sigma_obs, flux_obs + sigma_obs,
                     alpha=0.25, label="1σ uncertainty")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux / Transmission")
    plt.title("Observed Spectrum vs Best-Fit Telluric Model")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_full_range(interp, lam_grid, mode="pwv",
                    am_values=(1.0, 1.2, 1.5, 2.0),
                    pwv_values=(0.5, 1.0, 2.0, 5.0, 8.0),
                    temp_values=(270.0, 280.0, 290.0),
                    fixed_am=1.5, fixed_pwv=2.0, fixed_T=280.0,
                    lam_min=1500.0, lam_max=2500.0,
                    ylim=None):
    mask = (lam_grid >= lam_min) & (lam_grid <= lam_max)
    lam = lam_grid[mask]

    plt.figure(figsize=(12, 6))

    if mode == "airmass":
        for am in am_values:
            spec = eval_interp(interp, am, fixed_pwv, fixed_T)
            plt.plot(lam, spec[mask], label=f"AM={am:.2f}")
        title = f"Transmission vs Airmass (PWV={fixed_pwv:.2f} mm, T={fixed_T:.1f} K)"

    elif mode == "pwv":
        for pwv in pwv_values:
            spec = eval_interp(interp, fixed_am, pwv, fixed_T)
            plt.plot(lam, spec[mask], label=f"PWV={pwv:.2f} mm")
        title = f"Transmission vs PWV (AM={fixed_am:.2f}, T={fixed_T:.1f} K)"

    elif mode == "temperature":
        for T in temp_values:
            spec = eval_interp(interp, fixed_am, fixed_pwv, T)
            plt.plot(lam, spec[mask], label=f"T={T:.1f} K")
        title = f"Transmission vs Temperature (AM={fixed_am:.2f}, PWV={fixed_pwv:.2f} mm)"

    else:
        raise ValueError("mode must be 'airmass', 'pwv', or 'temperature'")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_full_range_difference(interp, lam_grid, mode="pwv",
                               ref_am=1.5, ref_pwv=2.0, ref_T=280.0,
                               am_values=(1.0, 1.2, 1.5, 2.0),
                               pwv_values=(0.5, 1.0, 2.0, 5.0, 8.0),
                               temp_values=(270.0, 280.0, 290.0),
                               lam_min=1500.0, lam_max=2500.0):
    mask = (lam_grid >= lam_min) & (lam_grid <= lam_max)
    lam = lam_grid[mask]
    ref = eval_interp(interp, ref_am, ref_pwv, ref_T)[mask]

    plt.figure(figsize=(12, 6))

    if mode == "airmass":
        for am in am_values:
            spec = eval_interp(interp, am, ref_pwv, ref_T)[mask]
            plt.plot(lam, spec - ref, label=f"AM={am:.2f}")
        title = f"ΔTransmission vs Airmass (ref: AM={ref_am:.2f}, PWV={ref_pwv:.2f}, T={ref_T:.1f})"

    elif mode == "pwv":
        for pwv in pwv_values:
            spec = eval_interp(interp, ref_am, pwv, ref_T)[mask]
            plt.plot(lam, spec - ref, label=f"PWV={pwv:.2f} mm")
        title = f"ΔTransmission vs PWV (ref: AM={ref_am:.2f}, PWV={ref_pwv:.2f}, T={ref_T:.1f})"

    elif mode == "temperature":
        for T in temp_values:
            spec = eval_interp(interp, ref_am, ref_pwv, T)[mask]
            plt.plot(lam, spec - ref, label=f"T={T:.1f} K")
        title = f"ΔTransmission vs Temperature (ref: AM={ref_am:.2f}, PWV={ref_pwv:.2f}, T={ref_T:.1f})"

    else:
        raise ValueError("mode must be 'airmass', 'pwv', or 'temperature'")

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("ΔTransmission")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def debug_geometry_tags(cfg_text):
    for line in cfg_text.splitlines():
        if any(tag in line for tag in [
            "<GEOMETRY>",
            "<GEOMETRY-OBS-ANGLE>",
            "<GEOMETRY-USER-PARAM>",
            "<OBJECT-DATE>",
            "<ATMOSPHERE-NGAS>",
            "<ATMOSPHERE-TYPE>",
        ]):
            print(line)

def plot_am_pwv_chi2_heatmap(
    interp,
    wave_obs,
    flux_obs,
    sigma_obs,
    mask,
    lam_grid,
    R_inst,
    T_fixed=280.0,
    dlam_fixed=0.0,
    am_bounds=(1.0, 2.5),
    pwv_bounds=(0.1, 10.0),
    n_am=120,
    n_pwv=120,
    show_best=True,
):
    """
    Heat map of chi-square as a function of (airmass, PWV),
    with temperature and wavelength shift held fixed.

    A diagonal valley in this plot indicates degeneracy/tradeoff:
    higher airmass can mimic higher PWV and vice versa.
    """
    am_vals = np.linspace(am_bounds[0], am_bounds[1], n_am)
    pwv_vals = np.linspace(pwv_bounds[0], pwv_bounds[1], n_pwv)

    chi2_map = np.full((len(pwv_vals), len(am_vals)), np.nan, dtype=float)

    for j, pwv_mm in enumerate(pwv_vals):
        for i, am in enumerate(am_vals):
            chi2_map[j, i] = chi2_full(
                [am, pwv_mm, float(T_fixed), float(dlam_fixed)],
                interp,
                wave_obs,
                flux_obs,
                sigma_obs,
                mask,
                lam_grid,
                R_inst,
            )

    # normalize so the minimum is 0 for easier interpretation
    finite = np.isfinite(chi2_map) & (chi2_map < 1e98)
    chi2_min = np.nanmin(chi2_map[finite])

    dchi2_map = np.full_like(chi2_map, np.nan)
    dchi2_map[finite] = chi2_map[finite] - chi2_min

    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        dchi2_map,
        origin="lower",
        aspect="auto",
        extent=[am_vals[0], am_vals[-1], pwv_vals[0], pwv_vals[-1]],
    )
    plt.colorbar(im, label=r"$\Delta \chi^2$")

    plt.xlabel("Airmass")
    plt.ylabel("PWV (mm)")
    plt.title(
        f"Chi-square Heat Map in Airmass–PWV Space\n"
        f"(T fixed = {T_fixed:.1f} K, dlam fixed = {dlam_fixed:.4f} nm)"
    )

    if show_best:
        min_idx = np.unravel_index(np.nanargmin(chi2_map), chi2_map.shape)
        best_pwv = pwv_vals[min_idx[0]]
        best_am = am_vals[min_idx[1]]

        plt.plot(best_am, best_pwv, "wx", markersize=10, markeredgewidth=2,
                 label=f"Best grid point: AM={best_am:.3f}, PWV={best_pwv:.3f}")
        plt.legend()

        print("\n[heatmap] Best point on mesh:")
        print(f"  airmass = {best_am:.6f}")
        print(f"  pwv_mm  = {best_pwv:.6f}")
        print(f"  chi2    = {chi2_map[min_idx]:.6f}")

    plt.tight_layout()
    plt.show()

    return am_vals, pwv_vals, chi2_map

def plot_am_pwv_valid_count_map(
    interp,
    wave_obs,
    flux_obs,
    sigma_obs,
    mask,
    lam_grid,
    R_inst,
    T_fixed=280.0,
    dlam_fixed=0.0,
    am_bounds=(1.0, 2.5),
    pwv_bounds=(0.1, 10.0),
    n_am=120,
    n_pwv=120,
):
    am_vals = np.linspace(am_bounds[0], am_bounds[1], n_am)
    pwv_vals = np.linspace(pwv_bounds[0], pwv_bounds[1], n_pwv)

    nvalid_map = np.full((len(pwv_vals), len(am_vals)), np.nan)

    for j, pwv_mm in enumerate(pwv_vals):
        for i, am in enumerate(am_vals):
            T_base = eval_interp(interp, am, pwv_mm, T_fixed)
            T_model = forward_model_from_base(T_base, dlam_fixed, wave_obs, lam_grid, R_inst)

            valid = (
                np.asarray(mask, bool)
                & np.isfinite(T_model)
                & np.isfinite(flux_obs)
                & np.isfinite(sigma_obs)
                & (sigma_obs > 0)
                & (T_model > 0.0)
            )

            nvalid_map[j, i] = np.sum(valid)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        nvalid_map,
        origin="lower",
        aspect="auto",
        extent=[am_vals[0], am_vals[-1], pwv_vals[0], pwv_vals[-1]],
    )
    plt.colorbar(im, label="Number of valid pixels")
    plt.xlabel("Airmass")
    plt.ylabel("PWV (mm)")
    plt.title("Validity Map in Airmass–PWV Space")
    plt.tight_layout()
    plt.show()

    return am_vals, pwv_vals, nvalid_map

def plot_am_dlam_chi2_heatmap(
    interp,
    wave_obs,
    flux_obs,
    sigma_obs,
    mask,
    lam_grid,
    R_inst,
    pwv_fixed=2.0,
    T_fixed=280.0,
    am_bounds=(1.0, 2.5),
    dlam_bounds=(-0.05, 0.05),
    n_am=140,
    n_dlam=140,
    mask_penalty=True,
):
    """
    Heat map of chi-square as a function of (airmass, dlam),
    with PWV and temperature held fixed.
    """
    am_vals = np.linspace(am_bounds[0], am_bounds[1], n_am)
    dlam_vals = np.linspace(dlam_bounds[0], dlam_bounds[1], n_dlam)

    chi2_map = np.full((len(dlam_vals), len(am_vals)), np.nan, dtype=float)

    for j, dlam_nm in enumerate(dlam_vals):
        for i, am in enumerate(am_vals):
            chi2_map[j, i] = chi2_full(
                [am, float(pwv_fixed), float(T_fixed), dlam_nm],
                interp,
                wave_obs,
                flux_obs,
                sigma_obs,
                mask,
                lam_grid,
                R_inst,
            )

    if mask_penalty:
        finite = np.isfinite(chi2_map) & (chi2_map < 1e98)
    else:
        finite = np.isfinite(chi2_map)

    if not np.any(finite):
        raise RuntimeError("No finite chi-square values found in the heat map.")

    chi2_min = np.nanmin(chi2_map[finite])
    dchi2_map = np.full_like(chi2_map, np.nan)
    dchi2_map[finite] = chi2_map[finite] - chi2_min

    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        dchi2_map,
        origin="lower",
        aspect="auto",
        extent=[am_vals[0], am_vals[-1], dlam_vals[0], dlam_vals[-1]],
    )
    plt.colorbar(im, label=r"$\Delta \chi^2$")
    plt.xlabel("Airmass")
    plt.ylabel(r"$d\lambda$ (nm)")
    plt.title(
        f"Chi-square Heat Map in Airmass–dlam Space\n"
        f"(PWV fixed = {pwv_fixed:.2f} mm, T fixed = {T_fixed:.1f} K)"
    )

    min_idx = np.unravel_index(np.nanargmin(np.where(finite, chi2_map, np.nan)), chi2_map.shape)
    best_dlam = dlam_vals[min_idx[0]]
    best_am = am_vals[min_idx[1]]

    plt.plot(best_am, best_dlam, "wx", markersize=10, markeredgewidth=2,
             label=f"Best grid point: AM={best_am:.3f}, dlam={best_dlam:.4f} nm")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n[am-dlam heatmap] Best point on mesh:")
    print(f"  airmass = {best_am:.6f}")
    print(f"  dlam_nm = {best_dlam:.6f}")
    print(f"  chi2    = {chi2_map[min_idx]:.6f}")

    return am_vals, dlam_vals, chi2_map

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # User choices
    # -----------------------------
    cfg = PipelineConfig(
        template_pwv_mm=20.0935,
        wavelength_cfg_nm=(1500.0, 2500.0, 50000.0),
        h5_filename="telluric_grid.h5",
    )

    airmass_grid = [1.0, 1.2, 1.5, 2.0]
    pwv_grid = [0.5, 1.0, 2.0, 5.0, 8.0]
    temp_grid = [270.0, 280.0, 290.0]

    # -----------------------------
    # Build or load grid
    # -----------------------------
    interp, lam_grid = prepare_pipeline(
        cfg_obj=cfg,
        airmass_grid=airmass_grid,
        pwv_grid=pwv_grid,
        temp_grid=temp_grid,
        force_rebuild=True,
    )

    # -----------------------------
    # Optional sanity checks
    # -----------------------------
    temp_report = validate_temperature_axis(
        interp=interp,
        lam_grid=lam_grid,
        am=1.5,
        pwv_mm=2.0,
        t1=270.0,
        t2=290.0,
    )
    if temp_report["max_abs_diff"] == 0.0:
        print(
            "\nWARNING: temperature axis appears inactive. "
            "For scientifically safer production use, consider collapsing temp_grid "
            "to a single value and fitting only (airmass, PWV, dlam)."
        )

    # -----------------------------
    # Synthetic-data smoke test
    # -----------------------------
    R_inst = 8000.0
    true_params = [1.5, 2.0, 280.0, 0.015]  # am, pwv_mm, T, dlam_nm

    wave_obs = lam_grid.copy()
    T_base_true = eval_interp(interp, true_params[0], true_params[1], true_params[2])
    flux_true = forward_model_from_base(T_base_true, true_params[3], wave_obs, lam_grid, R_inst)

    sigma_obs = 0.02 * np.ones_like(flux_true)
    rng = np.random.default_rng(12345)
    flux_obs = flux_true + sigma_obs * rng.normal(size=len(flux_true))
    mask = (
        np.isfinite(wave_obs)
        & (wave_obs >= 1850.0)
        & (wave_obs <= 1950.0)
    )

    best_params, chi2_val, res = fit_am_pwv_T_dlam(
        interp=interp,
        wave_obs=wave_obs,
        flux_obs=flux_obs,
        sigma_obs=sigma_obs,
        mask=mask,
        lam_grid=lam_grid,
        R_inst=R_inst,
        x0=(1.5, 2.0, 280.0, 0.0),
        bounds=((1.0, 2.5), (0.1, 10.0), (250.0, 320.0), (-0.05, 0.05)),
    )

    print("\nTrue parameters:     ", true_params)
    print("Recovered parameters:", best_params)
    print("Best chi2:           ", chi2_val)

    products = build_best_model_and_correct(
        best_params_4d=best_params,
        interp=interp,
        wave_obs=wave_obs,
        flux_obs=flux_obs,
        sigma_obs=sigma_obs,
        mask=mask,
        lam_grid=lam_grid,
        R_inst=R_inst,
    )

    print("\nPipeline smoke test complete.")

    interp, lam_grid = prepare_pipeline(
        cfg_obj=cfg,
        airmass_grid=airmass_grid,
        pwv_grid=pwv_grid,
        temp_grid=temp_grid,
        force_rebuild=False,
    )

    plot_grid_spectra_vs_airmass(interp, lam_grid, pwv_mm=2.0, surface_T=280.0)
    plot_grid_spectra_vs_pwv(interp, lam_grid, am=1.5, surface_T=280.0)
    plot_grid_spectra_vs_temperature(interp, lam_grid, am=1.5, pwv_mm=2.0)

    plot_zoom_region(interp, lam_grid, mode="pwv", lam_min=1850, lam_max=1950)
    plot_zoom_region(interp, lam_grid, mode="temperature", lam_min=1850, lam_max=1950)
    plot_zoom_region(interp, lam_grid, mode="airmass", lam_min=1850, lam_max=1950)
    plot_best_fit_model(wave_obs, flux_obs, sigma_obs, interp, lam_grid, best_params, R_inst)

    plot_full_range(interp, lam_grid, mode="airmass")
    plot_full_range(interp, lam_grid, mode="pwv")
    plot_full_range(interp, lam_grid, mode="temperature")

    plot_full_range_difference(interp, lam_grid, mode="airmass")
    plot_full_range_difference(interp, lam_grid, mode="pwv")
    plot_full_range_difference(interp, lam_grid, mode="temperature")

    cfg1 = build_psg_input(1.0, 2.0, 280.0, cfg)
    cfg2 = build_psg_input(2.0, 2.0, 280.0, cfg)

    print("=== AM = 1.0 ===")
    debug_geometry_tags(cfg1)

    print("=== AM = 2.0 ===")
    debug_geometry_tags(cfg2)

    # --- H2O comparison: low vs high PWV ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for pwv in [0.5, 2.0, 5.0, 8.0]:
        spec = eval_interp(interp, 1.5, pwv, 280.0)
        axes[0].plot(lam_grid, spec, label=f"PWV={pwv:.1f} mm")

    axes[0].set_ylabel("Total Transmission")
    axes[0].set_title("Total Telluric Transmission vs PWV (AM=1.5, T=280K)")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    # H2O column only — make a fresh PSG call at low and high PWV
    for pwv, color in [(0.5, "blue"), (8.0, "red")]:
        cfg_text = build_psg_input(1.5, pwv, 280.0, cfg)
        arr, cols = run_psg(cfg_text, cfg)
        if arr is None:
            continue
        lam_nm = 1e7 / arr[:, 0]
        order  = np.argsort(lam_nm)
        lam_nm = lam_nm[order]
        h2o_idx = find_column(cols, "H2O")
        if h2o_idx is None:
            h2o_idx = 2
        h2o = arr[order, h2o_idx]
        axes[1].plot(lam_nm, h2o, color=color, label=f"H2O only, PWV={pwv:.1f} mm")

    axes[1].set_ylabel("H2O Transmission")
    axes[1].set_title("H2O-only Transmission: Low vs High PWV")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(1500, 2500)

    plt.tight_layout()
    plt.show()

    am_vals, pwv_vals, chi2_map = plot_am_pwv_chi2_heatmap(
        interp=interp,
        wave_obs=wave_obs,
        flux_obs=flux_obs,
        sigma_obs=sigma_obs,
        mask=mask,
        lam_grid=lam_grid,
        R_inst=R_inst,
        T_fixed=280.0,
        dlam_fixed=0.0,
        am_bounds=(1.0, 2.5),
        pwv_bounds=(0.1, 10.0),
        n_am=120,
        n_pwv=120,
    )

    am_vals, dlam_vals, chi2_map = plot_am_dlam_chi2_heatmap(
        interp=interp,
        wave_obs=wave_obs,
        flux_obs=flux_obs,
        sigma_obs=sigma_obs,
        mask=mask,
        lam_grid=lam_grid,
        R_inst=R_inst,
        pwv_fixed=2.0,
        T_fixed=280.0,
        am_bounds   = (0.8, 1.8),
        dlam_bounds = (-0.10, 0.05),
        n_am=140,
        n_dlam=140,
    )
