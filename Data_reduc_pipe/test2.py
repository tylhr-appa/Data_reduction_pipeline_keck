import os, re, io, time, requests
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import shutil
import hashlib


# ============================================================
# PSG / GRID BUILDING
# ============================================================

PSG_API       = "https://psg.gsfc.nasa.gov/api.php"
TEMPLATE_PATH = "earth_cfg.txt"
OUT_DIR       = "telluric_grid"

# Molecular column names as PSG labels them in the header.
# Used by find_column() to resolve indices robustly.
PSG_TOTAL_LABEL = "Total"
PSG_H2O_LABEL   = "H2O"


# if os.path.exists("telluric_grid.h5"):
#     os.remove("telluric_grid.h5")
# shutil.rmtree("telluric_grid", ignore_errors=True)

def create_directory(path):
    os.makedirs(path, exist_ok=True)


def load_psg_template(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PSG template not found: {path}. "
            "Download manually from PSG interface."
        )
    with open(path, "r") as f:
        cfg = f.read()
    if "<ATMOSPHERE-LAYER" not in cfg:
        raise RuntimeError("Invalid PSG template file.")
    return cfg


def _get_tag(cfg: str, tag: str):
    m = re.search(rf"^<{re.escape(str(tag))}>\s*(.*)\s*$", cfg, flags=re.M)
    return None if m is None else m.group(1).strip()


def set_parameter(cfg, tag, value):
    """Replace or append a <TAG> line in the config."""
    val = str(value)
    pat = rf"(^<{re.escape(tag)}>\s*).*$"
    new, n = re.subn(pat, rf"\g<1>{val}", cfg, flags=re.MULTILINE)
    if n == 0:
        if not cfg.endswith("\n"):
            cfg += "\n"
        new = cfg + f"<{tag}> {val}\n"
    return new


def am_to_zenith_deg(am: float) -> float:
    """Convert airmass to zenith angle in degrees using sec(z) = AM."""
    am = max(float(am), 1.0)
    return float(np.degrees(np.arccos(np.clip(1.0 / am, 0.0, 1.0))))


def compute_template_pwv(cfg):
    """
    Compute a PWV proxy (mm) by pressure-integrating H2O VMR across layers.
    Uses finite differences of pressure for layer thickness rather than
    absolute pressure, avoiding the dimensional error in the original formula.
    """
    layers = re.findall(r"<ATMOSPHERE-LAYER-\d+>(.+)", cfg)
    if not layers:
        raise RuntimeError("No ATMOSPHERE-LAYER entries found.")

    pressures = []
    h2o_vmrs  = []
    for layer in layers:
        vals = [v.strip() for v in layer.split(",")]
        pressures.append(float(vals[0]))
        h2o_vmrs.append(float(vals[2]))

    pressures = np.array(pressures)   # atm, decreasing with altitude
    h2o_vmrs  = np.array(h2o_vmrs)

    # Layer thickness in pressure coords (always positive)
    # Use central differences for interior layers, one-sided at boundaries
    dp = np.abs(np.gradient(pressures))   # atm

    # PWV in mm: integral of rho_wv dz, converted via hydrostatic balance
    # dp/dz = -rho_air * g  →  dz = -dp / (rho_air * g)
    # rho_wv = VMR * (M_H2O/M_air) * rho_air
    # → PWV = sum(VMR * dp) * M_H2O/M_air / g * 1000  (mm)
    M_ratio   = 18.015 / 28.964          # M_H2O / M_dry_air
    atm_to_pa = 101325.0
    g         = 9.81

    pwv_m = np.sum(h2o_vmrs * dp) * M_ratio * atm_to_pa / g

    for tag, values in re.findall(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", cfg)[:3]:
        print(f"<{tag}> {values}")

    return pwv_m * 1000.0   # → mm


def scale_h2o_in_layers(cfg: str, scale: float) -> str:
    """Scale H2O abundance (index 2) in every ATMOSPHERE-LAYER-X entry."""
    def _scale(match):
        tag    = match.group(1)
        values = [v.strip() for v in match.group(2).split(",")]
        values[2] = f"{float(values[2]) * scale:.6e}"
        return f"<{tag}>{','.join(values)}"

    return re.sub(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", _scale, cfg)


def shift_temperature_in_layers(cfg: str, target_surface_T: float) -> str:
    """
    Shift every layer temperature by a constant offset so the first layer
    matches target_surface_T.
    """
    layers = re.findall(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", cfg)
    if not layers:
        raise ValueError("No ATMOSPHERE-LAYER entries found.")

    template_surface_T = float(layers[0][1].split(",")[1].strip())
    dT = float(target_surface_T) - template_surface_T

    def _shift(match):
        tag    = match.group(1)
        values = [v.strip() for v in match.group(2).split(",")]
        values[1] = f"{float(values[1]) + dT:.6f}"
        return f"<{tag}>{','.join(values)}"

    return re.sub(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", _shift, cfg)


def build_psg_input(am, pwv, T, wavelength_cfg_nm):
    """
    Build a PSG config string for the requested (airmass, PWV, temperature).

    PWV scaling: template PWV is computed from the clean config before any
    layer modification, then every H2O layer abundance is multiplied by
    (requested_pwv / template_pwv) so the grid axis has real physical meaning.
    """
    lam_min_nm, lam_max_nm, rp = wavelength_cfg_nm
    cfg = load_psg_template(TEMPLATE_PATH)

    # --- geometry ---
    zen = am_to_zenith_deg(am)
    cfg = set_parameter(cfg, "GEOMETRY",           "Lookingup")
    cfg = set_parameter(cfg, "GEOMETRY-OBS-ANGLE", f"{zen:.6f}")

    # --- spectral range ---
    wn1 = 1e7 / lam_max_nm
    wn2 = 1e7 / lam_min_nm
    cfg = set_parameter(cfg, "GENERATOR-RANGEUNIT",      "cm-1")
    cfg = set_parameter(cfg, "GENERATOR-RANGE1",         f"{wn1:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RANGE2",         f"{wn2:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTION",     f"{rp}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTIONUNIT", "RP")

    # --- PWV scaling ---
    # compute_template_pwv is called on the CLEAN template (before any layer
    # modifications) so temperature substitution cannot corrupt the read.
    template_pwv = compute_template_pwv(cfg)
    if template_pwv <= 0:
        raise RuntimeError("Template PWV is zero — check atmosphere layers.")
    scale = pwv / template_pwv
    print(f"[PWV] template={template_pwv:.4f} mm  requested={pwv:.4f} mm  "
          f"scale={scale:.4f}")
    cfg = scale_h2o_in_layers(cfg, scale)

    # --- temperature (applied after PWV scaling) ---
    cfg = shift_temperature_in_layers(cfg, T)
    cfg = set_parameter(cfg, "SURFACE-TEMPERATURE", f"{float(T):.6f}")

    return cfg

# --- ONE-TIME PWV CALIBRATION ---
# Strategy: run PSG at two H2O scales, measure transmission ratio in the
# 1850-1950 nm H2O band, interpolate to find what scale = 1.0 corresponds to
# in physical mm.

def calibrate_template_pwv(wavelength_cfg_nm, test_pwv_mm=5.0):
    """
    Calibrate TEMPLATE_PWV_MM by running PSG at two explicit scales and
    using Beer-Lambert scaling to infer the unscaled column.

    test_pwv_mm: a PWV value you're confident is well within the grid range.
    """
    # Run at scale=1.0 (raw template, no H2O modification)
    cfg_s1 = load_psg_template(TEMPLATE_PATH)
    cfg_s1 = _apply_geometry_and_range(cfg_s1, 1.0, wavelength_cfg_nm)
    # Do NOT scale H2O — this is the raw template
    arr1, _ = run_psg(cfg_s1)

    # Run at scale=0.5 (half the template H2O)
    cfg_s2 = load_psg_template(TEMPLATE_PATH)
    cfg_s2 = _apply_geometry_and_range(cfg_s2, 1.0, wavelength_cfg_nm)
    cfg_s2 = scale_h2o_in_layers(cfg_s2, 0.5)
    arr2, _ = run_psg(cfg_s2)

    if arr1 is None or arr2 is None:
        raise RuntimeError("Calibration PSG calls failed.")

    # Extract H2O column (index 2) in the 1850-1950 nm band
    wn1  = arr1[:, 0];  lam1 = 1e7 / wn1
    wn2  = arr2[:, 0];  lam2 = 1e7 / wn2
    band = (lam1 >= 1850) & (lam1 <= 1950)

    # Mean log-transmission ratio gives the column ratio
    # T = exp(-tau)  →  tau = -ln(T)
    # tau_scaled / tau_unscaled = scale
    # So: ln(T_half) / ln(T_full) = 0.5  (should be, under Beer-Lambert)
    # We use this to verify linearity, then read off the column directly.

    h2o_full = arr1[band, 2]   # H2O column at scale=1
    h2o_half = arr2[band, 2]   # H2O column at scale=0.5

    # Compute optical depth ratio
    tau_full = -np.log(np.clip(h2o_full, 1e-10, 1.0))
    tau_half = -np.log(np.clip(h2o_half, 1e-10, 1.0))

    valid = (tau_full > 0.01) & np.isfinite(tau_full) & np.isfinite(tau_half)
    ratio = np.median(tau_half[valid] / tau_full[valid])
    print(f"[calibration] tau_half/tau_full median ratio = {ratio:.4f}  "
          f"(expect ~0.5 if Beer-Lambert holds)")

    # Now: we know scale=0.5 gives tau_half, scale=1.0 gives tau_full.
    # We want to know what physical PWV corresponds to scale=1.0.
    # Run a third call at a physically known PWV using a rough first guess,
    # then interpolate.
    rough_guess_mm = 20.0   # adjust if ratio above is far from 0.5
    scale_for_test = test_pwv_mm / rough_guess_mm
    cfg_s3 = load_psg_template(TEMPLATE_PATH)
    cfg_s3 = _apply_geometry_and_range(cfg_s3, 1.0, wavelength_cfg_nm)
    cfg_s3 = scale_h2o_in_layers(cfg_s3, scale_for_test)
    arr3, _ = run_psg(cfg_s3)

    h2o_test = arr3[band, 2]
    tau_test  = -np.log(np.clip(h2o_test, 1e-10, 1.0))
    ratio_test = np.median(tau_test[valid] / tau_full[valid])
    # ratio_test = scale_for_test  →  template_pwv = test_pwv_mm / ratio_test
    template_pwv = test_pwv_mm / ratio_test
    print(f"[calibration] Estimated TEMPLATE_PWV_MM = {template_pwv:.4f} mm")
    return template_pwv

# ============================================================
# PSG RESPONSE PARSING
# ============================================================

def parse_psg_header(lines):
    """
    Extract column names from PSG comment lines.

    PSG writes one or more '#'-prefixed lines before the data.  The column
    header line contains recognisable gas names (H2O, CO2, Total, …).
    Returns a list of strings, one per data column, or [] if not found.

    Example header line:
        # Wave/freq  Total  H2O  CO2  O3  N2O  CO  CH4  O2  Rayleigh  CIA
    """
    gas_hints = {"H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2",
                 "Total", "Rayleigh", "CIA", "Wave", "Wave/freq"}

    for ln in lines:
        stripped = ln.strip()
        if not stripped.startswith("#"):
            continue
        parts = stripped.lstrip("#").split()
        # Require at least 3 recognisable tokens to treat as the column header
        if sum(1 for p in parts if p in gas_hints) >= 2:
            return parts

    return []


def find_column(col_names, label):
    """
    Return the integer index of *label* in col_names, or None if absent.
    Matching is case-insensitive and also handles 'Wave/freq' → 'Wave'.
    """
    label_lo = label.lower()
    for i, name in enumerate(col_names):
        if name.lower() == label_lo:
            return i
        # Allow "Wave/freq" to match "wave"
        if label_lo == "wave" and name.lower().startswith("wave"):
            return i
    return None


def run_psg(psg_config_text, timeout=180, max_tries=10, debug_dump=True,
            output_type="trn"):
    """
    Send full PSG config to the API and return:
        arr       : 2-D float array, shape (N_pixels, N_columns)
        col_names : list of column-name strings parsed from the '#' header
                    (empty list if the header could not be parsed)

    output_type : PSG 'type' parameter.  Use "trn" for total-only transmission,
                  or the same "trn" value — PSG automatically includes per-molecule
                  columns when the config requests them via ATMOSPHERE-GAS.
                  Pass "rad" for radiance output if needed elsewhere.

    Backoff: wait_s accumulates correctly across retries (was being reset
    inside the loop in the original code).
    """
    wait_s = 5

    for attempt in range(max_tries):
        print(f"\nAttempt {attempt+1}/{max_tries} sending config to PSG "
              f"(type={output_type})")

        try:
            response = requests.post(
                PSG_API,
                data={"file": psg_config_text, "type": output_type},
                timeout=timeout,
            )

            print("HTTP status:", response.status_code)

            if response.status_code != 200:
                print("Non-200 HTTP response.")
                if debug_dump:
                    with open("psg_http_error.txt", "w") as f:
                        f.write(response.text)
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            text = response.text.strip()

            if debug_dump:
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

            lines      = text.splitlines()
            col_names  = parse_psg_header(lines)

            data_lines = [
                ln for ln in lines
                if not ln.strip().startswith("#") and ln.strip()
            ]

            if not data_lines:
                print("No numeric data lines found.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            try:
                arr = np.genfromtxt(io.StringIO("\n".join(data_lines)),
                                    dtype=float)
            except Exception as e:
                print("Parsing error:", e)
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
                print("Unexpected array shape:", getattr(arr, "shape", None))
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            if not np.isfinite(arr).any():
                print("All values non-finite.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            print(f"PSG returned {arr.shape[0]} points, {arr.shape[1]} columns")
            if col_names:
                print("Column names:", col_names)
            else:
                print("Warning: could not parse column names from PSG header.")

            # Resolve total-transmission column for the diagnostic print
            total_idx = find_column(col_names, PSG_TOTAL_LABEL)
            if total_idx is not None and total_idx < arr.shape[1]:
                print("Transmission min/max:",
                      np.nanmin(arr[:, total_idx]),
                      np.nanmax(arr[:, total_idx]))
            else:
                print("Transmission min/max (col 1 fallback):",
                      np.nanmin(arr[:, 1]), np.nanmax(arr[:, 1]))

            return arr, col_names

        except requests.exceptions.RequestException as e:
            print("Network error:", e)
            time.sleep(wait_s)
            wait_s = min(wait_s * 1.5, 120)

    print("PSG failed after all retries.")
    return None, []


# ============================================================
# GRID BUILDING + HDF5 I/O
# ============================================================

def _stable_cfg_hash(cfg_text: str) -> str:
    lines      = [ln.rstrip() for ln in cfg_text.replace("\r\n", "\n").split("\n")]
    lines      = [ln for ln in lines if ln.strip()]
    normalized = "\n".join(lines) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def _grid_hash(wavelength_cfg, airmass_grid, pwv_grid, temp_grid):
    """Stable hash of the grid axes + wavelength config."""
    key = str((
        tuple(wavelength_cfg),
        tuple(sorted(airmass_grid)),
        tuple(sorted(pwv_grid)),
        tuple(sorted(temp_grid)),
    ))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def grid_is_valid(h5_filename, wavelength_cfg, airmass_grid,
                  pwv_grid, temp_grid):
    """
    Return True iff the HDF5 file exists and its stored grid_hash matches
    the hash of the current grid parameters.
    """
    if not os.path.exists(h5_filename):
        return False
    expected = _grid_hash(wavelength_cfg, airmass_grid, pwv_grid, temp_grid)
    try:
        with h5py.File(h5_filename, "r") as hf:
            stored = hf.attrs.get("grid_hash", "")
        return stored == expected
    except Exception:
        return False


def write_grid_to_hdf5(h5_filename, airmasses, pwv_values, temperatures,
                       wavelength_cfg_nm):
    create_directory(OUT_DIR)
    lam_min, lam_max, rp = wavelength_cfg_nm

    def cache_path(am, pwv, T, cfg_hash):
        return os.path.join(
            OUT_DIR,
            f"am{am:.3f}_pwv{pwv:.3f}_T{float(T):.1f}_"
            f"lam{lam_min:.0f}-{lam_max:.0f}_rp{int(rp)}_"
            f"cfg{cfg_hash}.dat",
        )

    def load_cached(path):
        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Cache file is not 2-column numeric data.")
        return arr

    def compute_or_load(am, pwv, T, force_recompute=False):
        psg_in   = build_psg_input(am, pwv, T, wavelength_cfg_nm)
        cfg_hash = _stable_cfg_hash(psg_in)
        p        = cache_path(am, pwv, T, cfg_hash)

        print(f"[compute_or_load] am={am:.2f} pwv={pwv:.2f} T={T} "
              f"hash={cfg_hash}")

        if (not force_recompute) and os.path.exists(p) \
                and os.path.getsize(p) > 0:
            print("  -> USING CACHE:", p)
            try:
                return load_cached(p), True
            except Exception as e:
                print("  -> cache read failed, recomputing:", repr(e))

        arr, _ = run_psg(psg_in)   # col_names not needed for grid building
        if arr is None:
            raise RuntimeError(
                f"PSG returned no valid data for am={am}, pwv={pwv}, T={T}"
            )

        np.savetxt(p, arr)
        return arr, False

    # Build wavelength reference from the first grid point
    ref_arr, _ = compute_or_load(airmasses[0], pwv_values[0], temperatures[0])
    wn_ref  = ref_arr[:, 0].astype(float)
    lam_ref = 1e7 / wn_ref
    o       = np.argsort(lam_ref)
    lam_ref = lam_ref[o]
    nlam    = len(lam_ref)

    # Column index for total transmission — use col 1 (always present)
    # Individual molecular columns are NOT stored in the interpolation grid;
    # they are fetched on-demand via a separate PSG call in __main__.
    TOTAL_COL = 1

    grid_specs = np.full(
        (len(airmasses), len(pwv_values), len(temperatures), nlam), np.nan
    )

    for i, am in enumerate(airmasses):
        for j, pwv in enumerate(pwv_values):
            for k, T in enumerate(temperatures):
                arr, _  = compute_or_load(am, pwv, T)
                wn      = arr[:, 0].astype(float)
                trans   = arr[:, TOTAL_COL].astype(float)
                lam_nm  = 1e7 / wn
                ord_    = np.argsort(lam_nm)
                lam_nm  = lam_nm[ord_]
                trans   = np.nan_to_num(trans[ord_], nan=0.0)
                grid_specs[i, j, k, :] = np.interp(
                    lam_ref, lam_nm, trans, left=np.nan, right=np.nan
                )

    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset("spectra",        data=grid_specs)
        hf.create_dataset("airmasses",      data=np.array(airmasses,    dtype=float))
        hf.create_dataset("pwv_values",     data=np.array(pwv_values,   dtype=float))
        hf.create_dataset("temperatures",   data=np.array(temperatures, dtype=float))
        hf.create_dataset("wavelengths_nm", data=lam_ref)
        # Store grid hash so we can detect stale grids without re-running PSG
        hf.attrs["grid_hash"] = _grid_hash(
            wavelength_cfg_nm, airmasses, pwv_values, temperatures
        )

    print(f"Grid written to {h5_filename}  "
          f"(hash={hf.attrs['grid_hash'] if False else _grid_hash(wavelength_cfg_nm, airmasses, pwv_values, temperatures)})")


# ============================================================
# LOAD GRID + INTERPOLATOR
# ============================================================

def load_hdf5_grid(h5_filename):
    with h5py.File(h5_filename, "r") as hf:
        spectra      = hf["spectra"][:]
        airmasses    = hf["airmasses"][:]
        pwv_values   = hf["pwv_values"][:]
        temperatures = hf["temperatures"][:]
        wavelengths  = hf["wavelengths_nm"][:]
    return spectra, airmasses, pwv_values, temperatures, wavelengths


def make_telluric_interp(h5_filename):
    spectra_grid, airmasses, pwv_values, temperatures, lam_grid = \
        load_hdf5_grid(h5_filename)

    interp = RegularGridInterpolator(
        points=(airmasses, pwv_values, temperatures),
        values=spectra_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp, lam_grid


def eval_interp(interp, am, pwv, T):
    """
    Convenience wrapper — always use this instead of calling interp directly.
    Returns a 1-D transmission spectrum.
    Centralises the correct [[am, pwv, T]] call shape so callers cannot
    accidentally use the broken scalar-tuple form.
    """
    return interp([[float(am), float(pwv), float(T)]])[0]


# ============================================================
# FORWARD MODEL
# ============================================================

def convolve_to_R(wave_nm, flux, R_inst):
    """
    Convolve flux(wave) with a Gaussian LSF at resolving power R_inst.
    Works in log-lambda space so sigma is constant in pixels.
    """
    wave_nm = np.asarray(wave_nm, float)
    flux    = np.asarray(flux,    float)

    lnw   = np.log(wave_nm)
    lnw_u = np.linspace(lnw.min(), lnw.max(), len(wave_nm))
    flux_u = np.interp(lnw_u, lnw, flux)

    sigma_lnw = (1.0 / R_inst) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dln       = lnw_u[1] - lnw_u[0]
    sigma_pix = sigma_lnw / dln

    flux_conv_u = gaussian_filter1d(flux_u, sigma_pix, mode="nearest")
    return np.interp(lnw, lnw_u, flux_conv_u)


def forward_model_from_base(T_base, dlam_nm, wave_obs, lam_grid, R_inst):
    """
    Apply convolution and wavelength shift, then resample to the observed grid.
    Continuum scaling is handled analytically in chi2_full.
    """
    T_conv       = convolve_to_R(lam_grid, T_base, R_inst)
    wave_shifted = wave_obs - dlam_nm
    return np.interp(wave_shifted, lam_grid, T_conv, left=np.nan, right=np.nan)


# ============================================================
# FITTING / CHI2
# ============================================================

def chi2_full(params, T_fixed, interp, wave_obs, flux_obs, sigma_obs,
              mask, lam_grid, R_inst):

    am, pwv, dlam = params

    T_base  = eval_interp(interp, am, pwv, T_fixed)
    T_model = forward_model_from_base(T_base, dlam, wave_obs, lam_grid, R_inst)

    valid = (
        mask
        & np.isfinite(T_model)
        & np.isfinite(flux_obs)
        & np.isfinite(sigma_obs)
        & (sigma_obs > 0)
        & (T_model > 0.5)
    )

    if valid.sum() < 50:
        return 1e99

    w    = 1.0 / sigma_obs[valid] ** 2
    lam  = wave_obs[valid]
    lam0 = np.median(lam)

    # Analytic continuum: (c0 + c1*(lambda - lambda0)) * T
    A  = np.vstack([T_model[valid], T_model[valid] * (lam - lam0)]).T
    Aw = A * np.sqrt(w[:, None])
    yw = flux_obs[valid] * np.sqrt(w)

    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    c0, c1 = coeffs

    model_full = (c0 + c1 * (lam - lam0)) * T_model[valid]
    resid      = flux_obs[valid] - model_full
    return float(np.sum(resid ** 2 * w))


def chi2_heatmap_am_pwv(am_scan, pwv_scan, T_fixed,
                        wave_obs, flux_obs, sigma_obs, mask,
                        telluric_interp, lam_grid, R_inst):
    """
    Heatmap uses the same analytic continuum fitting as chi2_full so the two
    χ² surfaces are directly comparable and the heatmap minimum actually
    corresponds to the optimizer's minimum.
    """
    chi2_map = np.full((len(am_scan), len(pwv_scan)), np.nan)

    for i, am in enumerate(am_scan):
        for j, pwv in enumerate(pwv_scan):
            chi2_map[i, j] = chi2_full(
                [am, pwv, 0.0],
                T_fixed,
                telluric_interp,
                wave_obs,
                flux_obs,
                sigma_obs,
                mask,
                lam_grid,
                R_inst,
            )

    return chi2_map


def fit_am_pwv_dlam(interp, T_fixed, wave_obs, flux_obs, sigma_obs,
                    mask, lam_grid, R_inst):

    x0     = [1.5, 2.0, 0.0]
    bounds = [(1.0, 2.5), (0.1, 10.0), (-0.05, 0.05)]

    res = minimize(
        chi2_full,
        x0,
        args=(T_fixed, interp, wave_obs, flux_obs, sigma_obs,
              mask, lam_grid, R_inst),
        bounds=bounds,
        method="L-BFGS-B",
    )

    return res.x, res.fun, res


# ============================================================
# PLOTTING
# ============================================================

def plot_full_spectrum(wave_nm, flux_obs, sigma_obs, telluric_interp,
                       lam_grid, best_params, T_fixed, R_inst):

    am_best, pwv_best, dlam_best = best_params

    T_base  = eval_interp(telluric_interp, am_best, pwv_best, T_fixed)
    T_model = forward_model_from_base(T_base, dlam_best, wave_nm, lam_grid, R_inst)

    valid = np.isfinite(T_model)
    lam   = wave_nm[valid]
    lam0  = np.median(lam)
    w     = 1.0 / sigma_obs[valid] ** 2

    A  = np.vstack([T_model[valid], T_model[valid] * (lam - lam0)]).T
    Aw = A * np.sqrt(w[:, None])
    yw = flux_obs[valid] * np.sqrt(w)

    coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    c0, c1 = coeffs

    model_full        = np.full_like(T_model, np.nan)
    model_full[valid] = (c0 + c1 * (lam - lam0)) * T_model[valid]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(wave_nm, flux_obs,   color="black", lw=0.8, alpha=0.9,
             label="Observed",       zorder=3)
    ax1.plot(wave_nm, model_full,  color="red",   lw=1.2,
             label="Best-fit model", zorder=2)
    ax1.fill_between(wave_nm,
                     flux_obs - sigma_obs, flux_obs + sigma_obs,
                     color="grey", alpha=0.3, label="1σ uncertainty")
    ax1.set_ylabel("Transmission")
    ax1.legend()
    ax1.set_title("Full Telluric Spectrum")

    ax2.plot(wave_nm, flux_obs - model_full, color="blue", lw=0.8)
    ax2.axhline(0, color="black", linestyle="--", lw=0.8)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Residual")

    plt.tight_layout()
    plt.show()


def plot_chi2_heatmap_am_pwv(am_scan, pwv_scan, T_fixed,
                             wave_obs, flux_obs, sigma_obs, mask,
                             telluric_interp, lam_grid, R_inst,
                             best_params=None, true_params=None,
                             show_contours=True):

    chi2_map = chi2_heatmap_am_pwv(
        am_scan, pwv_scan, T_fixed,
        wave_obs, flux_obs, sigma_obs, mask,
        telluric_interp, lam_grid, R_inst,
    )
    chi2_rel = chi2_map - np.nanmin(chi2_map)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        chi2_rel.T, origin="lower", aspect="auto",
        extent=[am_scan.min(), am_scan.max(),
                pwv_scan.min(), pwv_scan.max()],
        cmap="viridis",
    )
    plt.colorbar(im, label="Δχ²")

    if show_contours:
        plt.contour(
            am_scan, pwv_scan, chi2_rel.T,
            levels=[2.3, 6.17, 11.8],
            colors="white", linewidths=1.2,
        )

    if best_params is not None:
        plt.scatter(best_params[0], best_params[1],
                    color="red", marker="x", s=100, label="Best fit")
    if true_params is not None:
        plt.scatter(true_params[0], true_params[1],
                    color="white", marker="o", s=60, label="True")

    plt.xlabel("Airmass")
    plt.ylabel("PWV (mm)")
    plt.title("χ² Heatmap (Airmass vs PWV)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_h2o_comparison(lam_nm, h2o_raw, total_raw=None, R_inst=25000,
                        h2o_col_name="H2O"):
    """
    Plot raw and convolved H2O transmission alongside total transmission.

    h2o_col_name : label used in the plot title (from the parsed PSG header,
                   so it reflects what PSG actually gave us).
    """
    h2o_conv = convolve_to_R(lam_nm, h2o_raw, R_inst)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(lam_nm, h2o_raw, lw=1)
    axes[0].set_ylabel("Transmittance")
    axes[0].set_title(f"Raw PSG {h2o_col_name}")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(lam_nm, h2o_conv, lw=1)
    axes[1].set_ylabel("Transmittance")
    axes[1].set_title(f"Convolved {h2o_col_name} at R = {R_inst}")
    axes[1].set_ylim(0, 1.05)

    axes[2].set_xlabel("Wavelength [nm]")
    axes[2].set_ylabel("Transmittance")
    axes[2].set_ylim(0, 1.05)
    if total_raw is not None:
        total_conv = convolve_to_R(lam_nm, total_raw, R_inst)
        axes[2].plot(lam_nm, total_conv, lw=1)
        axes[2].set_title(f"Convolved Total Transmission at R = {R_inst}")
    else:
        axes[2].plot(lam_nm, h2o_conv, lw=1)
        axes[2].set_title("Convolved H2O (repeated)")

    plt.tight_layout()
    plt.show()


def print_column_diagnostics(arr_raw, col_names):
    """
    Print min/max/std/mean for every column in arr_raw, with names where
    available.  Useful for verifying which column contains H2O absorption.
    """
    n_cols = arr_raw.shape[1]
    print(f"\narr_raw shape: {arr_raw.shape}")
    header = f"{'Col':>4}  {'Name':>12}  {'Min':>12}  {'Max':>12}  "  \
             f"{'Std':>12}  {'Mean':>12}"
    print(header)
    print("-" * len(header))
    for i in range(n_cols):
        col  = arr_raw[:, i]
        name = col_names[i] if i < len(col_names) else "?"
        print(f"{i:>4}  {name:>12}  {col.min():>12.6f}  {col.max():>12.6f}  "
              f"{col.std():>12.6f}  {col.mean():>12.6f}")


# ============================================================
# MAIN WORKFLOW
# ============================================================

if __name__ == "__main__":

    # --- ONE-TIME PWV CALIBRATION ---
    # Send the raw template to PSG with no H2O scaling and check what it reports.
    cfg_raw = load_psg_template(TEMPLATE_PATH)

    # Set geometry and spectral range but do NOT call scale_h2o_in_layers
    zen = am_to_zenith_deg(1.0)
    cfg_raw = set_parameter(cfg_raw, "GEOMETRY",           "Lookingup")
    cfg_raw = set_parameter(cfg_raw, "GEOMETRY-OBS-ANGLE", f"{zen:.6f}")
    wn1 = 1e7 / 2500
    wn2 = 1e7 / 1500
    cfg_raw = set_parameter(cfg_raw, "GENERATOR-RANGEUNIT",      "cm-1")
    cfg_raw = set_parameter(cfg_raw, "GENERATOR-RANGE1",         f"{wn1:.6f}")
    cfg_raw = set_parameter(cfg_raw, "GENERATOR-RANGE2",         f"{wn2:.6f}")
    cfg_raw = set_parameter(cfg_raw, "GENERATOR-RESOLUTION",     "300000")
    cfg_raw = set_parameter(cfg_raw, "GENERATOR-RESOLUTIONUNIT", "RP")

    arr_cal, _ = run_psg(cfg_raw)
    if arr_cal is not None:
        print(f"\nCalibration PSG call succeeded — {arr_cal.shape[0]} points")
        print("Save this transmission spectrum to compare against scaled versions.")
        np.savetxt("psg_template_unscaled.dat", arr_cal)
        print("Written to psg_template_unscaled.dat")

    # H2O: 1.234e+02 mm-pr  [Precipitable mm]

    # --- Grid definition ---
    # R_psg  : native PSG sampling resolution — sets how finely PSG generates
    #          the model spectrum before any instrumental convolution.
    #          Should be >> R_inst so there are many pixels per resolution element.
    # R_inst : actual LIGER instrument resolving power used in convolve_to_R
    #          and all fitting functions.
    R_psg  = 300000   # PSG native grid sampling
    R_inst = 8000     # Keck LIGER instrument resolution

    wavelength_cfg = [1500, 2500, R_psg]
    airmass_grid   = [1.0, 1.2, 1.5, 2.0]
    pwv_grid       = [0.5, 1.0, 2.0, 3.0, 5.0]
    temp_grid      = [270, 280, 290]

    H5_FILE = "telluric_grid.h5"

    # --- Smart cache invalidation ---
    # Only wipe and rebuild if grid parameters have actually changed.
    # This replaces the previous unconditional delete which made caching useless.
    if grid_is_valid(H5_FILE, wavelength_cfg, airmass_grid, pwv_grid, temp_grid):
        print("Grid parameters unchanged — loading existing grid.")
    else:
        print("Grid parameters changed or grid missing — rebuilding.")
        if os.path.exists(H5_FILE):
            os.remove(H5_FILE)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        write_grid_to_hdf5(
            H5_FILE, airmass_grid, pwv_grid, temp_grid, wavelength_cfg
        )

    # --- Load interpolator ---
    telluric_interp, lam_grid = make_telluric_interp(H5_FILE)

    # --- Fixed / true parameters ---
    am        = 1.5
    pwv       = 2.0
    T_fixed   = 280
    dlam_true = 0.015

    # --- Evaluate grid at true parameters ---
    T_base = eval_interp(telluric_interp, am, pwv, T_fixed)

    # Temperature sensitivity sanity check
    spec1 = eval_interp(telluric_interp, 1.5, 2.0, 270.0)
    spec2 = eval_interp(telluric_interp, 1.5, 2.0, 290.0)
    print("Max |ΔT| from temperature change:", np.max(np.abs(spec1 - spec2)))

    # Zoom plot around spectrum centre
    centre = np.median(lam_grid)
    m      = (lam_grid > centre - 2) & (lam_grid < centre + 2)
    lam_zoom = lam_grid[m]
    T_zoom   = T_base[m]

    if lam_zoom.size == 0:
        raise RuntimeError("Zoom window empty — adjust selection.")

    p     = np.polyfit(lam_zoom, T_zoom, 1)
    trend = np.polyval(p, lam_zoom)

    plt.figure(figsize=(8, 4))
    plt.plot(lam_zoom, T_zoom - trend)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Detrended Transmission")
    plt.title("Zoomed spectrum (slope removed)")
    plt.show()

    # --- Generate synthetic observation ---
    wave_obs  = lam_grid.copy()
    flux_true = forward_model_from_base(T_base, dlam_true, wave_obs, lam_grid, R_inst)

    print("Mean transmission:", np.nanmean(flux_true))
    print("Max  transmission:", np.nanmax(flux_true))
    print("NaN count:",         np.isnan(flux_true).sum())

    sigma_obs = 0.02 * np.ones_like(flux_true)
    noise     = sigma_obs * np.random.randn(len(flux_true))
    flux_obs  = flux_true + noise

    mask = (
        np.isfinite(wave_obs)
        & (wave_obs >= 1500)
        & (wave_obs <= 2500)
    )

    # --- Fit ---
    best_params, chi2_val, res = fit_am_pwv_dlam(
        telluric_interp, T_fixed, wave_obs, flux_obs, sigma_obs,
        mask, lam_grid, R_inst,
    )

    print("\nTrue:      am={}, pwv={}, dlam={}".format(am, pwv, dlam_true))
    print("Recovered: am={:.4f}, pwv={:.4f}, dlam={:.6f}".format(*best_params))

    # --- χ² heatmap ---
    am_scan  = np.linspace(1.0, 2.5, 40)
    pwv_scan = np.linspace(0.1, 5.0, 40)

    plot_chi2_heatmap_am_pwv(
        am_scan, pwv_scan, T_fixed,
        wave_obs, flux_obs, sigma_obs, mask,
        telluric_interp, lam_grid, R_inst,
        best_params=(best_params[0], best_params[1]),
        true_params=(am, pwv),
    )

    # --- Full spectrum plot ---
    plot_full_spectrum(
        wave_obs, flux_obs, sigma_obs,
        telluric_interp, lam_grid,
        best_params, T_fixed, R_inst,
    )

    # --- Molecular component plot ---
    # A fresh PSG call is made here to retrieve per-molecule columns.
    # The column index for H2O is resolved from the parsed header rather than
    # being hardcoded, so it is robust to changes in which gases are active.
    psg_cfg = build_psg_input(am, pwv, T_fixed, wavelength_cfg)
    arr_raw, col_names = run_psg(psg_cfg)

    if arr_raw is None:
        raise RuntimeError("PSG failed for component plot.")

    # Sort into wavelength order
    wn     = arr_raw[:, 0]
    lam_nm = 1e7 / wn
    order  = np.argsort(lam_nm)
    lam_nm  = lam_nm[order]
    arr_raw = arr_raw[order, :]

    # Always print column diagnostics so unexpected layouts are visible
    print_column_diagnostics(arr_raw, col_names)

    # Resolve columns by name; fall back to known-good indices from your
    # diagnostic run (total=1, H2O=8) if the header could not be parsed.
    total_idx = find_column(col_names, PSG_TOTAL_LABEL)
    h2o_idx   = find_column(col_names, PSG_H2O_LABEL)

    if total_idx is None:
        print("Warning: 'Total' column not found by name — falling back to col 1.")
        total_idx = 1
    if h2o_idx is None:
        print("Warning: 'H2O' column not found by name — falling back to col 8.")
        h2o_idx = 8

    print(f"Using total_idx={total_idx}, h2o_idx={h2o_idx}")

    total_raw = arr_raw[:, total_idx] if total_idx < arr_raw.shape[1] else None
    h2o_raw   = arr_raw[:, h2o_idx]  if h2o_idx  < arr_raw.shape[1] else None

    if h2o_raw is not None:
        h2o_col_name = col_names[h2o_idx] if h2o_idx < len(col_names) else "H2O"
        plot_h2o_comparison(
            lam_nm, h2o_raw,
            total_raw=total_raw,
            R_inst=25000,
            h2o_col_name=h2o_col_name,
        )
    else:
        print("H2O column not present in PSG output — skipping component plot.")

    print("\nDone.")
    print(f"λ range: {lam_grid.min():.1f} – {lam_grid.max():.1f} nm")
    print(f"N points: {len(lam_grid)}")
    print(f"Median Δλ: {np.median(np.diff(lam_grid)):.4f} nm")
    print(f"Resolution element at 2000 nm: {2000/R_inst:.4f} nm")