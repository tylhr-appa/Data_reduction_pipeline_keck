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
def _find_template(name="earth_cfg.txt"):
    """Locate the PSG template, checking the script dir and Data_reduc_pipe/."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        os.path.join("Data_reduc_pipe", name),
        name,
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return name  # fall through to let load_psg_template raise a clear error

TEMPLATE_PATH = _find_template()
OUT_DIR       = "telluric_grid"

# Force PSG into the user-controlled atmosphere mode (Option B).
USE_OPTION_B_USER_ATMOSPHERE = True

# Which H2O control path to use once Option B is active.
# Set to either "layers" or "abun".
H2O_SCALING_MODE = "layers"


# Molecular column names as PSG labels them in the header.
# Used by find_column() to resolve indices robustly.

PSG_TOTAL_LABEL = "Total"
PSG_H2O_LABEL   = "H2O"

# PWV of the unscaled template in mm — the physical water vapour column that
# PSG models when ATMOSPHERE-ABUN for H2O = 1 (i.e. the template as-is).
#
# This is calibrated empirically by calibrate_template_pwv() which makes 3
# PSG calls and uses Beer-Lambert tau ratios to back out the column.  It cannot
# be read directly from the template file because the layer VMR format is
# ambiguous (units, conventions, and number of molecules vary by template).
#
# Workflow:
#   1. Leave TEMPLATE_PWV_MM = None and run the script once.
#   2. Read the printed value, e.g. "TEMPLATE_PWV_MM = 18.3421".
#   3. Paste it here and re-run.  The calibration block will be skipped.
#   4. Re-run calibration any time earth_cfg.txt is replaced.
TEMPLATE_PWV_MM = 20.0935   # <- set this after first calibration run



# ============================================================
# CONFIG HELPERS
# ============================================================

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
    """Return the value of a <TAG> line, or None if not present."""
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

def force_user_atmosphere_mode(cfg: str) -> str:
    """
    Force PSG to use a user-controlled atmosphere rather than the
    reanalysis-driven Earth atmosphere path.

    This is the Option B fix:
      - set a non-MERRA date
      - explicitly define the gas count
      - explicitly define the atmosphere source list
    """
    cfg = set_parameter(cfg, "OBJECT-DATE", "1970/01/01 00:00")
    cfg = set_parameter(cfg, "ATMOSPHERE-NGAS", "8")
    cfg = set_parameter(
        cfg,
        "ATMOSPHERE-TYPE",
        "HIT[1],HIT[2],HIT[3],HIT[4],HIT[5],HIT[6],HIT[7],HIT[22]"
    )
    return cfg


def am_to_zenith_deg(am: float) -> float:
    """Convert airmass to zenith angle in degrees using sec(z) = AM."""
    am = max(float(am), 1.0)
    return float(np.degrees(np.arccos(np.clip(1.0 / am, 0.0, 1.0))))


def _apply_geometry_and_range(cfg, am, wavelength_cfg_nm):
    """
    Set geometry (airmass -> zenith angle) and spectral range on a PSG config.
    Shared by build_psg_input and calibrate_template_pwv.
    """
    lam_min_nm, lam_max_nm, rp = wavelength_cfg_nm
    zen = am_to_zenith_deg(am)
    cfg = set_parameter(cfg, "GEOMETRY",           "Lookingup")
    cfg = set_parameter(cfg, "GEOMETRY-OBS-ANGLE", f"{zen:.6f}")
    wn1 = 1e7 / lam_max_nm
    wn2 = 1e7 / lam_min_nm
    cfg = set_parameter(cfg, "GENERATOR-RANGEUNIT",      "cm-1")
    cfg = set_parameter(cfg, "GENERATOR-RANGE1",         f"{wn1:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RANGE2",         f"{wn2:.6f}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTION",     f"{rp}")
    cfg = set_parameter(cfg, "GENERATOR-RESOLUTIONUNIT", "RP")
    return cfg


def scale_h2o_in_layers(cfg: str, scale: float) -> str:
    """
    Scale the H2O VMR (index 2) in every ATMOSPHERE-LAYER-X entry by scale.

    Patches layer VMRs directly.  PSG may ignore these in Equilibrium mode
    (see check_abun_sensitivity to determine which mechanism PSG respects).
    """
    def _scale(match):
        tag    = match.group(1)
        values = [v.strip() for v in match.group(2).split(",")]
        values[2] = f"{float(values[2]) * scale:.6e}"
        return f"<{tag}>{','.join(values)}"
    return re.sub(r"<(ATMOSPHERE-LAYER-\d+)>(.+)", _scale, cfg)


def _set_h2o_abundance(cfg, h2o_scale):
    """
    Set the H2O abundance scale factor in ATMOSPHERE-ABUN, leaving all
    other gases at 1.  This is the correct mechanism when PSG is in
    Equilibrium mode and ignores layer VMRs for gas abundances.

    H2O is always first in ATMOSPHERE-GAS for this template:
        H2O,CO2,O3,N2O,CO,CH4,O2,N2
    """
    gas_str = _get_tag(cfg, "ATMOSPHERE-GAS")
    if gas_str is None:
        raise RuntimeError("ATMOSPHERE-GAS tag not found in template.")
    n_gases      = len(gas_str.split(","))
    abun_vals    = ["1"] * n_gases
    abun_vals[0] = f"{h2o_scale:.8f}"
    return set_parameter(cfg, "ATMOSPHERE-ABUN", ",".join(abun_vals))


def apply_h2o_scaling(cfg: str, h2o_scale: float) -> str:
    """
    Apply H2O scaling using the selected control path.

    H2O_SCALING_MODE:
      - 'layers' : patch ATMOSPHERE-LAYER-X entries directly
      - 'abun'   : scale ATMOSPHERE-ABUN entry for H2O
    """
    mode = H2O_SCALING_MODE.lower().strip()

    if mode == "layers":
        return scale_h2o_in_layers(cfg, h2o_scale)
    elif mode == "abun":
        return _set_h2o_abundance(cfg, h2o_scale)
    else:
        raise ValueError(
            f"Unknown H2O_SCALING_MODE={H2O_SCALING_MODE!r}. "
            "Use 'layers' or 'abun'."
        )
    
def debug_layer_scaling_preview(scale=0.1, n_layers=3):
    """
    Print the first few ATMOSPHERE-LAYER-X lines before and after scaling
    so you can confirm that the H2O field is actually changing.
    """
    cfg0 = load_psg_template(TEMPLATE_PATH)
    cfg1 = scale_h2o_in_layers(cfg0, scale)

    pat = r"<(ATMOSPHERE-LAYER-\d+)>(.+)"
    lines0 = re.findall(pat, cfg0)
    lines1 = re.findall(pat, cfg1)

    print("\n[debug_layers] Preview of first scaled layers")
    for (tag0, vals0), (tag1, vals1) in zip(lines0[:n_layers], lines1[:n_layers]):
        v0 = [x.strip() for x in vals0.split(",")]
        v1 = [x.strip() for x in vals1.split(",")]

        print(f"\n{tag0}")
        print(f"  pressure    = {v0[0]}")
        print(f"  temperature = {v0[1]}")
        print(f"  H2O before  = {v0[2]}")
        print(f"  H2O after   = {v1[2]}")
        print(f"  CO2 check   = {v0[3]} -> {v1[3]}")
        print(f"  O2  check   = {v0[8]} -> {v1[8]}")

# ============================================================
# COLUMN IDENTIFICATION CHECK
# ============================================================

def check_h2o_column(wavelength_cfg_nm):
    """
    Identify which PSG output column actually contains H2O absorption by
    running two calls — one at full H2O and one at strongly reduced H2O —
    and finding which column changes the most.

    This version uses the same Option B atmosphere mode as the rest of the
    pipeline, so the diagnostic matches production behavior.
    """
    print("\n[h2o_check] Running 2 PSG calls to identify real H2O column.")

    # Call 1: full template H2O (scale = 1.0)
    cfg_full = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg_full = force_user_atmosphere_mode(cfg_full)
    cfg_full = _apply_geometry_and_range(cfg_full, 1.0, wavelength_cfg_nm)
    arr_full, col_names = run_psg(cfg_full)
    if arr_full is None:
        raise RuntimeError("[h2o_check] PSG call 1 failed.")

    # Call 2: near-zero H2O using the active scaling path
    cfg_zero = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg_zero = force_user_atmosphere_mode(cfg_zero)
    cfg_zero = _apply_geometry_and_range(cfg_zero, 1.0, wavelength_cfg_nm)
    cfg_zero = apply_h2o_scaling(cfg_zero, 0.0001)
    arr_zero, _ = run_psg(cfg_zero)
    if arr_zero is None:
        raise RuntimeError("[h2o_check] PSG call 2 failed.")

    # Compare in the 1850–1950 nm H2O band
    lam = 1e7 / arr_full[:, 0]
    band = (lam >= 1300) & (lam <= 1500)

    print(f"\n[h2o_check] Column comparison in 1850-1950 nm band "
          f"(full H2O vs near-zero H2O):")
    print(f"{'Col':>4}  {'Name':>12}  {'tau_full':>12}  {'tau_zero':>12}  "
          f"{'delta_tau':>12}")
    print("-" * 60)

    max_delta = 0.0
    real_h2o_idx = None

    for i in range(arr_full.shape[1]):
        if i == 0:
            continue

        t_full = arr_full[band, i]
        t_zero = arr_zero[band, i]

        tau_f = float(np.median(-np.log(np.clip(t_full, 1e-10, 1.0))))
        tau_z = float(np.median(-np.log(np.clip(t_zero, 1e-10, 1.0))))
        delta = abs(tau_f - tau_z)

        name = col_names[i] if i < len(col_names) else "?"
        marker = "<-- H2O" if delta > max_delta and delta > 0.001 else ""

        print(f"{i:>4}  {name:>12}  {tau_f:>12.6f}  {tau_z:>12.6f}  "
              f"{delta:>12.6f}  {marker}")

        if delta > max_delta:
            max_delta = delta
            real_h2o_idx = i

    print(f"\n[h2o_check] Column with largest tau change: col {real_h2o_idx} "
          f"({col_names[real_h2o_idx] if real_h2o_idx < len(col_names) else '?'})")
    print(f"[h2o_check] This is the real H2O column. "
          f"Update PSG_H2O_LABEL if it differs from 'H2O'.")


# ============================================================
# SENSITIVITY CHECK
# ============================================================

def check_abun_sensitivity(wavelength_cfg_nm):
    """
    Determine which H2O scaling mechanism PSG actually respects under the
    current atmosphere mode by comparing:
      1) ATMOSPHERE-ABUN scaling
      2) direct layer VMR scaling

    This version runs under Option B so the diagnostic matches the
    production configuration.
    """
    print("\n[sensitivity] Testing H2O scaling mechanisms — 6 PSG calls.")

    scales = [1.0, 0.1, 0.01]
    tau_abun = []
    tau_layer = []

    for scale in scales:
        # --- ABUN path ---
        cfg_a = load_psg_template(TEMPLATE_PATH)
        if USE_OPTION_B_USER_ATMOSPHERE:
            cfg_a = force_user_atmosphere_mode(cfg_a)
        cfg_a = _apply_geometry_and_range(cfg_a, 1.0, wavelength_cfg_nm)
        cfg_a = _set_h2o_abundance(cfg_a, scale)
        arr_a, col_names_a = run_psg(cfg_a)

        # --- layer path ---
        cfg_l = load_psg_template(TEMPLATE_PATH)
        if USE_OPTION_B_USER_ATMOSPHERE:
            cfg_l = force_user_atmosphere_mode(cfg_l)
        cfg_l = _apply_geometry_and_range(cfg_l, 1.0, wavelength_cfg_nm)
        cfg_l = scale_h2o_in_layers(cfg_l, scale)
        arr_l, col_names_l = run_psg(cfg_l)

        for label, arr, col_names, store in [
            ("ABUN",  arr_a, col_names_a, tau_abun),
            ("LAYER", arr_l, col_names_l, tau_layer),
        ]:
            if arr is None:
                store.append(None)
                print(f"[sensitivity] scale={scale:.3f}  {label:5s}  PSG FAILED")
                continue

            lam = 1e7 / arr[:, 0]
            band = (lam >= 1850) & (lam <= 1950)

            h2o_idx = find_column(col_names, PSG_H2O_LABEL)
            if h2o_idx is None:
                h2o_idx = 2

            h2o = arr[band, h2o_idx]
            tau = -np.log(np.clip(h2o, 1e-10, 1.0))
            med = float(np.median(tau))

            store.append(med)
            print(
                f"[sensitivity] scale={scale:.3f}  {label:5s}  "
                f"median_tau={med:.6f}  "
                f"H2O=[{h2o.min():.6f}, {h2o.max():.6f}]"
            )

    print("\n[sensitivity] --- VERDICT ---")
    abun_varies = (None not in tau_abun) and (max(tau_abun) - min(tau_abun) > 0.001)
    layer_varies = (None not in tau_layer) and (max(tau_layer) - min(tau_layer) > 0.001)

    if abun_varies and not layer_varies:
        print("[sensitivity] -> ATMOSPHERE-ABUN controls H2O: use H2O_SCALING_MODE='abun'")
    elif layer_varies and not abun_varies:
        print("[sensitivity] -> Layer VMR patching controls H2O: use H2O_SCALING_MODE='layers'")
    elif abun_varies and layer_varies:
        print("[sensitivity] -> Both mechanisms work: prefer H2O_SCALING_MODE='abun'")
    else:
        print("[sensitivity] -> NEITHER mechanism changes tau.")
        print("[sensitivity]    PSG is still not honoring user H2O control.")
        print("[sensitivity]    Re-check ATMOSPHERE-TYPE / template settings.")


# ============================================================
# PWV CALIBRATION
# ============================================================

def calibrate_template_pwv(wavelength_cfg_nm, test_pwv_mm=5.0,
                           rough_guess_mm=20.0):
    """
    Empirically determine TEMPLATE_PWV_MM — the physical PWV (mm) that PSG
    models when the selected H2O control path is set to scale = 1.0.

    This version uses the same atmosphere mode and scaling path as the
    production grid build.
    """
    print("\n[calibration] Starting template PWV calibration — 3 PSG calls.")

    # --- Call 1: scale = 1.0 ---
    cfg1 = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg1 = force_user_atmosphere_mode(cfg1)
    cfg1 = _apply_geometry_and_range(cfg1, 1.0, wavelength_cfg_nm)
    arr1, col_names1 = run_psg(cfg1)
    if arr1 is None:
        raise RuntimeError("[calibration] PSG call 1 (scale=1.0) failed.")

    # --- Call 2: scale = 0.5 ---
    cfg2 = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg2 = force_user_atmosphere_mode(cfg2)
    cfg2 = _apply_geometry_and_range(cfg2, 1.0, wavelength_cfg_nm)
    cfg2 = apply_h2o_scaling(cfg2, 0.5)
    arr2, col_names2 = run_psg(cfg2)
    if arr2 is None:
        raise RuntimeError("[calibration] PSG call 2 (scale=0.5) failed.")

    # Use the real H2O column if found; otherwise fall back to col 2
    h2o_idx1 = find_column(col_names1, PSG_H2O_LABEL)
    h2o_idx2 = find_column(col_names2, PSG_H2O_LABEL)
    if h2o_idx1 is None:
        h2o_idx1 = 2
    if h2o_idx2 is None:
        h2o_idx2 = 2

    lam1 = 1e7 / arr1[:, 0]
    band = (lam1 >= 1850) & (lam1 <= 1950)

    if band.sum() < 10:
        raise RuntimeError(
            "[calibration] Fewer than 10 pixels in 1850-1950 nm band. "
            "Check wavelength range covers this window."
        )

    h2o_full = arr1[band, h2o_idx1]
    h2o_half = arr2[band, h2o_idx2]

    tau_full = -np.log(np.clip(h2o_full, 1e-10, 1.0))
    tau_half = -np.log(np.clip(h2o_half, 1e-10, 1.0))

    valid = (tau_full > 0.01) & np.isfinite(tau_full) & np.isfinite(tau_half)

    print(f"[calibration] band pixels: {band.sum()}")
    print(f"[calibration] H2O min/max scale=1: {h2o_full.min():.6f}, {h2o_full.max():.6f}")
    print(f"[calibration] median tau scale=1: {np.median(tau_full):.6f}")
    print(f"[calibration] pixels with tau>0.01: {valid.sum()}")

    if valid.sum() < 5:
        raise RuntimeError(
            "[calibration] Too few pixels with tau > 0.01 in H2O band. "
            "H2O scaling is still not affecting the selected band strongly enough."
        )

    ratio_half = np.median(tau_half[valid] / tau_full[valid])
    ok = abs(ratio_half - 0.5) < 0.1
    print(
        f"[calibration] tau(0.5)/tau(1.0) = {ratio_half:.4f}  "
        f"({'OK' if ok else 'WARNING: non-linear or saturated band'})"
    )

    # --- Call 3: scale = test_pwv_mm / rough_guess_mm ---
    scale_test = test_pwv_mm / rough_guess_mm
    cfg3 = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg3 = force_user_atmosphere_mode(cfg3)
    cfg3 = _apply_geometry_and_range(cfg3, 1.0, wavelength_cfg_nm)
    cfg3 = apply_h2o_scaling(cfg3, scale_test)
    arr3, col_names3 = run_psg(cfg3)
    if arr3 is None:
        raise RuntimeError("[calibration] PSG call 3 (scale=test) failed.")

    h2o_idx3 = find_column(col_names3, PSG_H2O_LABEL)
    if h2o_idx3 is None:
        h2o_idx3 = 2

    h2o_test = arr3[band, h2o_idx3]
    tau_test = -np.log(np.clip(h2o_test, 1e-10, 1.0))
    ratio_test = np.median(tau_test[valid] / tau_full[valid])

    template_pwv = test_pwv_mm / ratio_test

    print(
        f"[calibration] scale_test={scale_test:.6f}  "
        f"tau_ratio={ratio_test:.6f}  "
        f"=> TEMPLATE_PWV_MM = {template_pwv:.4f} mm"
    )
    print(f"[calibration] Paste this at the top of the file:")
    print(f"[calibration]   TEMPLATE_PWV_MM = {template_pwv:.4f}")

    return template_pwv


# ============================================================
# BUILD PSG INPUT
# ============================================================

# def _remove_h2o_from_layers(cfg):
#     """
#     Remove H2O from ATMOSPHERE-LAYERS-MOLECULES so PSG does not pull it
#     from MERRA-2.

#     When H2O is listed in ATMOSPHERE-LAYERS-MOLECULES, PSG overrides the
#     layer VMRs with its MERRA-2 reanalysis data for the template date and
#     location, making PWV scaling impossible.  Removing H2O from this list
#     forces PSG to use ATMOSPHERE-ABUN instead, which we can control.

#     All other gases (CO2, O3, O2, etc.) remain in the layers list so they
#     continue to benefit from the MERRA-2-informed pressure/temperature
#     structure.
#     """
#     layers_mols = _get_tag(cfg, "ATMOSPHERE-LAYERS-MOLECULES")
#     if layers_mols is None:
#         return cfg
#     mols = [m.strip() for m in layers_mols.split(",")]
#     mols = [m for m in mols if m.upper() != "H2O"]
#     return set_parameter(cfg, "ATMOSPHERE-LAYERS-MOLECULES", ",".join(mols))


def build_psg_input(am, pwv, T, wavelength_cfg_nm):
    """
    Build a PSG config string for the requested (airmass, PWV, temperature).

    This version uses the same atmosphere mode and H2O scaling path as the
    calibration routine.
    """
    if TEMPLATE_PWV_MM is None:
        raise RuntimeError(
            "TEMPLATE_PWV_MM is not set. Run calibrate_template_pwv() once, "
            "note the printed value, and hardcode it at the top of the file."
        )
    if TEMPLATE_PWV_MM <= 0:
        raise RuntimeError("TEMPLATE_PWV_MM must be positive.")

    cfg = load_psg_template(TEMPLATE_PATH)

    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg = force_user_atmosphere_mode(cfg)

    cfg = _apply_geometry_and_range(cfg, am, wavelength_cfg_nm)

    h2o_scale = pwv / TEMPLATE_PWV_MM
    cfg = apply_h2o_scaling(cfg, h2o_scale)

    print(
        f"[PWV] template={TEMPLATE_PWV_MM:.4f} mm  "
        f"requested={pwv:.4f} mm  h2o_scale={h2o_scale:.8f}  "
        f"mode={H2O_SCALING_MODE}"
    )

    cfg = set_parameter(cfg, "SURFACE-TEMPERATURE", f"{float(T):.6f}")
    return cfg


# ============================================================
# PSG RESPONSE PARSING
# ============================================================

def parse_psg_header(lines):
    """
    Extract column names from PSG comment lines.

    PSG writes '#'-prefixed lines before the data.  The column header line
    contains recognisable gas names (H2O, CO2, Total, ...).
    Returns a list of strings, one per data column, or [] if not found.

    Example PSG header line:
        # Wave/freq  Total  H2O  CO2  O3  N2O  CO  CH4  O2  N2  CIA
    """
    gas_hints = {"H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2",
                 "Total", "Rayleigh", "CIA", "Wave", "Wave/freq", "N2"}

    for ln in lines:
        stripped = ln.strip()
        if not stripped.startswith("#"):
            continue
        parts = stripped.lstrip("#").split()
        if sum(1 for p in parts if p in gas_hints) >= 2:
            return parts

    return []


def find_column(col_names, label):
    """
    Return the integer index of *label* in col_names, or None if absent.
    Matching is case-insensitive; 'Wave/freq' matches 'wave'.
    """
    label_lo = label.lower()
    for i, name in enumerate(col_names):
        if name.lower() == label_lo:
            return i
        if label_lo == "wave" and name.lower().startswith("wave"):
            return i
    return None


def run_psg(psg_config_text, timeout=180, max_tries=10, debug_dump=True,
            output_type="trn"):
    """
    POST a PSG config to the API and return:
        arr       : 2-D float array, shape (N_pixels, N_columns)
        col_names : list of column-name strings from the '#' header
                    (empty list if the header could not be parsed)

    output_type : PSG 'type' POST parameter ('trn' for transmission).
    Backoff     : wait_s accumulates correctly across retries.
    """
    wait_s = 5

    for attempt in range(max_tries):
        print(f"\nAttempt {attempt+1}/{max_tries} — PSG (type={output_type})")

        try:
            response = requests.post(
                PSG_API,
                data={"file": psg_config_text, "type": output_type},
                timeout=timeout,
            )

            print("HTTP status:", response.status_code)

            if response.status_code != 200:
                print("Non-200 response.")
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
                print("PSG busy — waiting.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            if len(text) < 100:
                print("Response suspiciously short.")
                time.sleep(wait_s)
                wait_s = min(wait_s * 1.5, 120)
                continue

            lines     = text.splitlines()
            col_names = parse_psg_header(lines)

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
                arr = np.genfromtxt(
                    io.StringIO("\n".join(data_lines)), dtype=float
                )
            except Exception as e:
                print("Parse error:", e)
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
                print("Columns:", col_names)
            else:
                print("Warning: could not parse column names from header.")

            total_idx = find_column(col_names, PSG_TOTAL_LABEL)
            if total_idx is not None and total_idx < arr.shape[1]:
                print(f"Transmission min/max: "
                      f"{np.nanmin(arr[:, total_idx]):.4e}  "
                      f"{np.nanmax(arr[:, total_idx]):.6f}")
            else:
                print(f"Transmission min/max (col 1 fallback): "
                      f"{np.nanmin(arr[:, 1]):.4e}  "
                      f"{np.nanmax(arr[:, 1]):.6f}")

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
    """Normalise whitespace and return a 12-char SHA-256 prefix."""
    lines      = [ln.rstrip() for ln in cfg_text.replace("\r\n", "\n").split("\n")]
    lines      = [ln for ln in lines if ln.strip()]
    normalized = "\n".join(lines) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def _grid_hash(wavelength_cfg, airmass_grid, pwv_grid, temp_grid):
    """
    Stable hash of the grid axes + wavelength config.
    Also folds in TEMPLATE_PWV_MM so a recalibration triggers a rebuild.
    """
    key = str((
        tuple(wavelength_cfg),
        tuple(sorted(airmass_grid)),
        tuple(sorted(pwv_grid)),
        tuple(sorted(temp_grid)),
        float(TEMPLATE_PWV_MM) if TEMPLATE_PWV_MM is not None else None,
    ))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def grid_is_valid(h5_filename, wavelength_cfg, airmass_grid,
                  pwv_grid, temp_grid):
    """
    Return True iff the HDF5 file exists and its stored grid_hash matches
    the hash of the current grid parameters (including TEMPLATE_PWV_MM).
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
            raise ValueError("Cache file is not valid 2-D numeric data.")
        return arr

    def compute_or_load(am, pwv, T, force_recompute=False):
        psg_in   = build_psg_input(am, pwv, T, wavelength_cfg_nm)
        cfg_hash = _stable_cfg_hash(psg_in)
        p        = cache_path(am, pwv, T, cfg_hash)

        print(f"[grid] am={am:.2f} pwv={pwv:.2f} T={T}  hash={cfg_hash}")

        if (not force_recompute) and os.path.exists(p) \
                and os.path.getsize(p) > 0:
            print("  -> CACHE HIT:", p)
            try:
                return load_cached(p), True
            except Exception as e:
                print("  -> cache read failed, recomputing:", repr(e))

        arr, _ = run_psg(psg_in)
        if arr is None:
            raise RuntimeError(
                f"PSG returned no data for am={am}, pwv={pwv}, T={T}"
            )

        np.savetxt(p, arr)
        return arr, False

    # Wavelength reference from first grid point
    ref_arr, _ = compute_or_load(airmasses[0], pwv_values[0], temperatures[0])
    wn_ref  = ref_arr[:, 0].astype(float)
    lam_ref = 1e7 / wn_ref
    o       = np.argsort(lam_ref)
    lam_ref = lam_ref[o]
    nlam    = len(lam_ref)

    # Only total transmission (col 1) is stored in the interpolation grid.
    # Per-molecule columns are fetched on-demand in __main__.
    TOTAL_COL = 1

    grid_specs = np.full(
        (len(airmasses), len(pwv_values), len(temperatures), nlam), np.nan
    )

    for i, am in enumerate(airmasses):
        for j, pwv in enumerate(pwv_values):
            for k, T in enumerate(temperatures):
                arr, _ = compute_or_load(am, pwv, T)
                wn     = arr[:, 0].astype(float)
                trans  = arr[:, TOTAL_COL].astype(float)
                lam_nm = 1e7 / wn
                ord_   = np.argsort(lam_nm)
                lam_nm = lam_nm[ord_]
                trans  = np.nan_to_num(trans[ord_], nan=0.0)
                grid_specs[i, j, k, :] = np.interp(
                    lam_ref, lam_nm, trans, left=np.nan, right=np.nan
                )

    for i, am in enumerate(airmasses):
        for j, pwv in enumerate(pwv_values):
            for k, T in enumerate(temperatures):
                print(f"[loop] i={i} j={j} k={k} am={am} pwv={pwv} T={T}")
                arr, _ = compute_or_load(am, pwv, T)

    gh = _grid_hash(wavelength_cfg_nm, airmasses, pwv_values, temperatures)
    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset("spectra",        data=grid_specs)
        hf.create_dataset("airmasses",      data=np.array(airmasses,    dtype=float))
        hf.create_dataset("pwv_values",     data=np.array(pwv_values,   dtype=float))
        hf.create_dataset("temperatures",   data=np.array(temperatures, dtype=float))
        hf.create_dataset("wavelengths_nm", data=lam_ref)
        hf.attrs["grid_hash"]       = gh
        hf.attrs["template_pwv_mm"] = float(TEMPLATE_PWV_MM)

    print(f"Grid written to {h5_filename}  (hash={gh}  "
          f"template_pwv={TEMPLATE_PWV_MM:.4f} mm)")


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
    Evaluate the telluric interpolator at a single (am, pwv, T) point.
    Always use this wrapper — it enforces the correct [[am, pwv, T]] shape.
    """
    return interp([[float(am), float(pwv), float(T)]])[0]


# ============================================================
# FORWARD MODEL
# ============================================================

def convolve_to_R(wave_nm, flux, R_inst):
    """
    Convolve flux with a Gaussian LSF at resolving power R_inst.
    Operates in log-lambda space so sigma_pix is constant across the band.
    """
    wave_nm = np.asarray(wave_nm, float)
    flux    = np.asarray(flux,    float)

    lnw    = np.log(wave_nm)
    lnw_u  = np.linspace(lnw.min(), lnw.max(), len(wave_nm))
    flux_u = np.interp(lnw_u, lnw, flux)

    sigma_lnw   = (1.0 / R_inst) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dln         = lnw_u[1] - lnw_u[0]
    sigma_pix   = sigma_lnw / dln
    flux_conv_u = gaussian_filter1d(flux_u, sigma_pix, mode="nearest")

    return np.interp(lnw, lnw_u, flux_conv_u)


def forward_model_from_base(T_base, dlam_nm, wave_obs, lam_grid, R_inst):
    """
    Convolve T_base to R_inst, apply wavelength shift dlam_nm, and resample
    onto wave_obs.  Continuum scaling is handled analytically in chi2_full.
    """
    T_conv       = convolve_to_R(lam_grid, T_base, R_inst)
    wave_shifted = wave_obs - dlam_nm
    return np.interp(wave_shifted, lam_grid, T_conv, left=np.nan, right=np.nan)


# ============================================================
# FITTING / CHI2
# ============================================================

def chi2_full(params, T_fixed, interp, wave_obs, flux_obs, sigma_obs,
              mask, lam_grid, R_inst):
    """
    Weighted chi-squared with analytic linear continuum marginalisation.
    Continuum model: (c0 + c1*(lambda - lambda0)) * T_model
    """
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
    Evaluate chi2_full on a 2-D (airmass, PWV) grid at dlam=0.
    Uses the same analytic continuum as the optimizer so the heatmap
    minimum corresponds directly to the optimizer's minimum.
    """
    chi2_map = np.full((len(am_scan), len(pwv_scan)), np.nan)

    for i, am in enumerate(am_scan):
        for j, pwv in enumerate(pwv_scan):
            chi2_map[i, j] = chi2_full(
                [am, pwv, 0.0],
                T_fixed, telluric_interp,
                wave_obs, flux_obs, sigma_obs,
                mask, lam_grid, R_inst,
            )

    return chi2_map


def fit_am_pwv_dlam(interp, T_fixed, wave_obs, flux_obs, sigma_obs,
                    mask, lam_grid, R_inst):
    """L-BFGS-B fit for (airmass, PWV, wavelength shift)."""
    x0     = [1.5, 2.0, 0.0]
    bounds = [(1.0, 2.5), (0.1, 10.0), (-0.05, 0.05)]

    res = minimize(
        chi2_full, x0,
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
                     color="grey", alpha=0.3, label="1-sigma uncertainty")
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
    plt.colorbar(im, label="Delta chi2")

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
    plt.title("Chi2 Heatmap (Airmass vs PWV)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_h2o_comparison(lam_nm, h2o_raw, total_raw=None, R_inst=25000,
                        h2o_col_name="H2O"):
    """
    Three-panel plot: raw H2O, convolved H2O, convolved total transmission.
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
        axes[2].set_title("Convolved H2O (repeated — total not available)")

    plt.tight_layout()
    plt.show()


def print_column_diagnostics(arr_raw, col_names):
    """
    Print min/max/std/mean for every column in arr_raw with names from the
    parsed PSG header.  Run after every fresh PSG call to catch column
    ordering surprises early.
    """
    n_cols = arr_raw.shape[1]
    print(f"\narr_raw shape: {arr_raw.shape}")
    hdr = (f"{'Col':>4}  {'Name':>12}  {'Min':>12}  {'Max':>12}  "
           f"{'Std':>12}  {'Mean':>12}")
    print(hdr)
    print("-" * len(hdr))
    for i in range(n_cols):
        col  = arr_raw[:, i]
        name = col_names[i] if i < len(col_names) else "?"
        print(f"{i:>4}  {name:>12}  {col.min():>12.6f}  {col.max():>12.6f}  "
              f"{col.std():>12.6f}  {col.mean():>12.6f}")

def debug_h2o_response(wavelength_cfg_nm,
                       scale_lo=0.05,
                       scale_hi=1.0,
                       am=1.0,
                       T=280.0,
                       band_nm=(1500, 2500),
                       make_plots=True):
    """
    Diagnose where H2O scaling is actually affecting the PSG output.

    Runs two PSG calls with the same settings except for H2O scale, then:
      - compares all output columns
      - reports which column changes the most in tau
      - plots Total and H2O differences
      - checks whether total transmission changes even if PSG's H2O column does not
    """
    print("\n[debug_h2o] ==================================================")
    print(f"[debug_h2o] H2O_SCALING_MODE = {H2O_SCALING_MODE}")
    print(f"[debug_h2o] scale_lo={scale_lo}, scale_hi={scale_hi}, am={am}, T={T}")
    print("[debug_h2o] ==================================================")

    # ---------- low-H2O run ----------
    cfg_lo = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg_lo = force_user_atmosphere_mode(cfg_lo)
    cfg_lo = _apply_geometry_and_range(cfg_lo, am, wavelength_cfg_nm)
    cfg_lo = apply_h2o_scaling(cfg_lo, scale_lo)
    cfg_lo = set_parameter(cfg_lo, "SURFACE-TEMPERATURE", f"{float(T):.6f}")

    arr_lo, names_lo = run_psg(cfg_lo)
    if arr_lo is None:
        raise RuntimeError("[debug_h2o] low-H2O PSG call failed.")

    # ---------- high-H2O run ----------
    cfg_hi = load_psg_template(TEMPLATE_PATH)
    if USE_OPTION_B_USER_ATMOSPHERE:
        cfg_hi = force_user_atmosphere_mode(cfg_hi)
    cfg_hi = _apply_geometry_and_range(cfg_hi, am, wavelength_cfg_nm)
    cfg_hi = apply_h2o_scaling(cfg_hi, scale_hi)
    cfg_hi = set_parameter(cfg_hi, "SURFACE-TEMPERATURE", f"{float(T):.6f}")

    arr_hi, names_hi = run_psg(cfg_hi)
    if arr_hi is None:
        raise RuntimeError("[debug_h2o] high-H2O PSG call failed.")

    # ---------- wavelength alignment ----------
    lam_lo = 1e7 / arr_lo[:, 0]
    lam_hi = 1e7 / arr_hi[:, 0]

    ord_lo = np.argsort(lam_lo)
    ord_hi = np.argsort(lam_hi)

    lam_lo = lam_lo[ord_lo]
    lam_hi = lam_hi[ord_hi]
    arr_lo = arr_lo[ord_lo, :]
    arr_hi = arr_hi[ord_hi, :]

    if len(lam_lo) != len(lam_hi) or not np.allclose(lam_lo, lam_hi, rtol=0, atol=1e-8):
        raise RuntimeError("[debug_h2o] wavelength grids do not match between runs.")

    lam = lam_lo
    band = (lam >= band_nm[0]) & (lam <= band_nm[1])

    total_idx = find_column(names_hi, PSG_TOTAL_LABEL)
    h2o_idx   = find_column(names_hi, PSG_H2O_LABEL)

    if total_idx is None:
        total_idx = 1
    if h2o_idx is None:
        h2o_idx = 2

    print(f"[debug_h2o] total_idx={total_idx}, h2o_idx={h2o_idx}")
    print(f"[debug_h2o] wavelength range = {lam.min():.3f} - {lam.max():.3f} nm")
    print(f"[debug_h2o] band = {band_nm[0]} - {band_nm[1]} nm, N={band.sum()} pixels")

    # ---------- compare all columns ----------
    print("\n[debug_h2o] Column-by-column response to H2O scaling")
    print(f"{'Col':>4}  {'Name':>12}  {'max|dT|':>12}  {'median dT':>12}  {'median dTau':>12}")
    print("-" * 62)

    best_col = None
    best_dtau = -np.inf

    for i in range(arr_hi.shape[1]):
        if i == 0:
            continue

        x_lo = np.clip(arr_lo[band, i], 1e-12, 1.0)
        x_hi = np.clip(arr_hi[band, i], 1e-12, 1.0)

        dT   = x_hi - x_lo
        dTau = (-np.log(x_hi)) - (-np.log(x_lo))

        name = names_hi[i] if i < len(names_hi) else f"col{i}"

        med_dtau = float(np.median(np.abs(dTau)))
        print(f"{i:>4}  {name:>12}  {np.max(np.abs(dT)):>12.6e}  "
              f"{np.median(dT):>12.6e}  {med_dtau:>12.6e}")

        if med_dtau > best_dtau:
            best_dtau = med_dtau
            best_col = i

    best_name = names_hi[best_col] if best_col < len(names_hi) else f"col{best_col}"
    print(f"\n[debug_h2o] Most H2O-sensitive column in this band: {best_col} ({best_name})")

    # ---------- direct summaries ----------
    total_lo = np.clip(arr_lo[:, total_idx], 1e-12, 1.0)
    total_hi = np.clip(arr_hi[:, total_idx], 1e-12, 1.0)
    h2o_lo   = np.clip(arr_lo[:, h2o_idx],   1e-12, 1.0)
    h2o_hi   = np.clip(arr_hi[:, h2o_idx],   1e-12, 1.0)

    d_total = total_hi - total_lo
    d_h2o   = h2o_hi - h2o_lo

    print("\n[debug_h2o] Summary in selected band")
    print(f"  max|Total_hi - Total_lo| = {np.max(np.abs(d_total[band])):.6e}")
    print(f"  max|H2O_hi   - H2O_lo|   = {np.max(np.abs(d_h2o[band])):.6e}")
    print(f"  median|dTau_total|       = {np.median(np.abs((-np.log(total_hi[band])) - (-np.log(total_lo[band])))):.6e}")
    print(f"  median|dTau_h2o|         = {np.median(np.abs((-np.log(h2o_hi[band]))   - (-np.log(h2o_lo[band])))):.6e}")

    # ---------- narrow-band focus around strongest total response ----------
    idx_peak = np.argmax(np.abs(d_total[band]))
    lam_band = lam[band]
    lam_peak = lam_band[idx_peak]
    print(f"[debug_h2o] strongest total-response wavelength: {lam_peak:.3f} nm")

    # ---------- plots ----------
    if make_plots:
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(lam, total_lo, label=f"Total (scale={scale_lo})")
        plt.plot(lam, total_hi, label=f"Total (scale={scale_hi})", alpha=0.8)
        plt.xlim(band_nm)
        plt.ylabel("Transmission")
        plt.title("Total transmission: low vs high H2O")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(lam, h2o_lo, label=f"H2O (scale={scale_lo})")
        plt.plot(lam, h2o_hi, label=f"H2O (scale={scale_hi})", alpha=0.8)
        plt.xlim(band_nm)
        plt.ylabel("Transmission")
        plt.title("PSG H2O column: low vs high H2O")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(lam, d_total, label="Total_hi - Total_lo")
        plt.plot(lam, d_h2o, label="H2O_hi - H2O_lo")
        plt.xlim(band_nm)
        plt.ylabel("Δ Transmission")
        plt.title("Difference spectrum")
        plt.legend()

        zoom_halfwidth = 10.0  # nm
        zoom = (lam >= lam_peak - zoom_halfwidth) & (lam <= lam_peak + zoom_halfwidth)

        plt.subplot(4, 1, 4)
        plt.plot(lam[zoom], total_lo[zoom], label=f"Total (scale={scale_lo})")
        plt.plot(lam[zoom], total_hi[zoom], label=f"Total (scale={scale_hi})", alpha=0.8)
        plt.axvline(lam_peak, linestyle="--", alpha=0.6)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmission")
        plt.title(f"Zoom around strongest total response ({lam_peak:.3f} nm)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return {
        "lam_nm": lam,
        "arr_lo": arr_lo,
        "arr_hi": arr_hi,
        "col_names": names_hi,
        "total_idx": total_idx,
        "h2o_idx": h2o_idx,
        "best_col": best_col,
        "best_name": best_name,
        "lam_peak_nm": lam_peak,
        "max_abs_d_total": float(np.max(np.abs(d_total[band]))),
        "max_abs_d_h2o": float(np.max(np.abs(d_h2o[band]))),
    }

# ============================================================
# MAIN WORKFLOW
# ============================================================

if __name__ == "__main__":

    # R_psg  : PSG native sampling resolution (>> R_inst for oversampling)
    # R_inst : Keck LIGER instrument resolving power
    R_psg  = 300000
    R_inst = 8000

    wavelength_cfg = [1500, 2500, R_psg]
    airmass_grid   = [1.0, 1.2, 1.5, 2.0]
    pwv_grid       = [0.5, 1.0, 2.0, 3.0, 5.0]
    temp_grid      = [270, 280, 290]

    H5_FILE = "telluric_grid.h5"

    # ----------------------------------------------------------------
    # COLUMN IDENTIFICATION CHECK — run once to find real H2O column
    # ----------------------------------------------------------------
    # Uncomment the two lines below, run, paste output, then re-comment.
    # Makes 1 PSG call with near-zero H2O to identify which output column
    # actually contains H2O absorption by process of elimination.
    # check_h2o_column(wavelength_cfg)
    # raise SystemExit(0)

    # ----------------------------------------------------------------
    # SENSITIVITY CHECK — run once to determine correct H2O mechanism
    # ----------------------------------------------------------------
    # Uncomment the two lines below, run, read the VERDICT, then
    # re-comment them.  This makes 6 PSG calls.
    # check_abun_sensitivity(wavelength_cfg)
    # raise SystemExit(0)

    # ----------------------------------------------------------------
    # PWV CALIBRATION — runs once, then never again
    # ----------------------------------------------------------------
    # When TEMPLATE_PWV_MM is None, calibrate_template_pwv() makes 3 PSG
    # calls, prints the value, and exits.  Paste the printed value into
    # TEMPLATE_PWV_MM at the top of the file, then re-run normally.
    # Re-run calibration any time earth_cfg.txt is replaced.
    if TEMPLATE_PWV_MM is None:
        print("TEMPLATE_PWV_MM not set — running one-time calibration.")
        calibrated = calibrate_template_pwv(wavelength_cfg)
        print("\n*** Paste this at the top of the file and re-run: ***")
        print(f"    TEMPLATE_PWV_MM = {calibrated:.4f}")
        raise SystemExit(0)

    # ----------------------------------------------------------------
    # SMART CACHE INVALIDATION
    # ----------------------------------------------------------------
    # Rebuilds only when grid axes or TEMPLATE_PWV_MM have changed.
    # Manual pipeline logic changes (not reflected in parameters) still
    # require a manual cache wipe.
    if grid_is_valid(H5_FILE, wavelength_cfg, airmass_grid,
                     pwv_grid, temp_grid):
        print("Grid is current — loading.")
    else:
        print("Grid parameters changed or missing — rebuilding.")
        if os.path.exists(H5_FILE):
            os.remove(H5_FILE)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        write_grid_to_hdf5(
            H5_FILE, airmass_grid, pwv_grid, temp_grid, wavelength_cfg
        )

    # ----------------------------------------------------------------
    # LOAD INTERPOLATOR
    # ----------------------------------------------------------------
    telluric_interp, lam_grid = make_telluric_interp(H5_FILE)

    # ----------------------------------------------------------------
    # TRUE / FIXED PARAMETERS
    # ----------------------------------------------------------------
    am        = 1.5
    pwv       = 2.0
    T_fixed   = 280
    dlam_true = 0.015

    T_base = eval_interp(telluric_interp, am, pwv, T_fixed)

    # Temperature sensitivity sanity check
    spec1 = eval_interp(telluric_interp, 1.5, 2.0, 270.0)
    spec2 = eval_interp(telluric_interp, 1.5, 2.0, 290.0)
    print("Max |dT| from temperature change:", np.max(np.abs(spec1 - spec2)))

    # Zoom plot around spectrum centre
    centre   = np.median(lam_grid)
    m        = (lam_grid > centre - 2) & (lam_grid < centre + 2)
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

    # ----------------------------------------------------------------
    # SYNTHETIC OBSERVATION
    # ----------------------------------------------------------------
    wave_obs  = lam_grid.copy()
    flux_true = forward_model_from_base(
        T_base, dlam_true, wave_obs, lam_grid, R_inst
    )

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

    # ----------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------
    best_params, chi2_val, res = fit_am_pwv_dlam(
        telluric_interp, T_fixed, wave_obs, flux_obs, sigma_obs,
        mask, lam_grid, R_inst,
    )

    print("\nTrue:      am={}, pwv={}, dlam={}".format(am, pwv, dlam_true))
    print("Recovered: am={:.4f}, pwv={:.4f}, dlam={:.6f}".format(*best_params))

    # ----------------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------------
    am_scan  = np.linspace(1.0, 2.5, 40)
    pwv_scan = np.linspace(0.1, 5.0, 40)

    plot_chi2_heatmap_am_pwv(
        am_scan, pwv_scan, T_fixed,
        wave_obs, flux_obs, sigma_obs, mask,
        telluric_interp, lam_grid, R_inst,
        best_params=(best_params[0], best_params[1]),
        true_params=(am, pwv),
    )

    plot_full_spectrum(
        wave_obs, flux_obs, sigma_obs,
        telluric_interp, lam_grid,
        best_params, T_fixed, R_inst,
    )

    # ----------------------------------------------------------------
    # MOLECULAR COMPONENT PLOT
    # ----------------------------------------------------------------
    # Fresh PSG call to retrieve per-molecule columns.
    # Column indices are resolved by name from the parsed header.
    psg_cfg = build_psg_input(am, pwv, T_fixed, wavelength_cfg)
    arr_raw, col_names = run_psg(psg_cfg)

    if arr_raw is None:
        raise RuntimeError("PSG failed for component plot.")

    # Sort into wavelength order
    wn      = arr_raw[:, 0]
    lam_nm  = 1e7 / wn
    order   = np.argsort(lam_nm)
    lam_nm  = lam_nm[order]
    arr_raw = arr_raw[order, :]

    print_column_diagnostics(arr_raw, col_names)

    total_idx = find_column(col_names, PSG_TOTAL_LABEL)
    h2o_idx   = find_column(col_names, PSG_H2O_LABEL)

    if total_idx is None:
        print("Warning: 'Total' not found by name — falling back to col 1.")
        total_idx = 1
    if h2o_idx is None:
        print("Warning: 'H2O' not found by name — falling back to col 2.")
        h2o_idx = 2

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
        print("H2O column not present — skipping component plot.")

    print("\nDone.")
    print(f"lambda range: {lam_grid.min():.1f} - {lam_grid.max():.1f} nm")
    print(f"N points: {len(lam_grid)}")
    print(f"Median d-lambda: {np.median(np.diff(lam_grid)):.4f} nm")
    print(f"Resolution element at 2000 nm: {2000/R_inst:.4f} nm")