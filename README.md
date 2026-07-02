# Telluric Correction Pipeline for Keck HISPEC/LIGER

A PSG-based telluric correction module for the HISPEC Data Reduction Pipeline (DRP) at Keck Observatory. Developed as part of the undergraduate research program at Keck Observatory.

---

## Overview

This pipeline generates synthetic telluric transmission spectra using NASA's [Planetary Spectrum Generator (PSG)](https://psg.gsfc.nasa.gov), applies Beer-Lambert airmass scaling per molecular species, and divides the model from observed spectra to produce telluric-corrected output.

The pipeline is designed to slot into the HISPEC DRP as a telluric calibration primitive. It covers the full HISPEC wavelength range (0.96–2.5 µm) across the y, J, H, and K bands.

---

## Science Background

Earth's atmosphere absorbs starlight at specific wavelengths due to molecules including H₂O, CO₂, CH₄, CO, O₃, N₂O, and O₂. These telluric absorption features must be removed before doing science with near-infrared spectra. The pipeline models this absorption using PSG's HITRAN-based radiative transfer, then scales the model to the airmass of each observation.

PWV (precipitable water vapor) is fixed from MERRA-2 meteorological data embedded in the PSG template rather than fitted freely. Fitting PWV and airmass simultaneously is unreliable due to their degeneracy, this follows the same approach used by MOLECFIT, TelFit, and Xtellcor.

---

## Architecture

```
PSG API
  └── fetch_stitched_spectrum()                                   # sub-band split + stitch for wide ranges
        └── build_grid_hdf5()                                     # 3-D grid: [airmass, T, wavelength]
              └── scale_psg()                                     # Beer-Lambert per-species scaling
                    └── make_telluric_interp()                    # RegularGridInterpolator
                          └── fit_telluric()                       # Differential Evolution fit
                                └── apply_telluric_correction()

save_average_psg_model()                 # one-time: write average model to FITS
  └── load_average_psg_model()            # load FITS model at runtime
        └── apply_telluric_primitive()   # DRP-facing function
              └── read_spectrum_fits()   # flexible FITS reader
```

---

## Key Design Choices

**Sub-band stitching:** At R=200,000 over 1540 nm, PSG cannot return the full HISPEC range in one call. The pipeline splits the range into configurable sub-bands (~500 nm each with 10 nm overlap), calls PSG for each, and stitches the responses. Everything above this layer sees one continuous spectrum.

**Beer-Lambert scaling:** PSG is called once at airmass=1.0 per temperature node. `scale_psg()` fills the airmass axis analytically:
```
T(airmass) = T_H2O^(airmass + pwv) × (T_CO2 × T_CH4 × ... )^airmass
```
This eliminates the need for a PSG call per airmass value.

**Differential Evolution fitting:** Telluric chi-square surfaces have broad, shallow basins where gradient methods get stuck. DE explores the full parameter space before polishing with L-BFGS-B.

**Disk caching:** Every PSG call is cached to disk keyed by a SHA-256 hash of the full config. Rebuilding after code changes (not PSG config changes) costs zero API calls. Column names are saved in a sidecar `.cols` file so they survive cache hits.

---

## Installation

```bash
pip install numpy scipy matplotlib h5py requests astropy
```

Clone the repo and place your PSG Earth config template (`earth_cfg.txt`) in the same directory as `telluric_pipeline.py`, or in `Data_reduc_pipe/`.

A baseline PSG Earth config can be exported from [https://psg.gsfc.nasa.gov](https://psg.gsfc.nasa.gov). The template must contain `ATMOSPHERE-LAYER` entries and should have:
```
<GENERATOR-RANGE1> 960
<GENERATOR-RANGE2> 2500
<GENERATOR-RANGEUNIT> nm
```

---

## Quick Start

### 1. Build the transmission grid

```python
from telluric_pipeline import PipelineConfig, SiteConfig, prepare_pipeline

cfg = PipelineConfig(
    observation_date    = "2024/01/15 05:00",   # nighttime UT
    wavelength_range_nm = (960.0, 2500.0),
    sub_band_width_nm   = 500.0,
    sub_band_overlap_nm = 10.0,
    h5_filename         = "telluric_grid.h5",
)

interp, interp_h2o, lam_grid = prepare_pipeline(
    cfg_obj      = cfg,
    airmass_grid = [1.0, 1.2, 1.5, 2.0],
    temp_grid    = [280.0],
)
```

### 2. Generate the average PSG model FITS file (one-time)

```python
from telluric_pipeline import save_average_psg_model

save_average_psg_model(
    cfg_obj   = cfg,
    fits_path = "average_psg_model.fits",
    am        = 1.0,
    surface_T = 280.0,
)
```

### 3. Apply telluric correction to an observed spectrum

```python
from telluric_pipeline import read_spectrum_fits, apply_telluric_primitive

wave, flux, sigma, header = read_spectrum_fits("observation.fits")

result = apply_telluric_primitive(
    wave_obs    = wave,
    flux_obs    = flux,
    sigma_obs   = sigma,
    model_fits  = "average_psg_model.fits",
    output_fits = "corrected.fits",
    header      = header,       # airmass read from FITS header automatically
    fit_airmass = True,         # optionally fit airmass against the data
)
```

### 4. Trim to a specific HISPEC channel

```python
from telluric_pipeline import trim_spectrum

# Blue spectrograph (y+J, 980-1400 nm)
lam_b, flux_b = trim_spectrum(wave, flux, channel="bspec")

# Red spectrograph (H+K, 1400-2500 nm)
lam_r, flux_r = trim_spectrum(wave, flux, channel="rspec")

# Custom range
lam_h, flux_h = trim_spectrum(wave, flux, l0=1500.0, l1=1800.0)
```

---

## Output FITS Format

`apply_telluric_primitive()` writes a three-extension FITS file:

| Extension | Name | Contents |
|---|---|---|
| 0 | PRIMARY | Observation metadata + telluric correction keywords |
| 1 | CORRECTED | `[wavelength_nm, corrected_flux, corrected_sigma]` |
| 2 | TELLURIC_MODEL | `[wavelength_nm, telluric_transmission]` |

Key header keywords written to PRIMARY:

| Keyword | Description |
|---|---|
| `TCORR` | Telluric correction applied (True) |
| `TCMODEL` | Name of the PSG model file used |
| `TCAIRM` | Airmass applied |
| `TCAMFIT` | Whether airmass was fitted to the data |
| `TCPWV` | PWV perturbation applied (fixed, not fitted) |

---

## Diagnostic Plots

```python
from telluric_pipeline import (
    plot_raw_psg_response,
    plot_molecular_species,
    plot_vs_airmass,
    plot_am_dlam_heatmap,
)

# Three-panel Raw_PSG_H2O plot (native → R=25k → R=8k)
plot_raw_psg_response(cfg, am=1.5, R_mid=25_000, R_inst=8_000)

# Per-species transmission at instrument resolution
plot_molecular_species(lam_grid, mol_tuple, R_inst=8_000)

# Transmission vs airmass from the grid
plot_vs_airmass(interp, lam_grid, surface_T=280.0)

# Chi-square landscape in (airmass, dlam) space
plot_am_dlam_heatmap(interp, wave_obs, flux_obs, sigma_obs,
                     fit_mask, lam_grid, R_inst=8_000)
```

---

## HISPEC Channel Boundaries

| Channel | Bands | Range | Fiber |
|---|---|---|---|
| BSPEC | y + J | 980–1400 nm | Silica |
| RSPEC | H + K | 1400–2500 nm | ZBLAN |

> **Note:** The 1400 nm boundary is estimated from the HISPEC fiber delivery subsystem design (Jovanovic et al. 2024, SPIE 13096) and is subject to change before commissioning. Confirm with the HISPEC team before using in production.

---

## File Structure

```
telluric_pipeline.py      # main pipeline module
earth_cfg.txt             # PSG Earth atmosphere template (not tracked)
average_psg_model.fits    # pre-generated average model (generated on first run)
telluric_grid.h5          # cached transmission grid (generated on first run)
telluric_grid/            # per-call PSG response cache (.dat + .cols files)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Interpolation, convolution, optimization |
| `matplotlib` | Diagnostic plots |
| `h5py` | HDF5 grid storage |
| `requests` | PSG API calls |
| `astropy` | FITS I/O |

---

## Related Work

- [KPIC Pipeline](https://github.com/kpicteam/kpic_pipeline) — pathfinder instrument pipeline, `wavecal.py` informed the `scale_psg` interface
- [HISPEC DRP](https://oirlab.github.io/HISPEC_DRP/) — the DRP this primitive is designed to slot into
- [PSG](https://psg.gsfc.nasa.gov) — NASA Planetary Spectrum Generator
- MOLECFIT, TelFit, Xtellcor — professional telluric correction pipelines that informed the PWV-fixed design

---

## Status

This module is under active development as part of the HISPEC DRP telluric primitive (Goal 1 of the project roadmap). Testing against real HISPEC/LIGER FITS files is pending instrument commissioning.

- [x] PSG interface with sub-band stitching and disk caching
- [x] Beer-Lambert airmass grid (HDF5)
- [x] Average PSG model FITS export/loader
- [x] DRP-facing apply function with header airmass and optional fitting
- [x] Flexible FITS spectrum reader
- [x] BSPEC/RSPEC channel trimming
- [ ] Validate against real HISPEC/LIGER FITS files
- [ ] Atmospheric variability grid (Goal 3, TBD)
