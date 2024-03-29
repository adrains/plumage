# plumage
## Overview
Code developed to work with (largely) cool dwarf spectra from the [WiFeS instrument](https://rsaa.anu.edu.au/observatories/instruments/wide-field-spectrograph-wifes) on the ANU 2.3 m Telescope. Functionality:
- Data reduction/preparation--flux calibration, RV determination (see [Žerjal+2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..938Z/abstract)), normalisation, etc.
- _Synthetic_ fitting for parameter determination (per [Rains+2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.5788R/abstract)) using MARCS model spectra (grid from [Nordlander+2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.488L.109N/abstract)) and photometry.
- Light curve fitting of TESS light curves (per [Rains+2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.5788R/abstract)).
- _Data-driven_ fitting for parameter determination using the [_Cannon_](https://ui.adsabs.harvard.edu/abs/2015ApJ...808...16N/abstract) (per [Rains+2024](https://doi.org/10.1093/mnras/stae560)).

## Spectra Preparation
1D WiFeS spectra should be extracted as individual fits files using the functions `extract_stellar_spectra_fits` and `extract_stellar_spectra_fits_all` in pywifes_utils.py on the [PyWiFeS repository](https://github.com/PyWiFeS/pipeline). These fits files maintain the header information from the telescope, and contain at least the unfluxed 1D spectra in units of counts, and optionally the PyWiFeS flux and telluric corrected spectra.

## Import and RVs
The following two scripts import/consolidate the 1D WiFeS spectra into a single fits file, and fit synthetic templates to determine radial velocities. If spectra are not yet fluxed, they can be corrected for the mean atmospheric/telescope/instrument response function here. Generates bad pixel masks using the residuals from the best template fit.
- scripts_reduction/import_spectra.py
- scripts_reduction/determine_rvs.py

## Synthetic Spectral Fitting
Performs up to a 4 term spectral fit in T<sub>eff</sub>, logg, [Fe/H], and m<sub>bol</sub> using synthetic MARCS spectra and bolometric corrections. MARCS spectra are interpolated using IDL routines run from python using the [`pidly`](https://github.com/anthonyjsmith/pIDLy) library, and bolometric corrections from [`bolometric-corrections`](https://github.com/casaluca/bolometric-corrections). Literature photometry required for the m<sub>bol</sub> fit, with Gaia, 2MASS, and SkyMapper currently supported. logg can be fixed to the value determined from empirical mass and radius relations.
- scripts_synth/synth_fit.py

Can also generate synthetic spectra for the sample at the literature stellar parameters for the sample.
- scripts_synth/get_lit_param_synth.py

Note that the synthetic MARCS grid used for fitting is not currently publicly available. 

## Light Curve Fitting
Uses the [`Lightkurve`](https://github.com/KeplerGO/lightkurve) package to download and work with TESS light curves, and the [`batman`](https://github.com/lkreidberg/batman) package to model the transits themselves. Given the known orbital period, and estimated M<sub>★</sub>, we apply a prior for the semi-major axis during fitting. Limb darkening coefficients are chosen based on previously fitted stellar parameters, and fitted R<sub>★</sub> is used to determine the unscaled R<sub>P</sub> and semi-major axis.
- scripts_lc/lightcurve_fit.py

## Data-Driven Fitting with the Cannon
The _Cannon_ is implemented using [pyStan 2](https://github.com/stan-dev/pystan2), and there are currently three (T<sub>eff</sub>, logg, [Fe/H]) and four (T<sub>eff</sub>, logg, [Fe/H], [Ti/Fe]) label models. The core _Cannon_ scripts (in order) are:
- scripts_cannon/prepare_stannon_training_sample.py
- scripts_cannon/train_stannon.py
- scripts_cannon/make_stannon_diagnostics.py
- scripts_cannon/compare_stannon_models.py
- scripts_cannon/run_stannon.py

With the various settings required by these scripts stored in scripts_cannon/cannon_settings.yml.

Note that the trained three and four label _Cannon_ models themselves from [Rains+2024](https://doi.org/10.1093/mnras/stae560) will also be uploaded soon (time of writing 22/02/2024) to coincide with the paper's publication. We'll also make available a more general-use script and advice on running this _Cannon_ model on arbitrary cool dwarf spectra not observed with WiFeS. 

## Other Details
### Output Format
All scripts update a single fits file, with extensions added for each different script:
```
0  WAVE_B         PrimaryHDU       (n_px_b,)   
1  SPEC_B         ImageHDU         (n_px_b, n_star)   
2  SIGMA_B        ImageHDU         (n_px_b, n_star)   
3  WAVE_R         ImageHDU         (n_px_r,)   
4  SPEC_R         ImageHDU         (n_px_b, n_star)   
5  SIGMA_R        ImageHDU         (n_px_b, n_star)   
6  OBS_TAB        BinTableHDU      n_star R x 61C   
7  BAD_PX_MASK_B  ImageHDU         (n_px_b, n_star)   
8  BAD_PX_MASK_R  ImageHDU         (n_px_r, n_star)   
9  SYNTH_FIT_B    ImageHDU         (n_px_b, n_star)   
10  SYNTH_FIT_R   ImageHDU         (n_px_r, n_star)
11  TRANSIT_FITS  BinTableHDU      n_toi R x 76C
```

### Literature Relations and Data
- Empirical absolute 2MASS K<sub>S</sub> cool dwarf radius relation from [Mann et al. 2015](https://ui.adsabs.harvard.edu/#abs/2016ApJ...819...87M/abstract)
- Empirical absolute 2MASS K<sub>S</sub> cool dwarf mass relation from [Mann et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...871...63M/abstract)
- Limb darkening coefficients for TESS bandpass [Claret 2017](https://ui.adsabs.harvard.edu/abs/2017A&A...600A..30C/abstract)

### Required Files
- ID crossmatch file, with columns [source_id, 2MASS_Source_ID, HD, TOI, bayer, other, program, subset, program2, subset2]. Used to identify science program, and crossmatch ID.
- Literature information file, containing Gaia photometry and parallaxes, 2MASS photometry, and optionally SkyMapper photometry
