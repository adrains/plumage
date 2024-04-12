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

Note that the trained three and four label _Cannon_ models themselves from [Rains+2024](https://doi.org/10.1093/mnras/stae560) are available in the models folder, and a general use script for using these models to generate spectra and predict labels is available as scripts_cannon/example_cannon_script.py. This script is separate from the sequence outlined above, and is the thing to look at if you want to play around with the Cannon. Please get in touch if you have questions!

## Dependencies
For full functionality, the non-anaconda dependencies are:
- [pystan2](https://github.com/stan-dev/pystan2)
- [M_-M_K-](https://github.com/awmann/M_-M_K-)
- [pIDLy](https://github.com/anthonyjsmith/pIDLy)
- [Lightkurve](https://docs.lightkurve.org/)
- [batman](https://lkreidberg.github.io/batman/docs/html/index.html)
- [dustmaps](https://dustmaps.readthedocs.io/en/latest/)

However, if you just want to use the Cannon, pystan2 is technically (i.e. functionally) the only thing required.
