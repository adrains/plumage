"""Example script for using the Cannon models from Rains+24.

https://ui.adsabs.harvard.edu/abs/2024MNRAS.529.3171R/abstract

Rains+24 used PyStan v2.19.1.1. This code is not compatible with PyStan 3+.

Please get in touch if you have questions!
"""
import matplotlib.pyplot as plt
import stannon.stannon as stannon

# Load in an existing Cannon model
model_3L = "models/stannon_model_basic_3label_5024px_teff_logg_feh.fits"
model_4L = "models/stannon_model_basic_4label_5024px_teff_logg_feh_Ti_Fe.fits"
sm = stannon.load_model(model_4L)

# The model can then be used to predict stellar labels. Here we are simply
# inferring labels for the spectra used to originally train the model. Note
# that new spectra should be pseudo-continuum normalised in the same manner as
# the training sample, *and* be convolved to the same resolution as WiFeS
# (R~3000 for 4000 < λ 5400 Å, R~7000 for 5400 < λ 7000 Å). The normalisation
# can be done via stannon.prepare_cannon_spectra_normalisation (which behind
# the scenes is using plumage.spectra.gaussian_normalise_spectra), and the
# convolution via e.g. the instrBroadGaussFast function from PyAstronomy
# (https://pyastronomy.readthedocs.io/en/latest/index.html). Inferred labels
# should be corrected for observed offsets, and statistical uncertainties
# should be added in quadrature with the adopted precision values from Rains+24
# (see Section 3.4, Figure 6, and Figure 8).
test_fluxes = sm.training_data.copy()
test_fluxes[sm.bad_px_mask] = 1.0

test_ivars = sm.training_data_ivar.copy()
test_ivars[sm.bad_px_mask] = 1E4

pred_label_values, pred_label_sigmas_stat, chi2_all = sm.infer_labels(
    test_data=test_fluxes[:,sm.adopted_wl_mask],
    test_data_ivars=test_ivars[:,sm.adopted_wl_mask])

# Arbitrary spectra can also be generated given a set of stellar labels. Note
# that the Cannon extrapolates poorly, so be careful using this near (and
# especially beyond) the bounds of the training sample. The labels are (Teff, 
# logg, and [Fe/H]) for the 3 label model, with the four label model instead
# having (Teff, logg, [Fe/H], [Ti/Fe]).
params_3L = [3500, 4.75, 0.0,]
params_4L = [3500, 4.75, 0.0, 0.0]
model_flux = sm.generate_spectra(params_4L)

# Plot up a quick figure to visualise our new spectrum!
fig, ax = plt.subplots(1, 1, figsize=(12,3))
ax.plot(sm.wavelengths[sm.adopted_wl_mask], model_flux, linewidth=0.5,)
ax.set_xlabel(r"Wavelength (Å)")
ax.set_ylabel("Flux")
plt.tight_layout()