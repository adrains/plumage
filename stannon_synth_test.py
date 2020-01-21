"""
"""
import numpy as np
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon

label_names = ["teff", "logg", "feh"]
px_min = 0
px_max = 100

# Import template spectra
ref_params_all, ref_spec_all = synth.load_synthetic_templates(setting="R7000")

# Normalise template spectra
ref_spec_norm = spec.normalise_spectra(ref_spec_all)

# Mask wavelengths
ref_spec_masked = spec.mask_wavelengths(ref_spec_norm)

# Add "uncertainties", enforce limits
param_lims = {"teff":(3000,4000),"logg":None,"feh":None,"vsini":(0,1)}
ref_wl, ref_fluxes, ref_ivar, ref_params  = stannon.prepare_synth_training_set(
    ref_spec_masked,
    ref_params_all,
    param_lims=param_lims,
    drop_vsini=True,)

# Make model
sm = stannon.Stannon(ref_fluxes, ref_ivar, ref_params, label_names, ref_wl, 
                     "basic")
# Setup model, train
sm.whiten_labels()
sm.initialise_pixel_mask(px_min, px_max)
sm.train_cannon_model(suppress_output=True)

# Predict and plot
labels_pred, errs_all, chi2_all = sm.infer_labels(sm.masked_data, 
                                                  sm.masked_data_ivar) 
sm.plot_label_comparison(sm.training_labels, labels_pred)
sm.plot_theta_coefficients() 