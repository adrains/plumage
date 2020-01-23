"""
"""
import numpy as np
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon

label_names = ["teff", "logg", "feh"]


n_ref = 156
label = "BR_synth_test"

# Import template spectra
ref_wl_all, ref_spec, ref_params_all = synth.load_synthetic_templates(
    n_ref,
    label)

wl_2d = np.tile(ref_wl_all, n_ref).reshape((n_ref,len(ref_wl_all))) 
ref_spec_all = np.stack((wl_2d, ref_spec),axis=2).swapaxes(1,2)

px_min = 0
px_max = len(ref_wl_all)

# Normalise template spectra
ref_spec_norm = spec.normalise_spectra(ref_spec_all)

# Mask wavelengths
ref_spec_masked = spec.mask_wavelengths(ref_spec_norm)

# Add "uncertainties", enforce limits
param_lims = {"teff":None,"logg":None,"feh":None,"vsini":(0,1)}
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