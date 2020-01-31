"""
"""
import numpy as np
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon

label_names = ["teff", "logg", "feh"]

suppress_output = True
use_br_combined = False
normalise_spectra = True

px_min = 0
px_max = 1000#len(ref_wl_all)

# Import template spectra
if use_br_combined:
    n_ref = 156
    #n_ref = 624

    label = "BR_synth_test"
    #label = "BR_synth_test_norm"
    #label = "BR_synth_no_giants_norm"

    ref_wl_all, ref_spec, ref_params_all = synth.load_synthetic_templates(
        n_ref,
        label)

    wl_2d = np.tile(ref_wl_all, n_ref).reshape((n_ref,len(ref_wl_all))) 
    ref_spec_all = np.stack((wl_2d, ref_spec),axis=2).swapaxes(1,2)

else:
    ref_params_all, ref_spec_all = synth.load_synthetic_templates_legacy(
        "templates/templates_synth")

# Normalise template spectra
if normalise_spectra:
    ref_spec_norm = spec.normalise_spectra(ref_spec_all)
else:
    ref_spec_norm = ref_spec_all

# Mask wavelengths
ref_spec_masked = spec.mask_wavelengths(ref_spec_norm)

# Add "uncertainties", enforce limits
param_lims = {"teff":None,"logg":None,"feh":None,"vsini":None}
ref_wl, ref_fluxes, ref_ivar, ref_params  = stannon.prepare_synth_training_set(
    ref_spec_masked,
    ref_params_all,
    param_lims=param_lims,
    drop_vsini=True,)

# Make model
sm = stannon.Stannon(ref_fluxes, ref_ivar, ref_params, label_names, ref_wl, 
                     "basic")
# Setup model, train
sm.initialise_pixel_mask(px_min, px_max)
sm.train_cannon_model(suppress_output=suppress_output)

# Predict and plot
labels_pred, errs_all, chi2_all = sm.infer_labels(sm.masked_data, 
                                                  sm.masked_data_ivar)
sm.plot_label_comparison(sm.training_labels, labels_pred)
sm.plot_theta_coefficients() 