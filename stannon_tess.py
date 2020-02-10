"""
"""
import numpy as np
import plumage.utils as utils
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon

label_names = ["teff", "logg", "feh"]

suppress_output = True
use_br_combined = False
normalise_spectra = True

px_min = 0
px_max = -1#len(ref_wl_all)

# Import standard spectra
obs_std, spec_std_b, spec_std_r = spec.load_pkl_spectra("standard", rv_corr=True)
obs_tess, spec_tess_b, spec_tess_r = spec.load_pkl_spectra("tess", rv_corr=True)

# Load in standards
standards = utils.load_standards()
standards.pop("herczeg")

#------------------------------------------------------------------------------
# Setup training set
#------------------------------------------------------------------------------
# Parameter limits
teff_lims = (2500, 4000)
logg_lims = (4, 6)
feh_lims = None

# Get the parameters
std_params_all = utils.consolidate_standards(
    standards, 
    force_unique=True,
    remove_standards_with_nan_params=True,
    teff_lims=teff_lims,
    logg_lims=logg_lims, 
    feh_lims=feh_lims,
    assign_default_uncertainties=True,
    force_solar_missing_feh=True)

# Prepare the training set
std_obs, std_spec_b, std_spec_r, std_params = utils.prepare_training_set(
    obs_std, 
    spec_std_b,
    spec_std_r, 
    std_params_all, 
    do_wavelength_masking=True
    )   

label_values = std_params[label_names].values
wls = np.concatenate((std_spec_b[0,0,:],std_spec_r[0,0,:]))

training_set_flux, training_set_ivar = utils.prepare_fluxes(
    std_spec_b, 
    std_spec_r, 
    use_both_arms=True)

# Make model
sm = stannon.Stannon(training_set_flux, training_set_ivar, label_values, label_names, wls, 
                     "basic")
# Setup model, train
sm.initialise_pixel_mask(px_min, px_max)
sm.train_cannon_model(suppress_output=suppress_output)

# Predict and plot
labels_pred, errs_all, chi2_all = sm.infer_labels(sm.masked_data, 
                                                  sm.masked_data_ivar)
sm.plot_label_comparison(sm.training_labels, labels_pred)
sm.plot_theta_coefficients() 