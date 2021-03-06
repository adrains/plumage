"""
"""
import numpy as np
import plumage.utils as utils
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon
import matplotlib.pyplot as plt 

label_names = ["teff", "logg", "feh"]

suppress_output = True
use_br_combined = False
normalise_spectra = True

px_min = 0
px_max = None

# Import standard spectra
obs_std_all, spec_std_b_all, spec_std_r_all = spec.load_pkl_spectra(
    "standard", 
    rv_corr=True)
obs_tess, spec_tess_b, spec_tess_r = spec.load_pkl_spectra("tess", rv_corr=True)

# Load in standards
standards = utils.load_standards()
standards.pop("herczeg")

#------------------------------------------------------------------------------
# Setup training set
#------------------------------------------------------------------------------
# Parameter limits
teff_lims = (2500, 4500)
logg_lims = (4, 6.0)
feh_lims = (-1.25, 0.75)

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
obs_std, spec_std_b, spec_std_r, std_params = stannon.prepare_training_set(
    obs_std_all, 
    spec_std_b_all,
    spec_std_r_all, 
    std_params_all, 
    )

label_values = std_params[label_names].values

wls, training_set_flux, training_set_ivar = spec.prepare_spectra(
    spec_std_b, 
    spec_std_r, 
    use_both_arms=True,
    remove_blue_overlap=True)

#------------------------------------------------------------------------------
# Make and Train model
#------------------------------------------------------------------------------
# Make model
sm = stannon.Stannon(training_set_flux, training_set_ivar, label_values, 
                     label_names, wls, "basic")
# Setup model, train
sm.initialise_pixel_mask(px_min, px_max)
sm.train_cannon_model(suppress_output=suppress_output)

# Predict and plot
labels_pred, errs_all, chi2_all = sm.infer_labels(sm.masked_data, 
                                                  sm.masked_data_ivar)
sm.plot_label_comparison(sm.training_labels, labels_pred, teff_lims, logg_lims, feh_lims)
sm.plot_theta_coefficients() 

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
tess_wls, tess_flux, tess_ivar = spec.prepare_spectra(
    spec_tess_b, 
    spec_tess_r, 
    use_both_arms=True,
    remove_blue_overlap=True)

tess_labels_pred, tess_errs_all, tess_chi2_all = sm.infer_labels(
    tess_flux[:,sm.pixel_mask],
    tess_ivar[:,sm.pixel_mask])

plt.figure()
plt.scatter(tess_labels_pred[:,0], 
            tess_labels_pred[:,1], 
            c=tess_labels_pred[:,2])
plt.plot(label_values[:,0], label_values[:,1], "*", color="black")
cb = plt.colorbar()
cb.set_label(r"[Fe/H]")
plt.xlim([4500,2900])
plt.ylim([5.5,4])
plt.xlabel(r"T$_{\rm eff}$")
plt.ylabel(r"$\log g$")
plt.tight_layout()
plt.show()
plt.savefig("plots/tess_ms.png", dpi=200)