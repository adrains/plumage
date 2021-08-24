"""Script to train and cross validate a Cannon model
"""
import numpy as np
import plumage.utils as utils
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon
import matplotlib.pyplot as plt 

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
suppress_output = True
use_br_combined = False
normalise_spectra = True
add_photometry = False
do_cross_validation = True

wl_min = 0

poly_order = 4
model_type = "basic"
#model_type = "label_uncertainties"

model_save_path = "spectra"

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Import literature info for both standards and TESS targets
std_info = utils.load_info_cat(
    "data/std_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)
tess_info = utils.load_info_cat(
    "data/tess_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)

# Load results tables for both standards and TESS targets
obs_std = utils.load_fits_table("OBS_TAB", "std", path="spectra")
obs_tess = utils.load_fits_table("OBS_TAB", "tess", path="spectra")

# Load in RV corrected standard spectra
wave_std_br = utils.load_fits_image_hdu("rest_frame_wave", "std", arm="br")
spec_std_br = utils.load_fits_image_hdu("rest_frame_spec", "std", arm="br")
e_spec_std_br = utils.load_fits_image_hdu("rest_frame_sigma", "std", arm="br")

spec_std_br, e_spec_std_br = spec.normalise_spectra(
    wave_std_br,
    spec_std_br,
    e_spec_std_br,
    poly_order=poly_order)

# Load in RV corrected TESS spectra
wave_tess_br = utils.load_fits_image_hdu("rest_frame_wave", "tess", arm="br")
spec_tess_br = utils.load_fits_image_hdu("rest_frame_spec", "tess", arm="br")
e_spec_tess_br = utils.load_fits_image_hdu("rest_frame_sigma", "tess", arm="br")

spec_tess_br, e_spec_tess_br = spec.normalise_spectra(
    wave_tess_br,
    spec_tess_br,
    e_spec_tess_br,
    poly_order=poly_order,)

# Table joins
obs_join = obs_std.join(std_info, "source_id", rsuffix="_info")

obs_join_tess = obs_tess.join(tess_info, "source_id", rsuffix="_info")

#------------------------------------------------------------------------------
# Setup training set
#------------------------------------------------------------------------------
# Get the parameters
label_names = ["teff_synth", "logg_synth"]
e_label_names = ["e_teff_synth", "e_logg_synth"]

#std_mask = ~np.isnan(obs_join["teff_m15"])
std_mask = np.logical_or(
    ~np.isnan(obs_join["teff_m15"]),
    ~np.isnan(obs_join["teff_ra12"]))

# Preferentially use Mann+15 metallcities
fehs = np.atleast_2d([row["feh_ra12"] if np.isnan(row["feh_m15"]) else row["feh_m15"]
        for sid, row in obs_join[std_mask].iterrows()]).T
e_fehs = np.atleast_2d([row["e_feh_ra12"] if np.isnan(row["e_feh_m15"]) else row["e_feh_m15"]
        for sid, row in obs_join[std_mask].iterrows()]).T

# Prepare label values
label_values = np.hstack([obs_join[std_mask][label_names].values, fehs])
label_var = np.hstack([obs_join[std_mask][e_label_names].values, e_fehs])**0.5

# Test with uniform variances
#label_var = 1e-3 * np.ones_like(label_values)

label_names = ["teff", "logg", "feh"]

# Prepare fluxes
wls, training_set_flux, training_set_ivar = spec.prepare_cannon_spectra(
    wave_std_br,
    spec_std_br[std_mask],
    e_spec_std_br[std_mask],
    wl_min=wl_min,)

#------------------------------------------------------------------------------
# Photometry (optional)
#------------------------------------------------------------------------------
if add_photometry:
    phot_wls = np.array([
        4500,
        7500,
        6000,
        12350,
        16620,
        21590,
    ])

    abs_mags = np.array([
        "Bp_mag_abs",
        "Rp_mag_abs",
        "G_mag_abs",
        "J_mag_abs",
        "H_mag_abs",
        "K_mag_abs",
        ], dtype=object)

    e_abs_mags = np.array([
        "e_Bp_mag_abs",
        "e_Rp_mag_abs",
        "G_mag_abs",
        "e_J_mag_abs",
        "e_H_mag_abs",
        "e_K_mag_abs",
        ], dtype=object)

    # Add photometry
    wls = np.concatenate([wls, phot_wls])
    training_set_flux = np.concatenate(
        [training_set_flux, obs_join[std_mask][abs_mags].values], axis=1)
    training_set_ivar = np.concatenate(
        [training_set_ivar, 1/obs_join[std_mask][e_abs_mags].values**2], axis=1)

#------------------------------------------------------------------------------
# Make and Train model
#------------------------------------------------------------------------------
# Make model
sm = stannon.Stannon(
    training_data=training_set_flux,
    training_data_ivar=training_set_ivar,
    training_labels=label_values, 
    label_names=label_names,
    wavelengths=wls,
    model_type=model_type,
    training_variances=label_var,)

# Train model
sm.train_cannon_model(suppress_output=suppress_output)

# Predict and plot
if do_cross_validation:
    sm.run_cross_validation()

    labels_pred = sm.cross_val_labels

# Just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Work out uncertainties
label_pred_std = np.nanstd(label_values - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*label_pred_std))

# Plot diagnostic plots
sm.plot_label_comparison(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),)
sm.plot_theta_coefficients() 

# Save model
sm.save_model(model_save_path)

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
tess_wls, tess_flux, tess_ivar = spec.prepare_cannon_spectra(
    wave_tess_br,
    spec_tess_br,
    e_spec_tess_br,
    wl_min=wl_min,)

# Add photometry
if add_photometry:
    tess_wls = np.concatenate([tess_wls, phot_wls])
    tess_flux = np.concatenate(
        [tess_flux, obs_join_tess[abs_mags].values], axis=1)
    tess_ivar = np.concatenate(
        [tess_ivar, 1/obs_join_tess[e_abs_mags].values**2], axis=1)

# Predict
tess_labels_pred, tess_errs_all, tess_chi2_all = sm.infer_labels(
    tess_flux[:,sm.pixel_mask],
    tess_ivar[:,sm.pixel_mask])

# Plot
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
plt.savefig("plots/tess_cannon_keel.pdf", dpi=200)
