"""
"""
import numpy as np
import plumage.utils as utils
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon
import matplotlib.pyplot as plt 

suppress_output = True
use_br_combined = False
normalise_spectra = True

px_min = 0
px_max = None

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
wave_b = utils.load_fits_image_hdu("rest_frame_wave", "std", arm="b")
spec_norm_b = utils.load_fits_image_hdu("rest_frame_spec_norm", "std", arm="b")
e_spec_norm_b = utils.load_fits_image_hdu("rest_frame_sigma_norm", "std", arm="b")

wave_r = utils.load_fits_image_hdu("rest_frame_wave", "std", arm="r")
spec_norm_r = utils.load_fits_image_hdu("rest_frame_spec_norm", "std", arm="r")
e_spec_norm_r = utils.load_fits_image_hdu("rest_frame_sigma_norm", "std", arm="r")

# Load in RV corrected TESS spectra
tess_wave_b = utils.load_fits_image_hdu("rest_frame_wave", "tess", arm="b")
tess_spec_norm_b = utils.load_fits_image_hdu("rest_frame_spec_norm", "tess", arm="b")
e_tess_spec_norm_b = utils.load_fits_image_hdu("rest_frame_sigma_norm", "tess", arm="b")

tess_wave_r = utils.load_fits_image_hdu("rest_frame_wave", "tess", arm="r")
tess_spec_norm_r = utils.load_fits_image_hdu("rest_frame_spec_norm", "tess", arm="r")
e_tess_spec_norm_r = utils.load_fits_image_hdu("rest_frame_sigma_norm", "tess", arm="r")

# Table joins
obs_join = obs_std.join(std_info, "source_id", rsuffix="_info")

obs_join_tess = obs_tess.join(tess_info, "source_id", rsuffix="_info")

#------------------------------------------------------------------------------
# Photometry Definitions
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Setup training set
#------------------------------------------------------------------------------
# Parameter limits
teff_lims = (2500, 4500)
logg_lims = (4, 6.0)
feh_lims = (-1.25, 0.75)

# Get the parameters
label_names = ["teff_m15", "logg_m15", "feh_m15"]

std_mask = ~np.isnan(obs_join["teff_m15"])

#obs_mask = [sid in std_info[std_mask].index for sid in obs_std.index]

label_values = obs_join[std_mask][label_names].values

# Prepare fluxes
wls, training_set_flux, training_set_ivar = spec.prepare_spectra(
    wave_b,
    spec_norm_b[std_mask],
    e_spec_norm_b[std_mask],
    wave_r,
    spec_norm_r[std_mask],
    e_spec_norm_r[std_mask],
    use_both_arms=True,
    remove_blue_overlap=True)

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

# Print
std = np.std(label_values - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*std))

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
tess_wls, tess_flux, tess_ivar = spec.prepare_spectra(
    tess_wave_b,
    tess_spec_norm_b,
    e_tess_spec_norm_b,
    tess_wave_r,
    tess_spec_norm_r,
    e_tess_spec_norm_r,
    use_both_arms=True,
    remove_blue_overlap=True)

# Add photometry
tess_wls = np.concatenate([tess_wls, phot_wls])
tess_flux = np.concatenate(
    [tess_flux, obs_join_tess[abs_mags].values], axis=1)
tess_ivar = np.concatenate(
    [tess_ivar, 1/obs_join_tess[e_abs_mags].values**2], axis=1)

# Predict
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
plt.savefig("plots/tess_cannon_keel.pdf", dpi=200)
