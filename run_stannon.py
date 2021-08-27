"""Script to train and cross validate a Cannon model
"""
import numpy as np
import plumage.utils as utils
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon
import stannon.plotting as splt

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
suppress_output = True
use_br_combined = False
normalise_spectra = True
add_photometry = False
do_cross_validation = False

# Whether to do sigma clipping using trained Cannon model
do_iterative_bad_px_masking = True
flux_sigma_to_clip = 5

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
training_set_flux, training_set_ivar, bad_px_mask = stannon.prepare_fluxes(
    spec_std_br[std_mask],
    e_spec_std_br[std_mask],)

wls = wave_std_br

# Construct standard mask to mask emission regions
adopted_wl_mask = spec.make_wavelength_mask(
    wls,
    mask_emission=True,
    mask_sky_emission=False,
    mask_edges=True,)

# Enforce minimum wavelength
adopted_wl_mask = np.logical_and(adopted_wl_mask, wls > wl_min)

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
    training_variances=label_var,
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask,)

# Train model
print("\nRunning initial training...")
sm.train_cannon_model(suppress_output=suppress_output)

# If we run the iterative bad px masking, train again afterwards
if do_iterative_bad_px_masking:
    print("\nRunning iterative sigma clipping for bad px...")
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=flux_sigma_to_clip)
    sm.train_cannon_model(suppress_output=suppress_output)

# Predict and plot
if do_cross_validation:
    sm.run_cross_validation()

    labels_pred = sm.cross_val_labels

# Just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(model_save_path)

#------------------------------------------------------------------------------
# Diagnostic Plotting
#------------------------------------------------------------------------------
# Work out uncertainties
label_pred_std = np.nanstd(label_values - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*label_pred_std))

# Plot diagnostic plots
splt.plot_label_recovery(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),)

# Save theta coefficients - one for each WiFeS arm
splt.plot_theta_coefficients(
    sm,
    x_lims=(3500,5400),
    y_s2_lims=(-0.001, 0.01),
    x_ticks=(200,100),
    label="b") 

splt.plot_theta_coefficients(
    sm,
    x_lims=(5400,7000),
    y_s2_lims=(-0.0005, 0.005),
    x_ticks=(200,100),
    label="r")

# Plot comparison of observed vs model spectra. Here we've picked a set of 
# spectral types at approximately low, high, and solar [Fe/H] for testing.
source_ids = [
    "2595284016771502080",      # M5, +, LHS 3799
    "2640434056928150400",      # M5, 0, GJ 1286
    "2358524597030794112",      # M5, -, PM J01125-1659

    "2603090003484152064",      # M3, +, GJ 876
    "4508377078422114944",      # M4, 0, GJ 4065
    "4472832130942575872",      # M4, -, Gl 699

    "2910909931633597312",      # M3, +, LP 837-53
    "3184351876391975808",      # M2, 0, Gl 173
    "2979590513145784192",      # M2, -, Gl 180

    "145421309108301184",       # K8, +, Gl 169
    "2533723464155234176",      # K8, 0, Gl 56.3 B
    "1244644727396803584",      # K8, -, Gl 525
]
    #"3796072592206250624",      # M4, GJ 447
    #"3101920046552857728",      # M2, Gl 250 B
    #"3738099879558957952",      # M1, Gl 514
    #"3339921875389105152",      # K8, Gl 208
    #"3057712188691831936",      # K7, Gl 282B


splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    source_ids=source_ids,
    x_lims=(3500,5400),
    fn_label="b",)

splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    source_ids=source_ids,
    x_lims=(5400,7000),
    fn_label="r",)

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
tess_flux, tess_ivar, tess_bad_px_mask = stannon.prepare_fluxes(
    spec_tess_br,
    e_spec_tess_br,)

tess_wls = wave_tess_br

# Add photometry
if add_photometry:
    tess_wls = np.concatenate([tess_wls, phot_wls])
    tess_flux = np.concatenate(
        [tess_flux, obs_join_tess[abs_mags].values], axis=1)
    tess_ivar = np.concatenate(
        [tess_ivar, 1/obs_join_tess[e_abs_mags].values**2], axis=1)

# Predict
tess_labels_pred, tess_errs_all, tess_chi2_all = sm.infer_labels(
    tess_flux[:,sm.adopted_wl_mask],
    tess_ivar[:,sm.adopted_wl_mask])

# Plot CMD
splt.plot_cannon_cmd(
    benchmark_colour=obs_join[std_mask]["Bp-Rp"],
    benchmark_mag=obs_join[std_mask]["K_mag_abs"],
    benchmark_feh=fehs[:,0],
    science_colour=obs_join_tess["Bp-Rp"],
    science_mag=obs_join_tess["K_mag_abs"],)

# Plot Kiel Diagram for results
splt.plot_kiel_diagram(
    teffs=tess_labels_pred[:,0],
    e_teffs=np.repeat(label_pred_std[0], len(tess_labels_pred)),
    loggs=tess_labels_pred[:,1],
    e_loggs=np.repeat(label_pred_std[1], len(tess_labels_pred)),
    fehs=tess_labels_pred[:,2],
    label="science",)

# And one for the benchmarks
splt.plot_kiel_diagram(
    teffs=label_values[:,0],
    e_teffs=label_var[:,0]**2,
    loggs=label_values[:,1],
    e_loggs=label_var[:,1]**2,
    fehs=label_values[:,2],
    label="benchmark",)
