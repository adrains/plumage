"""Script to train and cross validate a Cannon model
"""
import numpy as np
import pandas as pd
import plumage.utils as utils
import stannon.stannon as stannon
import stannon.plotting as splt
import stannon.tables as stab
from numpy.polynomial.polynomial import Polynomial

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
# Whether Cannon should suppress ouput (recommended True unless debugging)
suppress_output = True

# Whether to run leave-one-out cross validation on Cannon model
do_cross_validation = True

# Whether to do sigma clipping using trained Cannon model. If True, an initial
# Cannon model is trained and its model spectra are used to sigma clip bad 
# pixels to not be considered for the subsequently trained and adopted model.
do_iterative_bad_px_masking = True
flux_sigma_to_clip = 5

# Normalisation - using using a Gaussian smoothed version of the spectrum, or 
# a much simpler polynomial fit. Only wavelengths > than wl_min_normalisation
# will be considered during either approach to avoid low-SNR blue pixels for
# the coolest stars. TODO: save these parameters in the Cannon model.
do_gaussian_spectra_normalisation = True
wl_min_normalisation = 4000
wl_broadening = 50
poly_order = 4

# Minimum and maximum wavelengths for Cannon model
wl_min_model = 4000
wl_max_model = 7000

# The Cannon model to use - either the 'basic' traditional Cannon model, or a
# model with label uncertainties. If modelling abundances, the version with
# label uncertainties should be used. Either 'basic' or 'label_uncertainties'.
model_type = "label_uncertainties"
use_label_uniform_variances = False

model_save_path = "spectra"
std_label = "cannon"

# Whether to fit for abundances from Montes+18. Not recommended to fit > 1-2.
# Available options: Na, Mg, Al, Si, Ca, Sc, Ti, V, Cr, Mn, Co, Ni
# Select as e.g.["X_H",..] or leave empty to not use abundances.
abundance_labels = ["T_H"]

label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = len(label_names)

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Import literature info
std_info = utils.load_info_cat(
    "data/std_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)

# Load results table
obs_std = utils.load_fits_table("OBS_TAB", std_label, path="spectra")

# Load in RV corrected standard spectra
wls = utils.load_fits_image_hdu("rest_frame_wave", std_label, arm="br")
spec_std_br = utils.load_fits_image_hdu("rest_frame_spec", std_label, arm="br")
e_spec_std_br = utils.load_fits_image_hdu("rest_frame_sigma", std_label, arm="br")

#------------------------------------------------------------------------------
# Fluxes, masking, and normalisation
#------------------------------------------------------------------------------
fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wls,
        spectra=spec_std_br,
        e_spectra=e_spec_std_br,
        wl_min_model=wl_min_model,
        wl_max_model=wl_max_model,
        wl_min_normalisation=wl_min_normalisation,
        wl_broadening=wl_broadening,
        do_gaussian_spectra_normalisation=do_gaussian_spectra_normalisation,
        poly_order=poly_order)

#------------------------------------------------------------------------------
# Setup training labels
#------------------------------------------------------------------------------
# Crossmatch fitted params with literature info
obs_join = obs_std.join(std_info, "source_id", rsuffix="_info")

# And crossmatch this with our set of CPM abundances
montes18_abund = pd.read_csv(
    "data/montes18_primaries.tsv",
    delimiter="\t",
    dtype={"source_id":str},
    na_filter=False)
montes18_abund.set_index("source_id", inplace=True)
obs_join = obs_join.join(montes18_abund, "source_id", rsuffix="_m18")

# Import Montes+18 abundance trends to have a better naive guess at abundances
# for stars with unknown abundances
montes18_abund_trends = pd.read_csv("data/montes18_abundance_trends.csv") 

# Prepare our labels
label_values_all, label_sigma_all, std_mask, label_sources = \
    stannon.prepare_labels(
        obs_join=obs_join,
        n_labels=n_labels,
        abundance_labels=abundance_labels,
        abundance_trends=montes18_abund_trends)

# Grab the IDs of the selected stars
benchmark_source_ids = obs_join[std_mask].index.values

label_values = label_values_all[std_mask]
label_var = label_sigma_all[std_mask]**0.5

# Test with uniform variances
if use_label_uniform_variances:
    label_var = 1e-3 * np.ones_like(label_values)

# Finally, select only fluxes from those stars with appropriate labels
training_set_flux = fluxes_norm[std_mask]
training_set_ivar = ivars_norm[std_mask]
training_bad_px_mask = bad_px_mask[std_mask]

#------------------------------------------------------------------------------
# Make and train model
#------------------------------------------------------------------------------
# Diagnostic summary
print("\n\n", "%"*80, "\n", sep="")
print("\tRunning Cannon model:\n\t", "-"*21, sep="")
print("\tmodel: \t\t\t = {}".format(model_type))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(label_names)))
print("\tlabels: \t\t = {}".format(label_names))
print("\tGaussian Normalisation:\t = {}".format(do_gaussian_spectra_normalisation))
if do_gaussian_spectra_normalisation:
    print("\twl broadening: \t\t = {:0.0f} A".format(wl_broadening))
else:
    print("\tpoly order: \t\t = {:0.0f}".format(poly_order))
print("\tcross validation: \t = {}".format(do_cross_validation))
print("\titerative masking: \t = {}".format(do_iterative_bad_px_masking))
print("\n", "%"*80, "\n\n", sep="")

# Make model
sm = stannon.Stannon(
    training_data=training_set_flux,
    training_data_ivar=training_set_ivar,
    training_labels=label_values,
    training_ids=benchmark_source_ids,
    label_names=label_names,
    wavelengths=wls,
    model_type=model_type,
    training_variances=label_var,
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=training_bad_px_mask,)

# Train model
print("\nRunning initial training with {} benchmarks...".format(len(label_values)))
sm.train_cannon_model(suppress_output=suppress_output)

# If we run the iterative bad px masking, train again afterwards
if do_iterative_bad_px_masking:
    print("\nRunning iterative sigma clipping for bad px...")
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=flux_sigma_to_clip)
    sm.train_cannon_model(suppress_output=suppress_output)

# Run cross validation
if do_cross_validation:
    sm.run_cross_validation()

    labels_pred = sm.cross_val_labels

# ...or just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(model_save_path)

#------------------------------------------------------------------------------
# Diagnostics and plotting
#------------------------------------------------------------------------------
# Work out uncertainties
label_pred_std = np.nanstd(label_values - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*label_pred_std))

fn_label = "_{}_{}_label_{}".format(
    model_type, len(label_names), "_".join(label_names))

# Label recovery for Teff, logg, and [Fe/H]
splt.plot_label_recovery(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    obs_join=obs_join[std_mask],
    fn_suffix=fn_label,
    teff_lims=(2750,4300),
    teff_ticks=(500,250,200,100),
    logg_ticks=(0.25,0.125,0.2,0.1),
    feh_lims=(-1.1,0.65),
    feh_ticks=(0.5,0.25,0.4,0.2),)

# Plot recovery for interferometric Teff, M+15 [Fe/H], RA+12 [Fe/H], CPM [Fe/H]
splt.plot_label_recovery_per_source( 
    label_values=sm.training_labels, 
    e_label_values=sm.training_variances**2, 
    label_pred=labels_pred, 
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L), 
    obs_join=obs_join[std_mask],
    fn_suffix=fn_label,)

# And finally plot the label recovery for any abundances we might be using
splt.plot_label_recovery_abundances(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    obs_join=obs_join[std_mask],
    fn_suffix=fn_label,
    abundance_labels=abundance_labels,
    feh_lims=(-0.65,0.65),
    feh_ticks=(0.4,0.2,0.2,0.1))

# Save theta coefficients - one for each WiFeS arm
splt.plot_theta_coefficients(
    sm,
    teff_scale=1.0,
    x_lims=(wl_min_model,5400),
    y_spec_lims=(0,2.25),
    y_theta_lims=(-0.25,0.25),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    label="b",
    linewidth=0.9,
    alpha=0.8,
    fn_suffix=fn_label,)

splt.plot_theta_coefficients(
    sm,
    teff_scale=1.0,
    x_lims=(5400,wl_max_model),
    y_spec_lims=(0,2.25),
    y_theta_lims=(-0.12,0.12),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    label="r",
    linewidth=0.9,
    alpha=0.8,
    fn_suffix=fn_label,)

# Plot comparison of observed vs model spectra. Here we've picked a set of 
# spectral types at sub-solar, solar, and super-solar [Fe/H] for visualisation.
representative_stars_source_ids = [
    "5853498713160606720",      # M5.5 +, GJ 551
    "2467732906559754496",      # M5.1, 0, GJ 3119
    "2358524597030794112",      # M5, -, PM J01125-1659
    "3195919322830293760",      # M5, Gl 166 C
    "6322070093095493504",      # M2, GJ 581
    "2603090003484152064",      # M3, +, GJ 876
    "4472832130942575872",      # M4, -, Gl 699
    "2910909931633597312",      # M3, +, LP 837-53
    "3184351876391975808",      # M2, 0, Gl 173
    "2739689239311660672",      # M0, Gl 908
    "145421309108301184",       # K8, +, Gl 169
    "4282578724832056576",      # M0.7, Gl 740
]

# Plot model spectrum performance for WiFeS blue band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=labels_pred,
    source_ids=representative_stars_source_ids,
    x_lims=(wl_min_model,5400),
    fn_label="b",)

# Plot model spectrum performance for WiFeS red band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=labels_pred,
    source_ids=representative_stars_source_ids,
    x_lims=(5400,wl_max_model),
    fn_label="r",)

# Do the same, but across all wavelengths and for all stars
bp_rp_order = np.argsort(obs_join[std_mask]["Bp-Rp"])
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=labels_pred,
    source_ids=obs_join[std_mask].index[bp_rp_order],
    x_lims=(wl_min_model,wl_max_model),
    fn_label="d",
    data_label="benchmark",)

#------------------------------------------------------------------------------
# Tables
#------------------------------------------------------------------------------
# Tabulate our adopted benchmark parameters
stab.make_table_benchmark_overview(
    obs_tab=obs_join[std_mask],
    labels=sm.training_labels,
    e_labels=sm.training_variances**2,
    label_sources=label_sources[std_mask],
    abundance_labels=abundance_labels,)