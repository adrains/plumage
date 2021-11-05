"""Script to import and run a trained Cannon model on science data.
"""
import os
import numpy as np
import pandas as pd
import plumage.utils as utils
import plumage.plotting as pplt
import stannon.stannon as stannon
import stannon.plotting as splt
import stannon.tables as stab

#------------------------------------------------------------------------------
# Parameters and Setup
#------------------------------------------------------------------------------
# Normalisation settings
do_gaussian_spectra_normalisation = True
wl_min_normalisation = 4000
wl_broadening = 50
poly_order = 4

# Model settings
wl_min_model = 4000
wl_max_model = 7000
npx = 5024
model_type = "label_uncertainties"
abundance_labels = ["Ti_H"]
label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = int(len(label_names))

# Import model
model_name = "stannon_model_{}_{}label_{}px_{}.pkl".format(
    model_type, n_labels, npx, "_".join(label_names))

cannon_model_path = os.path.join("spectra", model_name)

sm = stannon.load_model(cannon_model_path)

# Adopted label uncertainties and systematics based on cross validation
# performance on the benchmark set. Quoted systematics are fit - lit, meaning
# that a positive systematic means the Cannon has *overestimated* the value 
# (and thus the systematic should be substracted). The temperature systematic 
# and uncertainty will be adopted from just the interferometric set, whereas
# logg and [Fe/H] will be taken from the complete sample. 
# TODO: adopted_label_uncertainties = sm.adopted_label_uncertainties
adopted_label_systematics = np.array([17.57, 0.0, 0.02, -0.03,])
adopted_label_uncertainties = np.array([59.15, 0.06, 0.12, 0.09])

# Bp-Rp cutoff for TESS targets to align with benchmark sample
bp_rp_cutoff = 1.7

# Rp cutoff in R_earth for planet [Fe/H] correlation
rearth_cutoff = 2.5

# And finally which dataset we want to run on
dataset = "tess"

#------------------------------------------------------------------------------
# Import science spectra, normalise, and prepare fluxes
#------------------------------------------------------------------------------
# Import literature info for benchmarks
std_info = utils.load_info_cat(
    "data/std_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)

obs_std = utils.load_fits_table("OBS_TAB", "cannon", path="spectra")
obs_join_std = obs_std.join(std_info, "source_id", rsuffix="_info")

# And for TESS targets
tess_info = utils.load_info_cat(
    "data/tess_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)

obs_tess = utils.load_fits_table("OBS_TAB", "tess", path="spectra")
obs_join_tess = obs_tess.join(tess_info, "source_id", rsuffix="_info")

# Import planet info
tic_info_all = utils.load_fits_table("TRANSIT_FITS", "tess", path="spectra")
tic_info = tic_info_all[~np.isnan(tic_info_all["rp_rstar_fit"])]

# Get masks for which targets are appropriate for science and training sample
tess_mask = np.logical_and(
    obs_join_tess["Bp-Rp"].values > bp_rp_cutoff,
    ~np.isnan(obs_join_tess["G_mag"].values))

std_mask = np.array([sid in sm.training_ids for sid in obs_join_std.index.values])

# Load in RV corrected TESS spectra
wave_tess_br = utils.load_fits_image_hdu("rest_frame_wave", "tess", arm="br")
spec_tess_br = utils.load_fits_image_hdu("rest_frame_spec", "tess", arm="br")
e_spec_tess_br = utils.load_fits_image_hdu("rest_frame_sigma", "tess", arm="br")

print("Running on {} sample".format(dataset))

# Setup datasets as appropriate
if dataset is "tess":
    fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
        stannon.prepare_cannon_spectra_normalisation(
            wls=wave_tess_br,
            spectra=spec_tess_br[tess_mask],
            e_spectra=e_spec_tess_br[tess_mask],
            wl_max_model=wl_max_model,
            wl_min_normalisation=wl_min_normalisation,
            wl_broadening=wl_broadening,
            do_gaussian_spectra_normalisation=do_gaussian_spectra_normalisation,
            poly_order=poly_order)

    info_cat = obs_join_tess
    dataset_mask = tess_mask
    star_name_col = "TOI"
    star_label_tab = "TOI"
    caption_unique = "TESS candidate planet host"

elif dataset is "benchmark":
    fluxes_norm = sm.training_data
    ivars_norm = sm.training_data_ivar
    bad_px_mask = sm.bad_px_mask
    adopted_wl_mask = sm.adopted_wl_mask

    info_cat = obs_join_std
    dataset_mask = std_mask
    star_name_col = "simbad_name"
    star_label_tab = "Star"
    caption_unique = "Benchmark star"

# Apply bad pixel mask
fluxes_norm[bad_px_mask] = 1
ivars_norm[bad_px_mask] = 0

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
# Predict labels
labels_pred, e_labels_pred, chi2_all = sm.infer_labels(
    fluxes_norm[:,adopted_wl_mask],
    ivars_norm[:,adopted_wl_mask])

# Correct labels for systematics
systematic_vector = np.tile(adopted_label_systematics, np.sum(dataset_mask))
systematic_vector = systematic_vector.reshape([np.sum(dataset_mask), n_labels])
labels_pred -= systematic_vector

# Create uncertainties vector. TODO: do this in quadrature with those output
# from infer_labels
cross_val_sigma = np.tile(adopted_label_uncertainties, np.sum(dataset_mask))
cross_val_sigma = cross_val_sigma.reshape([np.sum(dataset_mask), n_labels])
e_labels_pred = np.sqrt(e_labels_pred**2 + cross_val_sigma**2)


# Plot a joint CMD of benchmarks and science targets
splt.plot_cannon_cmd(
    benchmark_colour=obs_join_std[std_mask]["Bp-Rp"],
    benchmark_mag=obs_join_std[std_mask]["K_mag_abs"],
    benchmark_feh=sm.training_labels[:,2],
    science_colour=obs_join_tess["Bp-Rp"],
    science_mag=obs_join_tess["K_mag_abs"],)

# Plot Kiel Diagram for results
splt.plot_kiel_diagram(
    teffs=labels_pred[:,0],
    e_teffs=e_labels_pred[:,0],
    loggs=labels_pred[:,1],
    e_loggs=e_labels_pred[:,1],
    fehs=labels_pred[:,2],
    label=dataset,)

# Plot observed vs model spectra comparison for all stars sorted by Bp-Rp
bp_rp_order = np.argsort(info_cat[dataset_mask]["Bp-Rp"].values)
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=info_cat[dataset_mask],
    fluxes=fluxes_norm,
    bad_px_masks=bad_px_mask,
    labels_all=labels_pred,
    source_ids=info_cat[dataset_mask].index[bp_rp_order],
    x_lims=(wl_min_model,wl_max_model),
    fn_label="d",
    data_label=dataset,
    star_name_col=star_name_col)

caption = "{} parameter fits (corrected for systematics)".format(caption_unique)

# Make results table
stab.make_table_parameter_fit_results(
    obs_tab=info_cat[dataset_mask],
    label_fits=labels_pred,
    e_label_fits=e_labels_pred,
    abundance_labels=abundance_labels,
    break_row=61,
    star_label=(star_label_tab, star_name_col),
    table_label=dataset,
    caption=caption,)

#------------------------------------------------------------------------------
# Planet-star correlations
#------------------------------------------------------------------------------
# Planet-[Fe/H] correlations
if dataset is "tess":
    pplt.plot_planet_feh_correlation(
        obs_tab=obs_join_tess[tess_mask],
        tic_info=tic_info,
        host_star_fehs=labels_pred[:,2],
        giant_planet_rp=rearth_cutoff,
        use_default_bins=False,
        custom_nbins=5,)
