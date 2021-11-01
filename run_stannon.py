"""Script to import and run a trained Cannon model on science data.
"""
import os
import numpy as np
import pandas as pd
import plumage.utils as utils
import plumage.plotting as pplt
import stannon.stannon as stannon
import stannon.plotting as splt
import matplotlib.pyplot as plt

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

#------------------------------------------------------------------------------
# Import science spectra, normalise, and prepare fluxes
#------------------------------------------------------------------------------
# Import literature info
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

# Import planet info
tic_info_all = utils.load_fits_table("TRANSIT_FITS", "tess", path="spectra")
tic_info = tic_info_all[~np.isnan(tic_info_all["rp_rstar_fit"])]

# Load results tables for both standards and TESS targets
obs_tess = utils.load_fits_table("OBS_TAB", "tess", path="spectra")

# Crossmatch
obs_join_tess = obs_tess.join(tess_info, "source_id", rsuffix="_info")

tess_mask = np.logical_and(
    obs_join_tess["Bp-Rp"].values > bp_rp_cutoff,
    ~np.isnan(obs_join_tess["G_mag"].values))

# Load in RV corrected TESS spectra
wave_tess_br = utils.load_fits_image_hdu("rest_frame_wave", "tess", arm="br")
spec_tess_br = utils.load_fits_image_hdu("rest_frame_spec", "tess", arm="br")
e_spec_tess_br = utils.load_fits_image_hdu("rest_frame_sigma", "tess", arm="br")

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

# Apply bad pixel mask
fluxes_norm[bad_px_mask] = 1
ivars_norm[bad_px_mask] = 0

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
# Predict labels
tess_labels_pred, tess_errs_all, tess_chi2_all = sm.infer_labels(
    fluxes_norm[:,sm.adopted_wl_mask],
    ivars_norm[:,sm.adopted_wl_mask])

# Correct labels for systematics
systematic_vector = np.tile(adopted_label_systematics, np.sum(tess_mask))
systematic_vector = systematic_vector.reshape([np.sum(tess_mask), n_labels])
tess_labels_pred -= systematic_vector

# Create uncertainties vector. TODO: do this in quadrature with those output
# from infer_labels
uncertainties_vector = np.tile(adopted_label_uncertainties, np.sum(tess_mask))
e_fit = uncertainties_vector.reshape([np.sum(tess_mask), n_labels])

"""
# Plot a joint CMD of benchmarks and science targets
splt.plot_cannon_cmd(
    benchmark_colour=std_info[std_mask]["Bp-Rp"],
    benchmark_mag=obs_join[std_mask]["K_mag_abs"],
    benchmark_feh=sm.label_values[:,2],
    science_colour=obs_join_tess["Bp-Rp"],
    science_mag=obs_join_tess["K_mag_abs"],)
"""

splt.plot_cannon_cmd(
    benchmark_colour=obs_join_tess["Bp-Rp"][tess_mask],
    benchmark_mag=obs_join_tess["K_mag_abs"][tess_mask],
    benchmark_feh=tess_labels_pred[:,2],
    science_colour=[2],
    science_mag=[7],)

# Plot Kiel Diagram for results
splt.plot_kiel_diagram(
    teffs=tess_labels_pred[:,0],
    e_teffs=e_fit[:,0],
    loggs=tess_labels_pred[:,1],
    e_loggs=e_fit[:,1],
    fehs=tess_labels_pred[:,2],
    label="science",)

#------------------------------------------------------------------------------
# Planet-star correlations
#------------------------------------------------------------------------------
# Planet-[Fe/H] correlations
pplt.plot_planet_feh_correlation(
    obs_tab=obs_join_tess[tess_mask],
    tic_info=tic_info,
    host_star_fehs=tess_labels_pred[:,2],
    giant_planet_rp=3,
    use_default_bins=False,
    custom_nbins=5,)
