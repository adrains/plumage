"""Script to run after train_stannon.py to generate diagnostic and results 
plots. For the MARCS spectra comparison to work, synthetic spectra at 
literature parameters should be generated using get_lit_param_synth.py.

This script is part of a series of Cannon scripts. The main sequence is:
 1) prepare_stannon_training_sample.py     --> label preparation
 2) train_stannon.py                       --> training and cross validation
 3) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 4) run_stannon.py                         --> running on science spectra
"""
import os
import numpy as np
import plumage.utils as pu
import stannon.stannon as stannon
import stannon.plotting as splt
import stannon.tables as st

#------------------------------------------------------------------------------
# Parameters and Setup
#------------------------------------------------------------------------------
label = "cannon"

# Model settings
wl_min_model = 6400
wl_max_model = 6800
wl_grating_changeover = 5400
npx = 852
model_type = "label_uncertainties"
abundance_labels = ["Ti_H"]
label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = int(len(label_names))

is_cross_validated = True
do_gaussian_spec_normalisation = True
wl_min_normalisation = 4000
wl_broadening = 50
poly_order = 4

only_plot_first_order_coeff = True

# Line lists to overplot against theta coefficients. Due to the density of
# atomic features in the blue, we have a more strict threshold for labelling.
line_list_file = "data/t3500_g+5.0_z+0.00_a+0.00_v1.00_latoms.eqw"
ew_min_ma_b = 400
ew_min_ma_r = 150

# Import model
model_name = "stannon_model_{}_{}label_{}px_{}.pkl".format(
    model_type, n_labels, npx, "_".join(label_names))

cannon_model_path = os.path.join("spectra", model_name)

sm = stannon.load_model(cannon_model_path)

# Import saved reference data and mask
obs_join = pu.load_fits_table("CANNON_INFO", "cannon")

is_cannon_benchmark = obs_join["is_cannon_benchmark"].values

# Grab our adopted labels--either from cross validation or just a quick check
if is_cross_validated:
    labels_pred = sm.cross_val_labels
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Adopted label uncertainties and systematics based on cross validation
# performance on the benchmark set. Quoted systematics are fit - lit, meaning
# that a positive systematic means the Cannon has *overestimated* the value 
# (and thus the systematic should be substracted). The temperature systematic 
# and uncertainty will be adopted from just the interferometric set, whereas
# logg and [Fe/H] will be taken from the complete sample. 
# TODO: adopted_label_uncertainties = sm.adopted_label_uncertainties
adopted_label_systematics = np.array([15.63, 0.0, 0.03, -0.01,])
adopted_label_uncertainties = np.array([60.18, 0.04, 0.11, 0.12])

# Which species to overplot on our theta plot. It gets very busy the more
# species we plot, so it's currently limited to the most prominent species.
species_to_plot = ["Ca 1", "Ti 1", "Fe 1",] 

#------------------------------------------------------------------------------
# Label Recovery
#------------------------------------------------------------------------------
# Work out uncertainties
label_pred_std = np.nanstd(sm.training_labels - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*label_pred_std))

fn_label = "_{}_{}_label_{}".format(
    model_type, len(label_names), "_".join(label_names))

# Label recovery for Teff, logg, and [Fe/H]
splt.plot_label_recovery(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**0.5,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    obs_join=obs_join[is_cannon_benchmark],
    fn_suffix=fn_label,
    teff_lims=(2750,4300),
    teff_ticks=(500,250,200,100),
    logg_ticks=(0.25,0.125,0.2,0.1),
    feh_lims=(-1.1,0.65),
    feh_ticks=(0.5,0.25,0.4,0.2),)

# Plot recovery for interferometric Teff, M+15 [Fe/H], RA+12 [Fe/H], CPM [Fe/H]
splt.plot_label_recovery_per_source( 
    label_values=sm.training_labels, 
    e_label_values=sm.training_variances**0.5, 
    label_pred=labels_pred, 
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L), 
    obs_join=obs_join[is_cannon_benchmark],
    fn_suffix=fn_label,)

# And finally plot the label recovery for any abundances we might be using
if len(abundance_labels) >= 1:
    splt.plot_label_recovery_abundances(
        label_values=sm.training_labels,
        e_label_values=sm.training_variances**0.5,
        label_pred=labels_pred,
        e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
        obs_join=obs_join[is_cannon_benchmark],
        fn_suffix=fn_label,
        abundance_labels=abundance_labels,
        feh_lims=(-0.65,0.65),
        feh_ticks=(0.4,0.2,0.2,0.1))

#------------------------------------------------------------------------------
# Theta Coefficients
#------------------------------------------------------------------------------
# Import line lists
line_list_b = pu.load_linelist(
    filename=line_list_file,
    wl_lower=wl_min_model,
    wl_upper=wl_grating_changeover,
    ew_min_ma=ew_min_ma_b,)

line_list_r = pu.load_linelist(
    filename=line_list_file,
    wl_lower=wl_grating_changeover,
    wl_upper=wl_max_model,
    ew_min_ma=ew_min_ma_r,)

# Plot theta coefficients for each WiFeS arm
splt.plot_theta_coefficients(
    sm,
    teff_scale=1.0,
    x_lims=(wl_min_model,wl_grating_changeover),
    y_spec_lims=(0,2.25),
    y_theta_lims=(-0.25,0.25),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    fn_label="b",
    linewidth=0.9,
    alpha=0.8,
    fn_suffix=fn_label,
    line_list=line_list_b,
    species_to_plot=species_to_plot,
    only_plot_first_order_coeff=only_plot_first_order_coeff,)

splt.plot_theta_coefficients(
    sm,
    teff_scale=1.0,
    x_lims=(wl_grating_changeover,wl_max_model),
    y_spec_lims=(0,2.25),
    y_theta_lims=(-0.12,0.12),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    fn_label="r",
    linewidth=0.9,
    alpha=0.8,
    fn_suffix=fn_label,
    line_list=line_list_r,
    species_to_plot=species_to_plot,
    only_plot_first_order_coeff=only_plot_first_order_coeff,)

#------------------------------------------------------------------------------
# Spectral Recovery
#------------------------------------------------------------------------------
# Plot comparison of observed vs model spectra *at the literature  parameters*. 
# Here we've picked a set of spectral types ranging over our BP-RP range.
representative_stars_source_ids = [
    "5853498713190525696",      # M5.5, GJ 551
    "2640434056928150400",      # M5, GJ 1286
    "2595284016771502080",      # M5, LHS 3799
    "2868199402451064064",      # M4.7, GJ 1288
    "6322070093095493504",      # M2, GJ 581
    "2603090003484152064",      # M3, GJ 876
    "4472832130942575872",      # M4, Gl 699
    "2910909931633597312",      # M3, LP 837-53
    "3184351876391975808",      # M2, Gl 173
    "2739689239311660672",      # M0, Gl 908
    "145421309108301184",       # K8, Gl 169
    "4282578724832056576",      # M0.7, Gl 740
]

# Plot model spectrum performance for WiFeS blue band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[is_cannon_benchmark],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(wl_min_model,wl_grating_changeover),
    data_label="b",
    fn_label=fn_label,)

# Plot model spectrum performance for WiFeS red band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[is_cannon_benchmark],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(wl_grating_changeover,wl_max_model),
    data_label="r",
    fn_label=fn_label,)

# Do the same, but across all wavelengths and for all stars
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[is_cannon_benchmark],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=obs_join.index,
    sort_col_name="BP_RP_dr3",
    x_lims=(wl_min_model,wl_max_model),
    data_label="d",
    fn_label=fn_label,)

# -----------------------------------------------------------------------------
# MARCS Spectra Comparison
# -----------------------------------------------------------------------------
# Note: we run get_lit_param_synth.py first to already have MARCS spectra.
# Since the MARCS grid doesn't have an abundance dimension, we can only make
# this plot for a 3 label model.
if n_labels == 3:
    # Load in RV corrected standard spectra
    wls = pu.load_fits_image_hdu("rest_frame_wave", label, arm="br")
    spec_marcs_br = \
        pu.load_fits_image_hdu("rest_frame_synth_lit", label, arm="br")
    e_spec_marcs_br = np.ones_like(spec_marcs_br)

    # TODO, HACK, replace all nan fluxes
    for spec_i in range(len(spec_marcs_br)):
        if np.sum(np.isnan(spec_marcs_br[spec_i])) > 1000:
            spec_marcs_br[spec_i] = np.ones_like(spec_marcs_br[spec_i])

    # Grab MARCS spectra for just our benchmarks
    fluxes_marcs_norm, _, bad_px_mask, _, _ = \
        stannon.prepare_cannon_spectra_normalisation(
            wls=wls,
            spectra=spec_marcs_br[is_cannon_benchmark],
            e_spectra=e_spec_marcs_br[is_cannon_benchmark],
            wl_min_model=wl_min_model,
            wl_max_model=wl_max_model,
            wl_min_normalisation=wl_min_normalisation,
            wl_broadening=wl_broadening,
            do_gaussian_spectra_normalisation=do_gaussian_spec_normalisation,
            poly_order=poly_order)

    # Plot Cannon vs MARCS spectra comparison over the entire spectral range
    splt.plot_spectra_comparison(
        sm=sm,
        obs_join=obs_join[is_cannon_benchmark],
        fluxes=fluxes_marcs_norm,
        bad_px_masks=bad_px_mask,
        labels_all=sm.training_labels,
        source_ids=representative_stars_source_ids,
        sort_col_name="BP_RP_dr3",
        x_lims=(wl_min_model,wl_max_model),
        fn_label=fn_label,
        data_label="marcs",
        data_plot_label="MARCS",
        data_plot_colour="b",)

#------------------------------------------------------------------------------
# Label Prediction
#------------------------------------------------------------------------------
# Reset data mask just in case (TODO: this should be fixed going forward)
sm.data_mask = np.full(sm.S, True)
sm.initialise_masking()

# Predict labels
labels_pred, e_labels_pred, chi2_all = sm.infer_labels(
    test_data=sm.masked_data,
    test_data_ivars=sm.masked_data_ivar)

# Correct labels for systematics
systematic_vector = np.tile(adopted_label_systematics, np.sum(sm.data_mask))
systematic_vector = systematic_vector.reshape([np.sum(sm.data_mask), n_labels])
labels_pred -= systematic_vector

# Create uncertainties vector. TODO: do this in quadrature with those output
# from infer_labels
cross_val_sigma = np.tile(adopted_label_uncertainties, np.sum(sm.data_mask))
cross_val_sigma = cross_val_sigma.reshape([np.sum(sm.data_mask), n_labels])
e_labels_pred = np.sqrt(e_labels_pred**2 + cross_val_sigma**2)

# Round our uncertainties, and check to see if the uncertainties are unique

#------------------------------------------------------------------------------
# Tables
#------------------------------------------------------------------------------
# Fit using fully trained Cannon
label_source_cols = ["label_source_{}".format(label) for label in label_names]

# Table summarising benchmark sample
st.make_table_sample_summary(obs_join)

# Tabulate our adopted and benchmark parameters
st.make_table_benchmark_overview(
    obs_tab=obs_join[is_cannon_benchmark],
    labels_adopt=sm.training_labels,
    sigmas_adopt=sm.training_variances**2,
    labels_fit=labels_pred,
    label_sources=obs_join[label_source_cols].values[is_cannon_benchmark],
    abundance_labels=abundance_labels,
    synth_logg_col="logg_synth",
    aberrant_logg_threshold=0.15,)

#------------------------------------------------------------------------------
# Benchmark CMD
#------------------------------------------------------------------------------
splt.plot_cannon_cmd(
    benchmark_colour=obs_join[is_cannon_benchmark]["BP_RP_dr3"],
    benchmark_mag=obs_join[is_cannon_benchmark]["K_mag_abs"],
    benchmark_feh=sm.training_labels[:,2],
    highlight_mask=obs_join[is_cannon_benchmark]["is_cpm"].values,
    highlight_mask_label="Binary Benchmark",)
