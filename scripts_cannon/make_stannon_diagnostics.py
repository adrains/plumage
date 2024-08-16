"""Script to run after train_stannon.py to generate diagnostic and results 
plots. For the MARCS spectra comparison to work, synthetic spectra at 
literature parameters should be generated using get_lit_param_synth.py.

This script is part of a series of Cannon scripts. The main sequence is:
 1) prepare_stannon_training_sample.py     --> label preparation
 2) train_stannon.py                       --> training and cross validation
 3) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 4) run_stannon.py                         --> running on science spectra

The parameters for steps 2) and 3) can be found in cannon_settings.py, which
should be modified/duplicated and imported.
"""
import os
import numpy as np
import plumage.utils as pu
import stannon.stannon as stannon
import stannon.plotting as splt
import stannon.tables as st
import stannon.utils as su

#------------------------------------------------------------------------------
# Import Settings
#------------------------------------------------------------------------------
cannon_settings_yaml = "scripts_cannon/cannon_settings.yml"
cs = su.load_cannon_settings(cannon_settings_yaml)

#------------------------------------------------------------------------------
# Parameters and Setup
#------------------------------------------------------------------------------
# Import model
model_name = "stannon_model_{}_{}label_{}px_{}.pkl".format(
    cs.model_type, cs.n_labels, cs.npx, "_".join(cs.label_names))

cannon_model_path = os.path.join("spectra", model_name)

sm = stannon.load_model(cannon_model_path)

# Import saved reference data and mask
obs_join = pu.load_fits_table("CANNON_INFO", cs.std_label)

is_cannon_benchmark = obs_join["is_cannon_benchmark"].values

obs_join = obs_join[is_cannon_benchmark]

is_binary = obs_join["is_cpm"].values

# Grab our adopted labels--either from cross validation or just a quick check
if cs.is_cross_validated:
    labels_pred = sm.cross_val_labels
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

#------------------------------------------------------------------------------
# Label Recovery
#------------------------------------------------------------------------------
# Work out uncertainties
label_pred_std = np.nanstd(sm.training_labels - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*label_pred_std))

fn_label = "_{}_{}_label_{}".format(
    cs.model_type, len(cs.label_names), "_".join(cs.label_names))

# Label recovery for Teff, logg, and [Fe/H]
splt.plot_label_recovery(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**0.5,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    fn_suffix=fn_label,
    teff_lims=(2500,5000),
    teff_ticks=(500,250,200,100),
    logg_ticks=(0.25,0.125,0.1,0.05),
    feh_lims=(-0.95,0.65),
    feh_ticks=(0.5,0.25,0.4,0.2),)

# Plot recovery for interferometric Teff, M+15 [Fe/H], RA+12 [Fe/H], CPM [Fe/H]
splt.plot_label_recovery_per_source( 
    label_values=sm.training_labels, 
    e_label_values=sm.training_variances**0.5, 
    label_pred=labels_pred, 
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L), 
    obs_join=obs_join,
    fn_suffix=fn_label,
    teff_lims=(2800,5000),
    feh_lims=(-1.05,0.65),
    teff_ticks=(500,250,200,100),
    feh_ticks=(0.5,0.25,0.25,0.125),
    do_plot_mid_K_panel=True,)

# And finally plot the label recovery for any abundances we might be using
if len(cs.abundance_labels) >= 1:
    n_binary = np.sum(is_binary)
    splt.plot_label_recovery_abundances(
        label_values=sm.training_labels[is_binary],
        e_label_values=sm.training_variances[is_binary]**0.5,
        label_pred=labels_pred[is_binary],
        e_label_pred=np.tile(label_pred_std, n_binary).reshape(n_binary, sm.L),
        obs_join=obs_join[is_binary],
        fn_suffix=fn_label,
        abundance_labels=cs.abundance_labels,
        feh_lims=(-0.15,0.4),
        feh_ticks=(0.2,0.1,0.1,0.05))

#------------------------------------------------------------------------------
# Theta Coefficients
#------------------------------------------------------------------------------
# Import line lists
line_list_b = pu.load_linelist(
    filename=cs.line_list_file,
    wl_lower=cs.wl_min_model,
    wl_upper=cs.wl_grating_changeover,
    ew_min_ma=cs.ew_min_ma_b,)

line_list_r = pu.load_linelist(
    filename=cs.line_list_file,
    wl_lower=cs.wl_grating_changeover,
    wl_upper=cs.wl_max_model,
    ew_min_ma=cs.ew_min_ma_r,)

# Plot theta coefficients for each WiFeS arm
splt.plot_theta_coefficients(
    sm,
    teff_scale=1.0,
    x_lims=(cs.wl_min_model,cs.wl_grating_changeover),
    y_spec_lims=(0,2.25),
    y_theta_linear_lims=(-0.12,0.12),
    y_theta_quadratic_lims=(-0.2,0.2),
    y_theta_cross_lims=(-0.3,0.3),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    fn_label="b",
    linewidth=0.5,
    alpha=0.8,
    fn_suffix=fn_label,
    line_list=line_list_b,
    species_to_plot=cs.species_to_plot,
    only_plot_first_order_coeff=cs.only_plot_first_order_coeff,)

splt.plot_theta_coefficients(
    sm,
    teff_scale=1.0,
    x_lims=(cs.wl_grating_changeover,cs.wl_max_model),
    y_spec_lims=(0,2.25),
    y_theta_linear_lims=(-0.12,0.12),
    y_theta_quadratic_lims=(-0.1,0.1),
    y_theta_cross_lims=(-0.2,0.2),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    fn_label="r",
    linewidth=0.5,
    alpha=0.8,
    fn_suffix=fn_label,
    line_list=line_list_r,
    species_to_plot=cs.species_to_plot,
    only_plot_first_order_coeff=cs.only_plot_first_order_coeff,)

#------------------------------------------------------------------------------
# Spectral Recovery
#------------------------------------------------------------------------------
# Plot comparison of observed vs model spectra *at the literature  parameters*. 

# Plot model spectrum performance for WiFeS blue band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join,
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=cs.representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(cs.wl_min_model,cs.wl_grating_changeover),
    data_label="b",
    fn_label=fn_label,
    fig_size=cs.spec_comp_fig_size,)

# Plot model spectrum performance for WiFeS red band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join,
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=cs.representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(cs.wl_grating_changeover,cs.wl_max_model),
    data_label="r",
    fn_label=fn_label,
    fig_size=cs.spec_comp_fig_size,)

# Do the same, but across all wavelengths
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join,
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=cs.representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(cs.wl_min_model,cs.wl_max_model),
    data_label="br",
    fig_size=cs.spec_comp_fig_size,
    fn_label=fn_label,)

# Ditto, but now for all stars
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join,
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=obs_join.index,
    sort_col_name="BP_RP_dr3",
    x_lims=(cs.wl_min_model,cs.wl_max_model),
    data_label="a",
    fig_size=(12, 80),
    fn_label=fn_label,)

#------------------------------------------------------------------------------
# Label Prediction
#------------------------------------------------------------------------------
# Reset data mask just in case (TODO: this should be fixed going forward)
sm.data_mask = np.full(sm.S, True)
sm.initialise_masking()

# Predict label values + statistical uncertainties
pred_label_values, pred_label_sigmas_stat, chi2_all = sm.infer_labels(
    test_data=sm.masked_data,
    test_data_ivars=sm.masked_data_ivar)

# Correct labels for systematics
systematic_vector = np.tile(cs.adopted_label_systematics, np.sum(sm.data_mask))
systematic_vector = \
    systematic_vector.reshape([np.sum(sm.data_mask), cs.n_labels])
pred_label_values_corr = pred_label_values - systematic_vector

# Create uncertainties vector
cross_val_sigma = np.tile(cs.adopted_label_uncertainties, np.sum(sm.data_mask))
cross_val_sigma = cross_val_sigma.reshape([np.sum(sm.data_mask), cs.n_labels])
pred_label_sigmas_total = \
    np.sqrt(pred_label_sigmas_stat**2 + cross_val_sigma**2)

# Save uncertainties
for lbl_i, label in enumerate(cs.label_names):
    obs_join["{}_cannon_value".format(label)] = pred_label_values_corr[:,lbl_i]
    obs_join["{}_cannon_sigma_statistical".format(label)] = \
        pred_label_sigmas_stat[:,lbl_i]
    obs_join["{}_cannon_sigma_total".format(label)] = \
        pred_label_sigmas_total[:,lbl_i]

# Flag abberant logg values
delta_logg = \
    np.abs(obs_join["label_adopt_logg"].values - pred_label_values[:,1])
has_aberrant_logg = delta_logg > cs.aberrant_logg_threshold
obs_join["logg_aberrant"] = has_aberrant_logg

#------------------------------------------------------------------------------
# Tables
#------------------------------------------------------------------------------
# Table summarising benchmark sample
st.make_table_sample_summary(obs_join)

# Tabulate our adopted and benchmark parameters
st.make_table_benchmark_overview(
    obs_tab=obs_join,
    label_names=cs.label_names,
    abundance_labels=cs.abundance_labels,
    break_row=90,)

#------------------------------------------------------------------------------
# Benchmark CMD
#------------------------------------------------------------------------------
# TODO: masking for K dwarfs
splt.plot_cannon_cmd(
    benchmark_colour=obs_join["BP_RP_dr3"],
    benchmark_mag=obs_join["K_mag_abs"],
    benchmark_feh=sm.training_labels[:,2],
    highlight_mask=obs_join["is_cpm"].values,
    highlight_mask_label="Binary Benchmark",
    highlight_mask_2=obs_join["is_mid_k_dwarf"].values,
    highlight_mask_label_2="Early-Mid K Dwarf",)

#------------------------------------------------------------------------------
# GALAH-Gaia -- Valenti & Fischer 2005 diagnostic
#------------------------------------------------------------------------------
splt.plot_abundance_trend_recovery(
    obs_join=obs_join,
    vf05_full_file=cs.vf05_full_file,
    vf05_sampled_file=cs.vf05_sampled_file,)

#------------------------------------------------------------------------------
# Save updated fits
#------------------------------------------------------------------------------
# TODO: save the predicted parameters back to the fits file, but only for the
# benchmark stars.
#pu.save_fits_table("CANNON_INFO", obs_join, "cannon")
"""
labels = ["label_cannon_{}".format(lbl) for lbl in cs.label_names]

labels_all = np.full((154, cs.n_labels), np.nan)

labels_all[is_cannon_benchmark] = labels_pred

obs_join_all = pu.load_fits_table("CANNON_INFO", "cannon")

obs_join_all[labels] = labels_all
"""
