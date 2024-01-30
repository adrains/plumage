"""Script to import and run a trained Cannon model on science data.
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

#------------------------------------------------------------------------------
# Import science spectra, normalise, and prepare fluxes
#------------------------------------------------------------------------------
# Import literature info for benchmarks
obs_join_std = pu.load_fits_table("CANNON_INFO", "cannon")

is_cannon_benchmark = obs_join_std["is_cannon_benchmark"].values

obs_join_std = obs_join_std[is_cannon_benchmark]

#------------------------------------------------------------------------------
# Import science targets, normalise, and prepare fluxes
#------------------------------------------------------------------------------
sci_info = pu.load_info_cat(
    cs.science_info,
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,)

obs_sci = pu.load_fits_table("OBS_TAB", "tess", path="spectra")
obs_join_sci = obs_sci.join(sci_info, "source_id", rsuffix="_info")

# Get masks for which targets are appropriate for science and training sample
sci_mask = np.logical_and(
    obs_join_sci["BP-RP"].values > cs.sci_bp_rp_cutoff,
    ~np.isnan(obs_join_sci["G_mag"].values))

n_sci = np.sum(sci_mask)

std_mask = np.array([sid in sm.training_ids for sid in obs_join_std.index.values])

# Load in RV corrected science spectra
wave_sci_br = pu.load_fits_image_hdu(
    "rest_frame_wave", cs.science_dataset, arm="br")
spec_sci_br = pu.load_fits_image_hdu(
    "rest_frame_spec", cs.science_dataset, arm="br")
e_spec_sci_br = pu.load_fits_image_hdu(
    "rest_frame_sigma", cs.science_dataset, arm="br")

print("Running on {} sample".format(cs.science_dataset))

# Setup dataset
fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wave_sci_br,
        spectra=spec_sci_br[sci_mask],
        e_spectra=e_spec_sci_br[sci_mask],
        wl_max_model=cs.wl_max_model,
        wl_min_normalisation=cs.wl_min_normalisation,
        wl_broadening=cs.wl_broadening,
        do_gaussian_spectra_normalisation=cs.do_gaussian_spectra_normalisation,
        poly_order=cs.poly_order)

# Apply bad pixel mask
fluxes_norm[bad_px_mask] = 1
ivars_norm[bad_px_mask] = 0

#------------------------------------------------------------------------------
# Predict labels for science targets
#------------------------------------------------------------------------------
# Predict labels
pred_label_values, pred_label_sigmas_stat, chi2_all = sm.infer_labels(
    fluxes_norm[:,adopted_wl_mask],
    ivars_norm[:,adopted_wl_mask])

# Correct labels for systematics
systematic_vector = np.tile(cs.adopted_label_systematics[:cs.n_labels], n_sci)
systematic_vector = \
    systematic_vector.reshape([n_sci, cs.n_labels])
pred_label_values_corr = pred_label_values - systematic_vector

# Create uncertainties vector
cross_val_sigma = np.tile(cs.adopted_label_uncertainties[:cs.n_labels], n_sci)
cross_val_sigma = cross_val_sigma.reshape([n_sci, cs.n_labels])
pred_label_sigmas_total = \
    np.sqrt(pred_label_sigmas_stat**2 + cross_val_sigma**2)

# Save uncertainties
for lbl_i, label in enumerate(cs.label_names):
    # Initialise columns
    obs_join_sci["{}_cannon_value".format(label)] = np.nan
    obs_join_sci["{}_cannon_sigma_statistical".format(label)] = np.nan
    obs_join_sci["{}_cannon_sigma_total".format(label)] = np.nan

    # Save values for unmasked science targets
    obs_join_sci.loc[sci_mask, "{}_cannon_value".format(label)] = \
        pred_label_values_corr[:,lbl_i]
    obs_join_sci.loc[sci_mask, "{}_cannon_sigma_statistical".format(label)] = \
        pred_label_sigmas_stat[:,lbl_i]
    obs_join_sci.loc[sci_mask, "{}_cannon_sigma_total".format(label)] = \
        pred_label_sigmas_total[:,lbl_i]

# Flag abberant logg values
obs_join_sci["logg_aberrant"] = np.nan

delta_logg = np.abs(obs_join_sci.loc[sci_mask, "logg_synth"].values
                    - pred_label_values[:,1])
has_aberrant_logg = delta_logg > cs.aberrant_logg_threshold
obs_join_sci.loc[sci_mask,"logg_aberrant"] = has_aberrant_logg

#------------------------------------------------------------------------------
# Plot a CMD and create a table
#------------------------------------------------------------------------------
# Plot a joint CMD of benchmarks and science targets
splt.plot_cannon_cmd(
    benchmark_colour=obs_join_std[std_mask]["BP_RP_dr3"],
    benchmark_mag=obs_join_std[std_mask]["K_mag_abs"],
    benchmark_feh=sm.training_labels[:,2],
    science_colour=obs_join_sci[sci_mask]["BP-RP"],
    science_mag=obs_join_sci[sci_mask]["K_mag_abs"],)

caption = ("{} parameter fits (corrected for systematics). A $\dagger$ "
           "indicates a difference in $\log g > {:0.2f}$ between the "
           "Cannon and the fits from \citet{{rains_characterization_2021}}.")
caption = caption.format(cs.caption_unique, cs.aberrant_logg_threshold)

# Make results table
st.make_table_parameter_fit_results(
    obs_tab=obs_join_sci[sci_mask],
    label_fits=pred_label_values,
    e_label_fits=pred_label_sigmas_total,
    abundance_labels=cs.abundance_labels,
    break_row=61,
    star_label=(cs.star_label_tab, cs.sci_star_name_col),
    table_label=cs.science_dataset,
    caption=caption,
    bp_mag_col="BP_mag",
    bp_rp_col="BP-RP",)

# ---
# HACK: temp plot for stars of interest
# ---

splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join_sci[sci_mask],
    fluxes=fluxes_norm,
    bad_px_masks=bad_px_mask,
    labels_all=pred_label_values_corr,
    source_ids=["6385548541499112448"],
    sort_col_name="BP_RP",
    x_lims=(cs.wl_min_model, cs.wl_max_model),
    data_label="d",
    fn_label="en",
    fig_size=[12,2],
    star_name_col=cs.sci_star_name_col,
    bp_rp_col="BP-RP",)