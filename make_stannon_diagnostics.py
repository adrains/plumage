"""Script to run after train_stannon.py to generate diagnostic and results 
plots. For the MARCS spectra comparison to work, synthetic spectra at 
literature parameters should be generated using get_lit_param_synth.py.
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
wl_min_model = 4000
wl_max_model = 7000
npx = 5024
model_type = "label_uncertainties"
abundance_labels = []
label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = int(len(label_names))

is_cross_validated = True
do_gaussian_spectra_normalisation = True
wl_min_normalisation = 4000
wl_broadening = 50
poly_order = 4

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
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    obs_join=[is_cannon_benchmark],
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
    obs_join=obs_join[is_cannon_benchmark],
    fn_suffix=fn_label,)

# And finally plot the label recovery for any abundances we might be using
splt.plot_label_recovery_abundances(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
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
# Plot theta coefficients for each WiFeS arm
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

#------------------------------------------------------------------------------
# Spectral Recovery
#------------------------------------------------------------------------------
# Plot comparison of observed vs model spectra *at the literature  parameters*. 
# Here we've picked a set of spectral types ranging over our BP-RP range.
representative_stars_source_ids = [
    #"5853498713190525696",      # M5.5, GJ 551
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
    x_lims=(wl_min_model,5400),
    fn_label="b",)

# Plot model spectrum performance for WiFeS red band
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[is_cannon_benchmark],
    fluxes=sm.training_data,
    bad_px_masks=sm.bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(5400,wl_max_model),
    fn_label="r",)

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
    fn_label="d",
    data_label="benchmark",)

# -----------------------------------------------------------------------------
# MARCS Spectra Comparison
# -----------------------------------------------------------------------------
# Note: we run get_lit_param_synth.py first to already have MARCS spectra

# Load in RV corrected standard spectra
wls = pu.load_fits_image_hdu("rest_frame_wave", label, arm="br")
spec_marcs_br = pu.load_fits_image_hdu("rest_frame_synth_lit", label, arm="br")
e_spec_marcs_br = np.ones_like(spec_marcs_br)

# TODO, HACK, replace all nan fluxes
for spec_i in range(len(spec_marcs_br)):
    if np.sum(np.isnan(spec_marcs_br[spec_i])) > 1000:
        spec_marcs_br[spec_i] = np.ones_like(spec_marcs_br[spec_i])

# Grab MARCS spectra for just our benchmarks
fluxes_marcs_norm, ivars_marcs_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wls,
        spectra=spec_marcs_br[is_cannon_benchmark],
        e_spectra=e_spec_marcs_br[is_cannon_benchmark],
        wl_min_model=wl_min_model,
        wl_max_model=wl_max_model,
        wl_min_normalisation=wl_min_normalisation,
        wl_broadening=wl_broadening,
        do_gaussian_spectra_normalisation=do_gaussian_spectra_normalisation,
        poly_order=poly_order)

# Plot Cannon vs MARCS spectra comparison over the entire spectral range
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[is_cannon_benchmark],
    fluxes=fluxes_marcs_norm,
    bad_px_masks=bad_px_mask,
    labels_all=sm.training_labels,
    source_ids=representative_stars_source_ids,
    x_lims=(4000,wl_max_model),
    fn_label="marcs",
    data_plot_label="MARCS",
    data_plot_colour="b",)

#------------------------------------------------------------------------------
# Tables
#------------------------------------------------------------------------------
label_source_cols = ["label_source_{}".format(label) for label in label_names]

# Tabulate our adopted benchmark parameters
st.make_table_benchmark_overview(
    obs_tab=obs_join[is_cannon_benchmark],
    labels=sm.training_labels,
    e_labels=sm.training_variances**2,
    label_sources=obs_join[label_source_cols].values[is_cannon_benchmark],
    abundance_labels=abundance_labels,)