"""Script to train, cross validate, and save a Cannon model. After a model has
been created, the scripts make_stannon_diagnostics.py and run_stannon.py should
be used.

Note that, at present, our Stan model is trained via optimising--a feature not
present in the more restricted pyStan version 3+. As such, this code only works
for pyStan version 2.

This script is part of a series of Cannon scripts. The main sequence is:
 1) assess_literature_systematics.py       --> benchmark chemistry compilation
 2) prepare_stannon_training_sample.py     --> label preparation
 3) train_stannon.py                       --> training and cross validation
 4) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 5) run_stannon.py                         --> running on science spectra

The parameters for step 2) can be found in cannon_settings.py, which should be
modified/duplicated and imported.
"""
import numpy as np
import pandas as pd
import plumage.utils as pu
import stannon.stannon as stannon
import stannon.plotting as splt
import stannon.utils as su
from datetime import datetime

#------------------------------------------------------------------------------
# Import Settings
#------------------------------------------------------------------------------
cannon_settings_yaml = "scripts_cannon/cannon_settings.yml"
cs = su.load_cannon_settings(cannon_settings_yaml)

#------------------------------------------------------------------------------
# Training sample
#------------------------------------------------------------------------------
# Import dataframe with benchmark parameters
obs_join = pu.load_fits_table("CANNON_INFO", cs.std_label)

# Now that we've collated all our labels and vetted all our targets, we still
# need to select which stars the Cannon will actually be trained on. This uses
# result in four different boolean arrays:
#   1) 'passed_quality_cuts'        ---> benchmarks must have this as True
#   2) 'has_complete_label_set'     ---> benchmarks must have this as True
#   3) 'is_cannon_benchmark'        ---> *potential* benchmarks
#   4) 'adopted_benchmark'          ---> *adopted* benchmarks

# A Cannon model is assumed to be uniquely defined by:
#   1) the model 'label', e.g. 'MK' for M and K dwarfs.
#   2) Number of labels, L
#   3) Number of pixels, P
#   4) Number of benchmarks, S
# And given these we save all results when using this model to a unique fits
# HDU, which allows for the same fits file to contain the results of multiple
# Cannon models using different sets of benchmarks, labels, or pixels.
if cs.do_use_subset_of_benchmark_sample:
    M_Ks = obs_join["K_mag_abs"].values
    bp_rp = obs_join["BP_RP_dr3"].values
    
    # Grab mask of candidate benchmarks
    icb = obs_join["is_cannon_benchmark"].values

    # Accept those benchmarks fainter (> M_Ks) than the warm M_Ks bound
    warm_MKs_bounds = (
        cs.BP_RP_vs_M_Ks_cut_gradient * bp_rp + cs.M_Ks_intercept_warm)
    within_warm_MKs_bounds = M_Ks >= warm_MKs_bounds

    # Accept those benchmarks brighter (< M_Ks) than the cool M_Ks bound
    cool_MKs_bounds = (
        cs.BP_RP_vs_M_Ks_cut_gradient * bp_rp + cs.M_Ks_intercept_cool)
    within_cool_MKs_bound = M_Ks <= cool_MKs_bounds

    # Adopt benchmarks candidate benchmarks within the warm and cool bounds
    within_bounds = np.logical_and(
        within_warm_MKs_bounds, within_cool_MKs_bound)
    
    adopted_benchmark = np.logical_and(icb, within_bounds)

else:
    adopted_benchmark = obs_join["is_cannon_benchmark"].values

if np.sum(adopted_benchmark) == 0:
    raise Exception("Zero benchmarks selected!")

# Create new DataFrame for results
result_df = pd.DataFrame(
    data={"source_id_dr3":obs_join.index,
          "adopted_benchmark":adopted_benchmark},)
result_df.set_index("source_id_dr3", inplace=True)

#------------------------------------------------------------------------------
# Training labels
#------------------------------------------------------------------------------
# Grab benchmark labels
label_value_cols = []
label_var_cols = []

for lbl_i, lbl in enumerate(cs.label_names):
    label_value_cols.append("label_adopt_{}".format(lbl))
    label_var_cols.append("label_adopt_var_{}".format(lbl))

label_values_all = obs_join[label_value_cols].values
label_var_all = obs_join[label_var_cols].values

# Scale our uncertainties by the constant per-label factors we've defined in
# our settings file.
n_std =len(obs_join)
label_var_all *= \
    np.tile(cs.lit_std_scale_fac, n_std).reshape((cs.n_labels, n_std)).T**2

# Optional for testing: run with uniform variances
if cs.use_label_uniform_variances:
    # Unlog logg, [Fe/H], and any abundances
    unlogged_label_values_all = label_values_all.copy()
    unlogged_label_values_all[:,1:] = 10**label_values_all[:,1:]

    # Calculate test variances for all labels based on uniform_var_frac_error
    label_std_all = unlogged_label_values_all * cs.uniform_var_frac_error

    # Relog logg, [Fe/H], and any abundances
    label_std_all[:,1:] = \
        (label_std_all[:,1:] / (np.log(10) * unlogged_label_values_all[:,1:]))
    
    label_var_all = label_std_all**2

#------------------------------------------------------------------------------
# Flux preparation
#------------------------------------------------------------------------------
# Load in RV corrected standard spectra
wls = pu.load_fits_image_hdu("rest_frame_wave", cs.std_label, arm="br")
spec_std_br = pu.load_fits_image_hdu("rest_frame_spec", cs.std_label, arm="br")
e_spec_std_br = pu.load_fits_image_hdu(
    "rest_frame_sigma", cs.std_label, arm="br")

# [Optional] Broaden fluxes to a lower resolution
if cs.do_constant_in_wl_spectral_broadening:
    wls_new, spec_std_br_broad, e_spec_std_br_broad = su.broaden_cannon_fluxes(
        wls=wls,
        spec_std_br=spec_std_br,
        e_spec_std_br=e_spec_std_br,
        target_delta_lambda=cs.target_delta_lambda,)
        
    # Swap references
    wls_unbroadened = wls
    spec_std_br_unbroadened = spec_std_br
    e_spec_std_br_unbroadened = e_spec_std_br

    wls = wls_new
    spec_std_br = spec_std_br_broad
    e_spec_std_br = e_spec_std_br_broad

# Normalise fluxes
fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wls,
        spectra=spec_std_br[adopted_benchmark],
        e_spectra=e_spec_std_br[adopted_benchmark],
        wl_min_model=cs.wl_min_model,
        wl_max_model=cs.wl_max_model,
        wl_min_normalisation=cs.wl_min_normalisation,
        wl_broadening=cs.wl_broadening,
        do_gaussian_spectra_normalisation=cs.do_gaussian_spectra_normalisation,
        poly_order=cs.poly_order)

# Similar to how we can optionally scale the uncertainties on our labels, we
# can do the same for the uncertainties on our spectra.
ivars_norm *= 1/(cs.spectra_std_scale_fac**2)

#------------------------------------------------------------------------------
# Make and train model
#------------------------------------------------------------------------------
# Diagnostic summary
print("\n\n", "%"*80, "\n", sep="")
print("\tModel Params:\n\t", "-"*21, sep="")
print("\tmodel: \t\t\t = {}".format(cs.model_type))
print("\tlambda: \t\t = {:0.0f}-{:0.0f} A".format(
    cs.wl_min_model, cs.wl_max_model))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(cs.label_names)))
print("\tlabels: \t\t = {}".format(cs.label_names))
print("\tn benchmarks: \t\t = {:0.0f}".format(np.sum(adopted_benchmark)))
print("\tGaussian Normalisation:\t = {}".format(
    cs.do_gaussian_spectra_normalisation))
if cs.do_gaussian_spectra_normalisation:
    print("\twl broadening: \t\t = {:0.0f} Å".format(cs.wl_broadening))
else:
    print("\tpoly order: \t\t = {:0.0f}".format(cs.poly_order))

print("\n\tTraining Params:\n\t", "-"*21, sep="")
print("\tcross validation: \t = {}".format(cs.do_cross_validation))
print("\titerative masking: \t = {}".format(cs.do_iterative_bad_px_masking))
print("\tinit with basic model: \t = {}".format(cs.init_with_basic_model))

print("\n\tTesting Params:\n\t", "-"*21, sep="")
print("\tuniform variances:\t = {}".format(cs.use_label_uniform_variances))
print("\tuniform var scale fac:\t = {}".format(cs.uniform_var_frac_error))
print("\tlit std scale fac:\t = {}".format(cs.lit_std_scale_fac))
print("\tspec std scale fac:\t = {}".format(cs.spectra_std_scale_fac))

print("\n", "%"*80, "\n\n", sep="")

# Make model
sm = stannon.Stannon(
    training_data=fluxes_norm,
    training_data_ivar=ivars_norm,
    training_labels=label_values_all[adopted_benchmark],
    training_ids=obs_join[adopted_benchmark].index.values,
    label_names=cs.label_names,
    wavelengths=wls,
    model_type=cs.model_type,
    training_variances=label_var_all[adopted_benchmark],
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask,)

# Timing
start_time = datetime.now()

# Train model
print("\n\n", "-"*100, sep="",)
print("\nFitting initial Cannon model with {} benchmarks\n".format(
    np.sum(adopted_benchmark)))
print("-"*100, "\n\n", sep="",)
print("Fitting started: {}\n".format(start_time.ctime()))
sm.train_cannon_model(
    suppress_stan_output=cs.suppress_stan_output,
    init_uncertainty_model_with_basic_model=cs.init_with_basic_model,
    max_iter=cs.max_iter,
    log_refresh_step=cs.log_refresh_step,)

finish_time = datetime.now()
time_elapsed = finish_time - start_time
print("\nFitting finished: {}\n".format(finish_time.ctime()))
print("Duration (hh:mm:ss.ms) {}\n".format(time_elapsed))

# If we run the iterative bad px masking, train again afterwards
if cs.do_iterative_bad_px_masking:
    print("\n\n", "-"*100, sep="",)
    print("\nFitting sigma clipped Cannon model based on initial model\n")
    print("-"*100, "\n\n", sep="",)
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=cs.flux_sigma_to_clip)
    sm.train_cannon_model(
        suppress_stan_output=cs.suppress_stan_output,
        init_uncertainty_model_with_basic_model=cs.init_with_basic_model,
        max_iter=cs.max_iter,
        log_refresh_step=cs.log_refresh_step,)

# Run cross validation
if cs.do_cross_validation:
    sm.run_cross_validation(
        suppress_stan_output=cs.suppress_stan_output,
        init_uncertainty_model_with_basic_model=cs.init_with_basic_model,
        max_iter=cs.max_iter,
        log_refresh_step=cs.log_refresh_step,
        n_statistical_samples=cs.n_cross_val_samples,)

    labels_pred = sm.cross_val_labels
    labels_pred_sigma = sm.cross_val_sigmas
    labels_pred_chi2 = sm.cross_val_chi2

    # Save cross validation label predictions
    for label_i, label in enumerate(sm.label_names):
        # Mean
        col = "label_cv_{}".format(label)
        cv_label_values = np.full(len(obs_join), np.nan)
        cv_label_values[adopted_benchmark] = sm.cross_val_labels[:,label_i]
        result_df[col] = cv_label_values

        # Sigma
        col = "sigma_label_cv_{}".format(label)
        cv_label_sigmas = np.full(len(obs_join), np.nan)
        cv_label_sigmas[adopted_benchmark] = sm.cross_val_sigmas[:,label_i]
        result_df[col] = cv_label_sigmas

    # chi2
    cv_chi2 = np.full(len(obs_join), np.nan)
    cv_chi2[adopted_benchmark] = sm.cross_val_chi2
    result_df["chi2_cv"] = cv_chi2

# ...or just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

print("Model training complete!")

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
# Save model
sm.save_model(cs.model_save_path, name=cs.model_label,)

# Store results DF
fits_ext_label = "{}_{}L_{}P_{}S".format(
    cs.model_label, sm.L, sm.P, sm.S)

pu.save_fits_table(
    extension="CANNON_MODEL",
    dataframe=result_df,
    label=cs.std_label,
    path=cs.model_save_path,
    ext_label=fits_ext_label)

# TODO: save training settings
pass

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------
# Print Summary from leave-one-out cross-validation
resid_CV = np.nanstd(sm.training_labels - labels_pred, axis=0)

summary = ""

for label_i, label in enumerate(sm.label_names):
    if label == "teff":
        summary += "\nΔTeff = {:0.0f} K"
    elif label == "logg":
        summary += "Δlogg = {:0.2f} dex"
    else:
        summary += "Δ[{}]".format(label.replace("_", "/")) + " = {:0.2f} dex"

    if label_i < sm.L-1:
        summary += ", "

print("Leave-one-out Recovery:", summary.format(*tuple(resid_CV)), sep="\n")

# Compute the difference between true and input labels
fn_label = "_{}_{}_label_{}".format(
    cs.model_type, len(cs.label_names), "_".join(cs.label_names))

if cs.model_type == "label_uncertainties":
    splt.plot_label_uncertainty_adopted_vs_true_labels(sm, fn_label=fn_label)
