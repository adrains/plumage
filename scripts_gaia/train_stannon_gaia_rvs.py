"""Script to train, cross validate, and save a Cannon model. The parameters for
this script can be found in cannon_settings_gaia_rvs.yml.

This is part of a series of Cannon scripts in scripts_gaia:
  1) collate_spectra_gaia_rvs.py
  2) prepare_training_sample_gaia_rvs.py
  3) train_stannon_gaia_rvs.py
  4) make_stannon_diagnostics_gaia_rvs.py
  4) run_stannon_gaia_rvs.py

Note that, at present, our Stan model is trained via optimising--a feature not
present in the more restricted pyStan version 3+. As such, this code only works
for pyStan version 2.
"""
import numpy as np
import pandas as pd
import plumage.utils as pu
import stannon.stannon as stannon
import stannon.plotting as splt
import stannon.utils as su
from datetime import datetime
from collections import Counter

#------------------------------------------------------------------------------
# Import Settings
#------------------------------------------------------------------------------
cannon_settings_yaml = "scripts_gaia/cannon_settings_gaia_rvs.yml"
cs = su.load_cannon_settings(cannon_settings_yaml)

#------------------------------------------------------------------------------
# Training sample
#------------------------------------------------------------------------------
# Import dataframe with benchmark parameters
obs_join = pu.load_fits_table(
    extension="OBS_TAB",
    label=cs.fits_label,
    fn_base=cs.fits_fn_base,
    path=cs.fits_folder,)

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
#   3) Polynomial order, O
#   4) Number of pixels, P
#   5) Number of benchmarks, S
# And given these we save all results when using this model to a unique fits
# HDU, which allows for the same fits file to contain the results of multiple
# Cannon models using different sets of benchmarks, labels, or pixels.
if cs.do_use_subset_of_benchmark_sample:
    M_G = obs_join["M_G"].values
    bp_rp = obs_join["bp_rp"].values
    
    # Grab mask of candidate benchmarks
    icb = obs_join["is_cannon_benchmark"].values

    within_bp_rp_bounds = np.logical_and(
        bp_rp > cs.BP_RP_lims[0], bp_rp < cs.BP_RP_lims[1])

    within_M_G_bounds = np.logical_and(
            M_G > cs.M_G_lims[0], M_G < cs.M_G_lims[1])

    adopted_benchmark = np.all((
        within_bp_rp_bounds, within_M_G_bounds, icb), axis=0)

else:
    adopted_benchmark = obs_join["is_cannon_benchmark"].values

if np.sum(adopted_benchmark) == 0:
    raise Exception("Zero benchmarks selected!")

if len(set(Counter(obs_join[adopted_benchmark].index.values).values())) > 1:
    raise Exception("Duplicate target IDs detected!")

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
# Load rest-frame spectra
wls, fluxes_norm, ivars_norm, bad_px_mask, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_gaia_rvs(
        fits_folder=cs.fits_folder,
        fits_fn_base=cs.fits_fn_base,
        fits_label=cs.fits_label,
        wl_min_model=cs.wl_min_model,
        wl_max_model=cs.wl_max_model,)

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
print("\torder: \t\t\t = {}".format(cs.cannon_order))
print("\tlambda: \t\t = {:0.0f}-{:0.0f} A".format(
    cs.wl_min_model, cs.wl_max_model))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(cs.label_names)))
print("\tlabels: \t\t = {}".format(cs.label_names))
print("\tn training sample: \t = {:0.0f}".format(np.sum(adopted_benchmark)))
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
    training_data=fluxes_norm[adopted_benchmark],
    training_data_ivar=ivars_norm[adopted_benchmark],
    training_labels=label_values_all[adopted_benchmark],
    training_ids=obs_join[adopted_benchmark].index.values,
    label_names=cs.label_names,
    wavelengths=wls,
    model_type=cs.model_type,
    order=cs.cannon_order,
    training_variances=label_var_all[adopted_benchmark],
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask[adopted_benchmark],)

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
        n_statistical_samples=cs.n_cross_val_samples,
        do_LOOCV=cs.do_LOOCV,
        k_folds=cs.k_folds,
        k_fold_CV_rng_seed=cs.k_fold_CV_rng_seed,)

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
fits_ext_label = "{}_{}L_{}O_{}P_{}S".format(
    cs.model_label, sm.L, sm.O, sm.P, sm.S)

pu.save_fits_table(
    extension="CANNON_MODEL",
    dataframe=result_df,
    label=cs.fits_label,
    fn_base=cs.fits_fn_base,
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
