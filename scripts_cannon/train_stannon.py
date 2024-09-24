"""Script to train, cross validate, and save a Cannon model. After a model has
been created, the scripts make_stannon_diagnostics.py and run_stannon.py should
be used.

Note that, at present, our Stan model is trained via optimising--a feature not
present in the more restricted pyStan version 3+. As such, this code only works
for pyStan version 2.

This script is part of a series of Cannon scripts. The main sequence is:
 1) prepare_stannon_training_sample.py     --> label preparation
 2) train_stannon.py                       --> training and cross validation
 3) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 4) run_stannon.py                         --> running on science spectra

The parameters for steps 2) and 3) can be found in cannon_settings.py, which
should be modified/duplicated and imported.
"""
import numpy as np
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
# Training labels
#------------------------------------------------------------------------------
# Import dataframe with benchmark parameters
obs_join = pu.load_fits_table("CANNON_INFO", cs.std_label)

is_cannon_benchmark = obs_join["is_cannon_benchmark"].values

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
        spectra=spec_std_br[is_cannon_benchmark],
        e_spectra=e_spec_std_br[is_cannon_benchmark],
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
print("\tn benchmarks: \t\t = {:0.0f}".format(np.sum(is_cannon_benchmark)))
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
    training_labels=label_values_all[is_cannon_benchmark],
    training_ids=obs_join[is_cannon_benchmark].index.values,
    label_names=cs.label_names,
    wavelengths=wls,
    model_type=cs.model_type,
    training_variances=label_var_all[is_cannon_benchmark],
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask,)

# Timing
start_time = datetime.now()

# Train model
print("\n\n", "-"*100, sep="",)
print("\nFitting initial Cannon model with {} benchmarks\n".format(
    np.sum(is_cannon_benchmark)))
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
        log_refresh_step=cs.log_refresh_step,)

    labels_pred = sm.cross_val_labels

# ...or just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(cs.model_save_path)

print("Model training complete!")

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------
fn_label = "_{}_{}_label_{}".format(
    cs.model_type, len(cs.label_names), "_".join(cs.label_names))

uncertainties = np.nanstd(sm.training_labels - labels_pred, axis=0)

summary = \
    "\nΔ Teff = {:0.0f} K, Δlogg = {:0.2f} dex, Δ[Fe/H] = {:0.2f} dex"
print("Training set uncertainties:\n", summary.format(*tuple(uncertainties)))

# Compute the difference between true and input labels
if cs.model_type == "label_uncertainties":
    splt.plot_label_uncertainty_adopted_vs_true_labels(sm, fn_label=fn_label)
