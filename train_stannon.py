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
"""
import numpy as np
import plumage.utils as pu
import stannon.stannon as stannon
import stannon.plotting as splt

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
# Suppress ouput from Stan during training. It is recommended to always have
# this on for the label_uncertainties model since it takes much longer to
# converge than the less complex basic model.
suppress_stan_output = False

# Whether to initialise theta and s2 vectors for a label uncertainty model
# using the vectors from a trained basic model. The idea is that, even though
# these will ultimately be different, it's a better initial guess than just 
# starting with an array of zeroes. Initial testing bears this out, as the log
# probability starting points from a naive initial guess are radically
# different from those informed by a basic model (log prob -500,000 vs +260,000
# for a test case 842 px model)
init_with_basic_model = False

# The maximum amount of iterations Stan will run while fitting the model. The
# label_uncertainty model requires a higher number of iterations to converge,
# though preliminary testing indicates the model is ~mostly converge after the
# first ~10% of iterations and the remaining 90% are "fine tuning"--useful to
# know for testing purposes to save time. Note that the basic model is trained
# pixel-by-pixel with each pixel taking approximately a few hundred iterations
# to converge, so max_iter isn't really relevant here.
max_iter = 50000

# By default Stan only logs a fitting update once every max_iter/10 iterations.
# For large max_iter values, this might not be frequent enough--especially when
# testing--so this can be updated here. 
refresh_rate_frac = 1000
log_refresh_step = int(max_iter / refresh_rate_frac)

# Whether to run leave-one-out cross validation on Cannon model. If yes, the
# cross validation is done using the same suppress_stan_output, 
# init_with_basic_model, and max_iter settings as the original training.
do_cross_validation = False

# Whether to do sigma clipping using trained Cannon model. If True, an initial
# Cannon model is trained and its model spectra are used to sigma clip bad 
# pixels to not be considered for the subsequently trained and adopted model.
# Note that this can potentially cause unexpected results if the initial
# Cannon model is poorly trained, so best to leave as False when testing.
do_iterative_bad_px_masking = False
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
# For a 3-term model, the basic model (on motley) takes of order ~1 min to
# train, and the label uncertainties model takes of order ~20 min. The latter 
# increases to ~33 min when training a 4 term model with [Ti/H]. Note that
# these numbers are for a *single* model, and that cross validation increases
# the runtime by a factor of N_stars. TODO: update these numbers.
model_type = "label_uncertainties"

# For testing purposes, we can run with uniform label variances instead of the
# literature uncertainties. To do this, we assign every label the same
# percentage uncertainty. For instance, uniform_var_frac_error = 0.01 means
# that all labels have a 1% uncertainty. If this is set to False, we instead
# run with the observed label uncertainties.
use_label_uniform_variances = False
uniform_var_frac_error = 0.01

# Also for testing we can rescale the literature uncertainties by a constant
# amount. For instance, lit_std_scale_fac = 0.1 means that we adopt
# uncertainties 10x smaller than observed. This only takes effect if 
# use_label_uniform_variances is set to False. Set to 1.0 if we don't want to
# scale the uncertainties.
lit_std_scale_fac = 1.0

model_save_path = "spectra"
std_label = "cannon"

# Whether to fit for abundances. At the moment our abundance heirarchy is
# Montes+18 > Valenti+Fischer05 > Adibekyan+12. Not recommended to fit > 1-2.
# Available options (for Montes+18, which is the most complete): 
# Na, Mg, Al, Si, Ca, Sc, Ti, V, Cr, Mn, Co, Ni
# Select as e.g.["X_H",..] or leave empty to not use abundances.
abundance_labels = ["Ti_H"]

label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = len(label_names)

#------------------------------------------------------------------------------
# Training labels
#------------------------------------------------------------------------------
# Import dataframe with benchmark parameters
obs_join = pu.load_fits_table("CANNON_INFO", "cannon")

is_cannon_benchmark = obs_join["is_cannon_benchmark"].values

# Grab benchmark labels
label_value_cols = []
label_var_cols = []

for lbl_i, lbl in enumerate(label_names):
    label_value_cols.append("label_adopt_{}".format(lbl))
    label_var_cols.append("label_adopt_var_{}".format(lbl))

label_values_all = obs_join[label_value_cols].values
label_var_all = obs_join[label_var_cols].values * lit_std_scale_fac**2

# Optional for testing: run with uniform variances
if use_label_uniform_variances:
    # Unlog logg, [Fe/H], and any abundances
    unlogged_label_values_all = label_values_all.copy()
    unlogged_label_values_all[:,1:] = 10**label_values_all[:,1:]

    # Calculate test variances for all labels based on uniform_var_frac_error
    label_std_all = unlogged_label_values_all * uniform_var_frac_error

    # Relog logg, [Fe/H], and any abundances
    label_std_all[:,1:] = \
        (label_std_all[:,1:] / (np.log(10) * unlogged_label_values_all[:,1:]))
    
    label_var_all = label_std_all**2

#------------------------------------------------------------------------------
# Flux preparation
#------------------------------------------------------------------------------
# Load in RV corrected standard spectra
wls = pu.load_fits_image_hdu("rest_frame_wave", std_label, arm="br")
spec_std_br = pu.load_fits_image_hdu("rest_frame_spec", std_label, arm="br")
e_spec_std_br = pu.load_fits_image_hdu("rest_frame_sigma", std_label, arm="br")

fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wls,
        spectra=spec_std_br[is_cannon_benchmark],
        e_spectra=e_spec_std_br[is_cannon_benchmark],
        wl_min_model=wl_min_model,
        wl_max_model=wl_max_model,
        wl_min_normalisation=wl_min_normalisation,
        wl_broadening=wl_broadening,
        do_gaussian_spectra_normalisation=do_gaussian_spectra_normalisation,
        poly_order=poly_order)

#------------------------------------------------------------------------------
# Make and train model
#------------------------------------------------------------------------------
# Diagnostic summary
print("\n\n", "%"*80, "\n", sep="")
print("\tModel Params:\n\t", "-"*21, sep="")
print("\tmodel: \t\t\t = {}".format(model_type))
print("\tlambda: \t\t = {:0.0f}-{:0.0f} A".format(wl_min_model, wl_max_model))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(label_names)))
print("\tlabels: \t\t = {}".format(label_names))
print("\tn benchmarks: \t\t = {:0.0f}".format(np.sum(is_cannon_benchmark)))
print("\tGaussian Normalisation:\t = {}".format(
    do_gaussian_spectra_normalisation))
if do_gaussian_spectra_normalisation:
    print("\twl broadening: \t\t = {:0.0f} Å".format(wl_broadening))
else:
    print("\tpoly order: \t\t = {:0.0f}".format(poly_order))

print("\n\tTraining Params:\n\t", "-"*21, sep="")
print("\tcross validation: \t = {}".format(do_cross_validation))
print("\titerative masking: \t = {}".format(do_iterative_bad_px_masking))
print("\tinit with basic model: \t = {}".format(init_with_basic_model))

print("\n\tTesting Params:\n\t", "-"*21, sep="")
print("\tuniform variances:\t = {}".format(use_label_uniform_variances))
print("\tuniform var scale fac:\t = {}".format(uniform_var_frac_error))
print("\tlit std scale fac:\t = {}".format(lit_std_scale_fac))

print("\n", "%"*80, "\n\n", sep="")

# Make model
sm = stannon.Stannon(
    training_data=fluxes_norm,
    training_data_ivar=ivars_norm,
    training_labels=label_values_all[is_cannon_benchmark],
    training_ids=obs_join[is_cannon_benchmark].index.values,
    label_names=label_names,
    wavelengths=wls,
    model_type=model_type,
    training_variances=label_var_all[is_cannon_benchmark],
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask,)

# Train model
print("\n\n", "-"*100, sep="",)
print("\nFitting initial Cannon model with {} benchmarks\n".format(
    np.sum(is_cannon_benchmark)))
print("-"*100, "\n\n", sep="",)
sm.train_cannon_model(
    suppress_stan_output=suppress_stan_output,
    init_uncertainty_model_with_basic_model=init_with_basic_model,
    max_iter=max_iter,
    log_refresh_step=log_refresh_step,)

# If we run the iterative bad px masking, train again afterwards
if do_iterative_bad_px_masking:
    print("\n\n", "-"*100, sep="",)
    print("\nFitting sigma clipped Cannon model based on initial model\n")
    print("-"*100, "\n\n", sep="",)
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=flux_sigma_to_clip)
    sm.train_cannon_model(
        suppress_stan_output=suppress_stan_output,
        init_uncertainty_model_with_basic_model=init_with_basic_model,
        max_iter=max_iter,
        log_refresh_step=log_refresh_step,)

# Run cross validation
if do_cross_validation:
    sm.run_cross_validation(
        suppress_stan_output=suppress_stan_output,
        init_uncertainty_model_with_basic_model=init_with_basic_model,
        max_iter=max_iter,
        log_refresh_step=log_refresh_step,)

    labels_pred = sm.cross_val_labels

# ...or just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(model_save_path)

print("Model training complete!")

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------
fn_label = "_{}_{}_label_{}".format(
    model_type, len(label_names), "_".join(label_names))

uncertainties = np.nanstd(sm.training_labels - labels_pred, axis=0)

summary = \
    "\nΔ Teff = {:0.0f} K, Δlogg = {:0.2f} dex, Δ[Fe/H] = {:0.2f} dex"
print("Training set uncertainties:\n", summary.format(*tuple(uncertainties)))

# Compute the difference between true and input labels
if model_type == "label_uncertainties":
    splt.plot_label_uncertainty_adopted_vs_true_labels(sm, fn_label=fn_label)
