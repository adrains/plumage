"""Script to train, cross validate, and save a Cannon model. After a model has
been created, the scripts make_stannon_diagnostics.py and run_stannon.py should
be used.

This script is part of a series of Cannon scripts. The main sequence is:
 1) prepare_stannon_training_sample.py     --> label preparation
 2) train_stannon.py                       --> training and cross validation
 3) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 4) run_stannon.py                         --> running on science spectra
"""
import numpy as np
import plumage.utils as pu
import stannon.stannon as stannon

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
# Suppress ouput from Stan during training (recommended True unless debugging)
suppress_stan_output = False

# Whether to initialise theta and s2 vectors for a label uncertainty model
# using the vectors from a trained basic model. The idea is that, even though
# these will ultimately be different, it's a better initial guess than just 
# starting with an array of zeroes.
init_with_basic_model = True

# The maximum amount of iterations Stan will run while fitting the model
max_iter = 100000

# Whether to run leave-one-out cross validation on Cannon model
do_cross_validation = False

# Whether to do sigma clipping using trained Cannon model. If True, an initial
# Cannon model is trained and its model spectra are used to sigma clip bad 
# pixels to not be considered for the subsequently trained and adopted model.
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
wl_min_model = 5400
wl_max_model = 7000

# The Cannon model to use - either the 'basic' traditional Cannon model, or a
# model with label uncertainties. If modelling abundances, the version with
# label uncertainties should be used. Either 'basic' or 'label_uncertainties'.
# For a 3-term model, the basic model (on motley) takes of order ~1 min to
# train, and the label uncertainties model takes of order ~20 min. The latter 
# increases to ~33 min when training a 4 term model with [Ti/H]. Note that
# these numbers are for a *single* model, and that cross validation increases
# the runtime by a factor of N_stars.
model_type = "label_uncertainties"
use_label_uniform_variances = False

model_save_path = "spectra"
std_label = "cannon"

# Whether to fit for abundances. At the moment our abundance heirarchy is
# Montes+18 > Valenti+Fischer05 > Adibekyan+12. Not recommended to fit > 1-2.
# Available options (for Montes+18, which is the most complete): 
# Na, Mg, Al, Si, Ca, Sc, Ti, V, Cr, Mn, Co, Ni
# Select as e.g.["X_H",..] or leave empty to not use abundances.
abundance_labels = []

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
label_var_all = obs_join[label_var_cols].values

# TODO HACK: unlog logged labels and propagate their uncertainties
label_values_all[:,1:] = 10**label_values_all[:,1:]
label_var_all[:,1:] = \
    (label_var_all[:,1:]**0.5 * np.log(10) * label_values_all[:,1:])**2

# Optional for testing: run with uniform variances
if use_label_uniform_variances:
    label_var_all = label_var_all*0 + 1e-3  # NaN values will remain NaN

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
print("\tRunning Cannon model:\n\t", "-"*21, sep="")
print("\tmodel: \t\t\t = {}".format(model_type))
print("\tlambda: \t\t\t = {:0.0f}-{:0.0f} A".format(wl_min_model, wl_max_model))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(label_names)))
print("\tlabels: \t\t = {}".format(label_names))
print("\tn benchmarks: \t\t = {:0.0f}".format(np.sum(is_cannon_benchmark)))
print("\tGaussian Normalisation:\t = {}".format(
    do_gaussian_spectra_normalisation))
if do_gaussian_spectra_normalisation:
    print("\twl broadening: \t\t = {:0.0f} A".format(wl_broadening))
else:
    print("\tpoly order: \t\t = {:0.0f}".format(poly_order))
print("\tuniform variances: \t = {}".format(use_label_uniform_variances))
print("\tcross validation: \t = {}".format(do_cross_validation))
print("\titerative masking: \t = {}".format(do_iterative_bad_px_masking))
print("\tinit with basic model: \t = {}".format(init_with_basic_model))
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
print("\nRunning initial training with {} benchmarks...".format(
    np.sum(is_cannon_benchmark)))
sm.train_cannon_model(
    suppress_stan_output=suppress_stan_output,
    init_uncertainty_model_with_basic_model=init_with_basic_model,
    max_iter=max_iter,)

# If we run the iterative bad px masking, train again afterwards
if do_iterative_bad_px_masking:
    print("\nRunning iterative sigma clipping for bad px...")
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=flux_sigma_to_clip)
    sm.train_cannon_model(suppress_stan_output=suppress_stan_output)

# Run cross validation
if do_cross_validation:
    sm.run_cross_validation()

    labels_pred = sm.cross_val_labels

# ...or just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(model_save_path)

print("Model training complete!")

# TODO: relog labels
sm_labels = sm.training_labels.copy()
sm_labels[:,1:] = np.log10(sm_labels[1,1:])
labels_pred[:,1:] = np.log10(labels_pred[:,1:])
uncertainties = np.std(sm_labels - labels_pred, axis=0)  