"""This file contains the common set of parameters used to train and plot 
diagnostic plots for a Cannon model.

This is part of a series of Cannon scripts. The main sequence is:
 1) prepare_stannon_training_sample.py     --> label preparation
 2) train_stannon.py                       --> training and cross validation
 3) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 4) run_stannon.py                         --> running on science spectra

Where the values in this file currently apply to items 2) and 3).

TODO: redo this file in a more robust format than just an import, e.g. YAML.
"""
#------------------------------------------------------------------------------
# Model Settings
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
max_iter = 100000

# By default Stan only logs a fitting update once every max_iter/10 iterations.
# For large max_iter values, this might not be frequent enough--especially when
# testing--so this can be updated here. 
refresh_rate_frac = 1000
log_refresh_step = int(max_iter / refresh_rate_frac)

# Whether to run leave-one-out cross validation on Cannon model. If yes, the
# cross validation is done using the same suppress_stan_output, 
# init_with_basic_model, and max_iter settings as the original training.
do_cross_validation = False
is_cross_validated = do_cross_validation # alt name for clarity when plotting

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
wl_min_model = 5400
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

# To constrain the parameter space (or just for testing) when using the label
# uncertainties model, we can rescale the literature uncertainties by a
# constant amount on a per-label basis. For instance, setting 
# lit_std_scale_fac = [0.1, 0.1, 0.1, 0.1] means that we adopt
# uncertainties 10x smaller than observed for each label of a four label model.
# This only takes effect if use_label_uniform_variances is set to False. Set to 
# an array of ones if we don't want to scale the uncertainties.
lit_std_scale_fac = [1, 1, 1, 1]

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
# Plotting Settings
#------------------------------------------------------------------------------
# Models are currently saved  with N_px in the filename.
# TODO: come up with a more informative/robust labelling scheme.
npx = 454

# The grating at which the WiFeS B3000 spectra transition to R7000 spectra to
# use when plotting separate 'b' and 'r' plots.
wl_grating_changeover = 5400

# If this is true, we only plot the first order/linear theta coefficients. If
# False, we include extra panels for the cross and quadtratic terms
only_plot_first_order_coeff = True