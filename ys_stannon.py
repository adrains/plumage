"""Script to train and test a Stannon model with label uncertainties on 
APOGEE test data. Originally written by Dr Andy Casey.
"""
import os
import numpy as np
import logging
import pickle
from astropy.table import Table
from tqdm import tqdm

import plumage.spectra as spec
import plumage.utils as utils
import stannon.stan_utils as sutils
from stannon.vectorizer import PolynomialVectorizer

here = os.path.dirname(__file__)
#------------------------------------------------------------------------------
# Prepare training data
#------------------------------------------------------------------------------
print("Importing young star training set...")
# Import data
observations, spectra_b, spectra_r = spec.load_pkl_spectra(516, rv_corr=True)

# Load in standards
standards = utils.load_standards(remove_standards_with_nan_params=True)  

# Get the parameters
std_params_all = utils.consolidate_standards(
    standards, 
    force_unique=True,
    assign_default_uncertainties=True)

# Prepare the training set
std_observations, std_spectra_r, std_params = utils.prepare_training_set(
    observations, 
    spectra_r, 
    std_params_all, 
    do_wavelength_masking=True
    )   

# Separate out the spectra
training_set_flux = std_spectra_r[:,1,:]
#training_set_ivar = 1/std_spectra_r[:,2,:]**2 / 3
training_set_ivar = 1e5 * np.ones_like(std_spectra_r[:,2,:])

# If flux is nan, set to 1 and give high variance (inverse variance of 0)
training_set_ivar[~np.isfinite(training_set_flux)] = 0
training_set_flux[~np.isfinite(training_set_flux)] = 1

# If the inverse variance is nan, do the same
training_set_flux[~np.isfinite(training_set_ivar)] = 1
training_set_ivar[~np.isfinite(training_set_ivar)] = 0

# Edges of spectra appear to cause issues, let's just not consider them
training_set_flux = training_set_flux[:,5:-5]
training_set_ivar = training_set_ivar[:,5:-5]

#------------------------------------------------------------------------------
# Format the training data
#------------------------------------------------------------------------------
label_names = ["teff", "logg", "feh"]
label_e_names = ["e_teff", "e_logg", "e_feh"]
label_values = std_params[label_names].values
label_stdevs = std_params[label_e_names].values

# Whiten the labels
ts_mean_labels = np.nanmean(label_values, axis=0)
ts_stdev_labels = np.nanstd(label_values, axis=0)

whitened_label_values = (label_values - ts_mean_labels)/ts_stdev_labels
whitened_label_variances = 1e-3 * np.ones_like(whitened_label_values)
#whitened_label_variances = (label_stdevs - ts_mean_labels)/ts_stdev_labels**2

S, P = training_set_flux.shape
L = len(label_names)

# Generate a pixel mask for scaling/testing the training
px_min = 1100
px_max = 1150
pixel_mask = np.zeros(P, dtype=bool)
pixel_mask[px_min:px_max] = True

training_set_flux = training_set_flux[:, pixel_mask]
training_set_ivar = training_set_ivar[:, pixel_mask]
P = sum(pixel_mask)

model = sutils.read(os.path.join(here, "stannon/cannon-3L-O2-many-pixels.stan"))

theta = np.nan * np.ones((P, 10))
s2 = np.nan * np.ones(P)

#------------------------------------------------------------------------------
# Run the Cannon
#------------------------------------------------------------------------------
# Whether to train all at once, or one pixel at a time
run_tqdm = False

print("Running Stannon on %i pixels" % len(training_set_flux))
if run_tqdm:
    data_dict = dict(S=S, P=1, L=L,
                    label_means=whitened_label_values,
                    label_variances=whitened_label_variances)

    init_dict = dict(true_labels=whitened_label_values,
                     s2=[1],
                     theta=np.atleast_2d(np.hstack([1, np.zeros(theta.shape[1] - 1)])))

    # Train the model one pixel at a time
    for p in tqdm(range(P), desc="Training"):
        data_dict.update(y=np.atleast_2d(training_set_flux[:, p]).T,
                         y_var=1/np.atleast_2d(training_set_ivar[:, p]).T)

        kwds = dict(data=data_dict, init=init_dict)

        # Suppress Stan output. This is dangerous!
        with sutils.suppress_output() as sm:
            try:
                p_opt = model.optimizing(**kwds)

            except:
                logging.exception(f"Exception occurred when optimizing pixel index {p}")

            else:
                if p_opt is not None:
                    theta[p] = p_opt["theta"]
                    s2[p] = p_opt["s2"]

else:
    data_dict = dict(S=S, P=P, L=L,
                     y=training_set_flux,
                     y_var=1/training_set_ivar,
                     label_means=whitened_label_values,
                     label_variances=whitened_label_variances)

    init_dict = dict(true_labels=whitened_label_values,
                     s2=np.ones(P),
                     theta=np.hstack([np.ones((P, 1)), np.zeros((P, 9))]))
    kwds = dict(data=data_dict, init=init_dict)

    optim = model.optimizing(**kwds)

    theta = optim["theta"]
    s2 = optim["s2"]

#------------------------------------------------------------------------------
# Test the Cannon model
#------------------------------------------------------------------------------
labels_pred, errors, chi2 = sutils.infer_labels(theta, s2, training_set_flux, 
                                         training_set_ivar, ts_mean_labels, 
                                         ts_stdev_labels) 

sutils.compare_labels(ts_mean_labels, labels_pred)