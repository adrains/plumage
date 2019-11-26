"""Script to train and test a simple Stannon model on APOGEE test data. 
Originally written by Dr Andy Casey.
"""
import os
import numpy as np
import logging
import pickle
from astropy.table import Table
from tqdm import tqdm

import plumage.spectra as spec
import plumage.utils as utils
import matplotlib.pyplot as plt
import stannon.stan_utils as sutils
from stannon.vectorizer import PolynomialVectorizer

here = os.path.dirname(__file__)

use_both_arms = True

#------------------------------------------------------------------------------
# Prepare training data
#------------------------------------------------------------------------------
print("Importing young star training set...")
# Import data
observations, spectra_b, spectra_r = spec.load_pkl_spectra(516, rv_corr=True)

# Load in standards
standards = utils.load_standards() 

# Limit the parameter space (set to None if no limitations)
teff_lims = (2500, 5250)
logg_lims = (4, 5.1)
feh_lims = (-0.75, 1.0)

# Get the parameters
std_params_all = utils.consolidate_standards(
    standards, 
    force_unique=True,
    remove_standards_with_nan_params=True,
    teff_lims=teff_lims,
    logg_lims=logg_lims, 
    feh_lims=feh_lims,
    assign_default_uncertainties=True)

# Prepare the training set
std_obs, std_spec_b, std_spec_r, std_params = utils.prepare_training_set(
    observations, 
    spectra_b,
    spectra_r, 
    std_params_all, 
    do_wavelength_masking=True
    )   

# Separate out the spectra
if use_both_arms:
    training_set_flux = np.concatenate((std_spec_b[:,1,:], std_spec_r[:,1,:]), 
                                       axis=1)
    training_set_ivar = np.concatenate((1/std_spec_b[:,2,:]**2, 
                                       1/std_spec_r[:,2,:]**2), axis=1)
else:
    training_set_flux = std_spec_r[:,1,:]
    training_set_ivar = 1/std_spec_r[:,2,:]**2
#training_set_ivar = 1e5 * np.ones_like(std_spectra_r[:,2,:]) # TEMPORARY

# If flux is nan, set to 1 and give high variance (inverse variance of 0)
training_set_ivar[~np.isfinite(training_set_flux)] = 1e-8
training_set_flux[~np.isfinite(training_set_flux)] = 1

# If the inverse variance is nan, do the same
training_set_flux[~np.isfinite(training_set_ivar)] = 1
training_set_ivar[~np.isfinite(training_set_ivar)] = 1e-8

# Edges of spectra appear to cause issues, let's just not consider them
#min_px = 6
#training_set_flux = training_set_flux[:,5:-5]
#training_set_ivar = training_set_ivar[:,5:-5]/

#------------------------------------------------------------------------------
# Stannon stuff
#------------------------------------------------------------------------------
label_names = ["teff", "logg", "feh"]
#label_e_names = ["e_teff", "e_logg", "e_feh"]
label_values = std_params[label_names].values
#label_stdevs = std_params[label_e_names].values

# Whiten the labels
ts_mean_labels = np.nanmean(label_values, axis=0)
ts_stdev_labels = np.nanstd(label_values, axis=0)

whitened_label_values = (label_values - ts_mean_labels)/ts_stdev_labels

S, P = training_set_flux.shape
L = len(label_names)

# Generate a pixel mask for scaling/testing the training
px_min = 3700
px_max = 3850
pixel_mask = np.zeros(P, dtype=bool)
pixel_mask[px_min:px_max] = True

training_set_flux = training_set_flux[:, pixel_mask]
training_set_ivar = training_set_ivar[:, pixel_mask]
P = sum(pixel_mask)


#------------------------------------------------------------------------------
# Run the Cannon
#------------------------------------------------------------------------------
# This is a weird way to build a design matrix  but life is short
vectorizer = PolynomialVectorizer(label_names, 2)
design_matrix = vectorizer(whitened_label_values)

model = sutils.read(os.path.join(here, "stannon", f"cannon-{L:.0f}L-O2.stan"))

theta = np.nan * np.ones((P, 10))
s2 = np.nan * np.ones(P)

data_dict = dict(P=1, S=S, DM=design_matrix.T)
init_dict = dict(theta=np.atleast_2d(np.hstack([1, np.zeros(theta.shape[1] - 1)])), 
                 s2=1)

for p in tqdm(range(P), desc="Training"):

    data_dict.update(y=training_set_flux[:, p],
                     y_var=1/training_set_ivar[:, p])

    kwds = dict(data=data_dict, init=init_dict)

    # Suppress Stan output. This is dangerous!
    with sutils.suppress_output() as sm:
        #if True:
        try:
            p_opt = model.optimizing(**kwds)

        except:
            logging.exception(f"Exception occurred when optimizing pixel index {p}")

        else:
            if p_opt is not None:
                theta[p] = p_opt["theta"]
                s2[p] = p_opt["s2"]

#------------------------------------------------------------------------------
# Test the Cannon model
#------------------------------------------------------------------------------
# Plot diagnostic plots
fig, axes = plt.subplots(4, 1, figsize=(4, 16))
for i, ax in enumerate(axes):
    ax.plot(theta[:,  i])

fig, ax = plt.subplots()
ax.plot(s2**0.5)

# Infer labels
labels_pred, errors, chi2 = sutils.infer_labels(theta, s2, training_set_flux, 
                                         training_set_ivar, ts_mean_labels, 
                                         ts_stdev_labels) 
sutils.compare_labels(label_values, labels_pred)
