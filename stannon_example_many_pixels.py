"""Script to train and test a Stannon model with label uncertainties on 
APOGEE test data. Originally written by Dr Andy Casey.
"""
import os
import numpy as np
import logging
import pickle
from astropy.table import Table
from tqdm import tqdm

import stannon.stan_utils as sutils
from stannon.vectorizer import PolynomialVectorizer

here = os.path.dirname(__file__)

DATA_PATH = os.path.join(here, "stannon/apogee")

# Load the training set labels.
training_set_labels = Table.read(os.path.join(DATA_PATH, "apogee-dr14-giants.fits"))

# Load the training set spectra.
with open(os.path.join(DATA_PATH, "apogee-dr14-giants-flux-ivar.pkl"), "rb") as fp:
    training_set_flux, training_set_ivar = pickle.load(fp, encoding="latin-1")

label_names = ("TEFF", "LOGG", "FE_H")
label_means = np.vstack([training_set_labels[ln] for ln in label_names]).T

# whiten the labels
mu = np.mean(label_means, axis=0)
sigma = np.std(label_means, axis=0)

whitened_label_means = (label_means - mu)/sigma
whitened_label_variances = 1e-3 * np.ones_like(whitened_label_means) # MAGIC HACK

S, P = training_set_flux.shape
L = len(label_names)

# Generate a pixel mask for fun.
pixel_mask = np.zeros(P, dtype=bool)
#pixel_mask[:] = True
pixel_mask[:100] = True

training_set_flux = training_set_flux[:, pixel_mask]
training_set_ivar = training_set_ivar[:, pixel_mask]
P = sum(pixel_mask)

model = sutils.read(os.path.join(here, "stannon/cannon-3L-O2-many-pixels.stan"))

theta = np.nan * np.ones((P, 10))
s2 = np.nan * np.ones(P)

# Whether to train all at once, or one pixel at a time
run_tqdm = True

if run_tqdm:
    data_dict = dict(S=S, P=1, L=L,
                    label_means=whitened_label_means,
                    label_variances=whitened_label_variances)

    init_dict = dict(true_labels=whitened_label_means,
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
                     label_means=whitened_label_means,
                     label_variances=whitened_label_variances)

    init_dict = dict(true_labels=whitened_label_means,
                     s2=np.ones(P),
                     theta=np.hstack([np.ones((P, 1)), np.zeros((P, 9))]))
    kwds = dict(data=data_dict, init=init_dict)

    optim = model.optimizing(**kwds)

    theta = optim["theta"]
    s2 = optim["s2"]

# Determine labels
labels_pred, errors, chi2 = sutils.infer_labels(theta, s2, training_set_flux, 
                                         training_set_ivar, mu, sigma) 

sutils.compare_labels(label_means, labels_pred)