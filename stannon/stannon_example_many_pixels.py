

import os
import numpy as np
import logging
import pickle
from astropy.table import Table
from tqdm import tqdm

import stan_utils as stan
from vectorizer import PolynomialVectorizer

here = os.path.dirname(__file__)

DATA_PATH = os.path.join(here, "data")

# Load the training set labels.
training_set_labels = Table.read(os.path.join(DATA_PATH, "apogee-dr14-giants.fits"))

# Load the training set spectra.
with open(os.path.join(DATA_PATH, "apogee-dr14-giants-flux-ivar.pkl"), "rb") as fp:
    training_set_flux, training_set_ivar = pickle.load(fp, encoding="latin-1")

label_names = ("TEFF", "LOGG", "FE_H")
label_means = np.vstack([training_set_labels[ln] for ln in label_names]).T



# whiten the labels
mu = np.mean(label_means, axis=0)
sigma = np.mean(label_means, axis=0)

whitened_label_means = (label_means - mu)/sigma
whitened_label_variances = 1e-3 * np.ones_like(whitened_label_means) # MAGIC HACK

S, P = training_set_flux.shape
L = len(label_names)

# Generate a pixel mask for fun.
pixel_mask = np.zeros(P, dtype=bool)
pixel_mask[250:350] = True

training_set_flux = training_set_flux[:, pixel_mask]
training_set_ivar = training_set_ivar[:, pixel_mask]
P = sum(pixel_mask)




model = stan.read("cannon-3L-O2-many-pixels.stan")

data_dict = dict(S=S, P=P, L=L,
                 y=training_set_flux,
                 y_var=1/training_set_ivar,
                 label_means=whitened_label_means,
                 label_variances=whitened_label_variances)

init_dict = dict(true_labels=whitened_label_means,
                 s2=np.ones(P),
                 theta=np.hstack([np.ones((P, 1)), np.zeros((P, 9))]))



kwds = dict(data=data_dict, init=init_dict)


foo = model.optimizing(**kwds)




raise a