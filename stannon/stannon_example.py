

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
labels = np.vstack([training_set_labels[ln] for ln in label_names]).T

# whiten the labels
mu = np.mean(labels, axis=0)
sigma = np.mean(labels, axis=0)

whitened_labels = (labels - mu)/sigma

S, P = training_set_flux.shape
L = len(label_names)

# Generate a pixel mask for fun.
pixel_mask = np.zeros(P, dtype=bool)
pixel_mask[250:350] = True

training_set_flux = training_set_flux[:, pixel_mask]
training_set_ivar = training_set_ivar[:, pixel_mask]
P = sum(pixel_mask)


# This is a weird way to build a design matrix  but life is short
vectorizer = PolynomialVectorizer(label_names, 2)
design_matrix = vectorizer(whitened_labels)


model = stan.read(os.path.join(here, f"cannon-{L:.0f}L-O2.stan"))


theta = np.nan * np.ones((P, 10))
s2 = np.nan * np.ones(P)

data_dict = dict(P=1, S=S, DM=design_matrix.T)
init_dict = dict(theta=np.atleast_2d(np.hstack([1, np.zeros(theta.shape[1] - 1)])), s2=1)


for p in tqdm(range(P), desc="Training"):

    data_dict.update(y=training_set_flux[:, p],
                     y_var=1/training_set_ivar[:, p])


    kwds = dict(data=data_dict, init=init_dict)

    # Suppress Stan output. This is dangerous!
    with stan.suppress_output() as sm:
        #if True:
        try:
            p_opt = model.optimizing(**kwds)

        except:
            logging.exception(f"Exception occurred when optimizing pixel index {p}")

        else:
            if p_opt is not None:
                theta[p] = p_opt["theta"]
                s2[p] = p_opt["s2"]



import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(4, 16))
for i, ax in enumerate(axes):
    ax.plot(theta[:,  i])



fig, ax = plt.subplots()
ax.plot(s2**0.5)
