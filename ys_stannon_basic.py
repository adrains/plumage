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
    assign_default_uncertainties=True,
    force_solar_missing_feh=True)

# Prepare the training set
std_obs, std_spec_b, std_spec_r, std_params = utils.prepare_training_set(
    observations, 
    spectra_b,
    spectra_r, 
    std_params_all, 
    do_wavelength_masking=True
    )   

wls = np.concatenate((std_spec_b[0,0,:],std_spec_r[0,0,:]))

training_set_flux, training_set_ivar = utils.prepare_fluxes(
    std_spec_b, 
    std_spec_r, 
    use_both_arms)

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
px_min = 0#P-300
px_max = 500#P
pixel_mask = np.zeros(P, dtype=bool)
pixel_mask[px_min:px_max] = True

wls = wls[pixel_mask]
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

#------------------------------------------------------------------------------
# Make diagnostic plots
#------------------------------------------------------------------------------
# Plot Teff comparison
plt.figure()
plt.scatter(label_values[:,0],labels_pred[:,0], c=label_values[:,2],marker="o")
plt.plot(np.arange(2500,5500),np.arange(2500,5500),"-",color="black")
plt.xlabel(r"T$_{\rm eff}$ (Lit)")
plt.ylabel(r"T$_{\rm eff}$ (Cannon)")
cb = plt.colorbar()
cb.set_label(r"[Fe/H]")
plt.xlim([2800,5100])
plt.ylim([2800,5100])
plt.savefig("plots/presentations/ms_teff_vs_teff.png",fpi=300)

# Plot logg comparison
plt.figure()
plt.scatter(label_values[:,1],labels_pred[:,1], c=label_values[:,0],marker="o")
plt.plot(np.arange(3.5,5.5,0.1),np.arange(3.5,5.5,0.1),"-",color="black")
plt.xlim([4.0,5.1])
plt.ylim([4.0,5.1])
plt.ylabel(r"$\log g$ (Cannon)")
plt.xlabel(r"$\log g$ (Lit)")
cb = plt.colorbar()
cb.set_label(r"[Fe/H]")
plt.savefig("plots/presentations/ms_logg_vs_logg.png",fpi=300)

# Plot Fe/H comparison
plt.figure()
plt.scatter(label_values[:,2],labels_pred[:,2], c=label_values[:,0],marker="o",
            cmap="magma") 
plt.plot(np.arange(-0.6,0.5,0.05),np.arange(-0.6,0.5,0.05),"-",color="black")
plt.xlabel(r"[Fe/H] (Lit)")
plt.ylabel(r"[Fe/H] (Cannon)") 
cb = plt.colorbar() 
cb.set_label(r"T$_{\rm eff}$")
plt.savefig("plots/presentations/ms_feh_vs_feh.png",fpi=300)

# Plot of theta coefficients
fig, axes = plt.subplots(4, 1, sharex=True)
axes = axes.flatten()

for star in training_set_flux:
    axes[0].plot(wls, star, linewidth=0.2)

axes[1].plot(wls, theta[:,1], linewidth=0.5)
axes[2].plot(wls, theta[:,2], linewidth=0.5)
axes[3].plot(wls, theta[:,3], linewidth=0.5)

axes[0].set_ylim([0,3])
axes[1].set_ylim([-0.5,0.5])
axes[2].set_ylim([-0.5,0.5])
axes[3].set_ylim([-0.1,0.1])

axes[0].set_ylabel(r"Flux")
axes[1].set_ylabel(r"$\theta$ T$_{\rm eff}$")
axes[2].set_ylabel(r"$\theta$ $\log g$")
axes[3].set_ylabel(r"$\theta$ $[Fe/H]$")
plt.xlabel("Wavelength (A)")

#------------------------------------------------------------------------------
# Test against TOIs
#------------------------------------------------------------------------------
"""
toi_obs, toi_spec_b, toi_spec_r, toi_cat = utils.prepare_training_set(
    observations, 
    spectra_b,
    spectra_r, 
    toi_cat, 
    do_wavelength_masking=True
    )

toi_flux, toi_ivar = utils.prepare_fluxes(toi_spec_b, toi_spec_r, True)

toi_flux = toi_flux[:, pixel_mask]
toi_ivar = toi_ivar[:, pixel_mask]

toi_labels_pred, toi_errors, toi_chi2 = sutils.infer_labels(theta, s2, toi_flux, 
                                         toi_ivar, ts_mean_labels, 
                                         ts_stdev_labels)

toi_obs["teff_cannon"] = toi_labels_pred[:,0]
toi_obs["logg_cannon"] = toi_labels_pred[:,1]
toi_obs["feh_cannon"] = toi_labels_pred[:,2]

# Check uncertainties
toi_e_teff = np.abs(toi_obs["teff_cannon"] - toi_obs["teff_fit"])
print("Mean Teff error:", toi_e_teff.mean())
print("Mean Teff error:", toi_e_teff.median())

# plotting 
plt.scatter(toi_obs["teff_cannon"], toi_obs["logg_cannon"], c=toi_obs["feh_cannon"])
cb = plt.colorbar()
cb.set_label(r"[Fe/H]")
plt.xlim([5250,2900])
plt.ylim([5.1,4])
plt.xlabel(r"T$_{\rm eff}$")
plt.ylabel(r"$\log g$")
plt.scatter(std_params["teff"], std_params["logg"], c="white", marker="*", edgecolors="black")
plt.tight_layout()
plt.savefig("plots/presentations/toi_cannon.png",fpi=300)
"""