"""Function to fit polynomials to the MIKE normalised flat field sensitivity
function which are the blaze convolved with the flat lamp SED. We then save the
fitted coefficients to a CSV as:

<base_path>/mike_norm_flat_poly_coef_<poly_order>_<arm>.csv
"""
import os
import numpy as np
import pandas as pd
import plumage.spectra_mike as sm
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# This is the multi-fits file used during the flat field fit
ref_spec = "spectra/mike_MK_unnormalised/ej131red_multi.fits"

# Polynomial oder for the fitting
poly_order = 7

# Arm of the spectrograph fitted so we can label the saved file accordingly
arm = "r"

# Folder to save to
base_path = "data/"

# -----------------------------------------------------------------------------
# Observation settings
# -----------------------------------------------------------------------------
# Import the reference spectrum
data_dict = sm.load_single_mike_fits(ref_spec)

# Import the wavelength scale, normalised flat spectra, and order numbers
wave = data_dict["wave"]
flat = data_dict["spec_flat"]
orders = data_dict["orders"]

# Create a mask for wavelengths where the flat has zero-flux
is_zero = flat == 0

# -----------------------------------------------------------------------------
# Polynomial Fitting
# -----------------------------------------------------------------------------
# Grab dimensions for convenience
(n_order, n_px) = wave.shape

# Initialise output array to save fitted polynomial coefficients
poly_coef = np.zeros((n_order, poly_order+1))

# Keep track of the polynomial domains that we use when fitting. These are the
# min and max extent of the non-zero pixels, beyond which we should not
# consider the polynomial valid. The Polynomial class/fit transforms from
# domain --> window, where window is by default [-1,1].
domain_min = np.zeros(n_order)
domain_max = np.zeros(n_order)

plt.close()
fig, axes = plt.subplots(figsize=(10,6))

# Fit polynomials to all orders, and plot the results
for order_i in range(n_order):
    ww = wave[order_i][~is_zero[order_i]]
    fl = flat[order_i][~is_zero[order_i]]

    domain_min[order_i] = np.min(ww)
    domain_max[order_i] = np.max(ww)

    poly = Polynomial.fit(
        x=ww,
        y=fl,
        deg=poly_order,
        domain=(domain_min[order_i], domain_max[order_i]),)
    
    poly_coef[order_i] = poly.coef

    axes.plot(
        wave[order_i],
        flat[order_i], 
        linewidth=0.5,
        c="k",
        label="Raw" if order_i == 0 else None)
    
    axes.plot(
        ww,
        poly(ww),
        linewidth=0.5,
        c="r",
        label="Fit (order: {})".format(poly_order) if order_i == 0 else None)

leg = axes.legend()
axes.set_xlabel("Wavelength (Å)")
axes.set_ylabel("Normalised Flat")
plt.tight_layout()
plt.savefig("plots/order_fit_{}_order_{}.pdf".format(arm, poly_order))

# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
# Create DataFrame
coef_cols = ["coef_{}".format(oi) for oi in range(0, poly_order+1, 1)]
cols = coef_cols + ["domain_min", "domain_max"]

data = np.hstack((poly_coef, domain_min[:, None], domain_max[:, None]))

df = pd.DataFrame(index=pd.Index(orders, name="orders"), data=data, columns=cols)

# Save the results to a CSV
fn = "mike_norm_flat_poly_coef_{}_{}.csv".format(poly_order, arm)
df.to_csv(os.path.join(base_path, fn))