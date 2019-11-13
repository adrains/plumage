"""Script to process science spectra

Link if the IERS is iffy: https://github.com/astropy/astropy/issues/8981
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import plumage.synthetic as synth
import plumage.spectra as spec
import plumage.plotting as pplt
import plumage.utils as utils
from astropy.table import Table

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_spectra = False                    # Whether to load in pickle spectra
n_spec = 516                            # If loading, which pickle of N spectra
disable_auto_max_age = False            # Useful if IERS broken

flux_corrected = True                   # If *all* data is fluxed
telluric_corrected = True               # If *all* data is telluric corr

cat_type="csv"                          # Crossmatch catalogue type
cat_file="data/all_2m3_star_ids.csv"    # Crossmatch catalogue 

# -----------------------------------------------------------------------------
# Load in extracted 1D spectra from fits files or pickle
# -----------------------------------------------------------------------------
# 1D extracted spectra fits files have extensions as follows:
#  1) uncalibrated
#  2) fluxed
#  3) telluric
# Here is where we pick the extension for our science data
if not flux_corrected and not telluric_corrected:
    ext_sci = 1
elif flux_corrected and not telluric_corrected:
    ext_sci = 2
elif flux_corrected and telluric_corrected:
    ext_sci = 3
else:
    ext_sci = 1

if load_spectra:
    # Load in science spectra
    print("Importing science spectra...")
    observations, spectra_b, spectra_r = spec.load_pkl_spectra(n_spec) 

else:
    # Do initial import
    print("Doing inital spectra import...")
    observations, spectra_b, spectra_r = spec.load_all_spectra(ext_sci=ext_sci)
    spec.save_pkl_spectra(observations, spectra_b, spectra_r)

# -----------------------------------------------------------------------------
# Normalise science and template spectra
# -----------------------------------------------------------------------------
print("Normalise blue science spectra...")
spectra_b_norm = spec.normalise_spectra(spectra_b, True)
print("Normalise red science spectra...")
spectra_r_norm = spec.normalise_spectra(spectra_r, True)

# Make synthetic templates [requires IDL]
#ref_spec = synth.get_template_spectra(teffs, loggs, fehs)

# Load in template spectra
print("Load in synthetic templates...")
ref_params, ref_spec = synth.load_synthetic_templates(setting="R7000")  

# Normalise template spectra
print("Normalise synthetic templates...")
ref_spec_norm = spec.normalise_spectra(ref_spec)  

# -----------------------------------------------------------------------------
# Compute barycentric correction
# -----------------------------------------------------------------------------
print("Compute barycentric corrections...")

# IERS broken, hack
if False:
    from astropy.utils import iers
    from astropy.utils.iers import conf as iers_conf
    url = "https://datacenter.iers.org/data/9/finals2000A.all"
    iers_conf.iers_auto_url = url
    iers_conf.reload()

bcors = spec.compute_barycentric_correction(observations["ra"], 
                                            observations["dec"], 
                                            observations["mjd"], "SSO",
                                            disable_auto_max_age)
observations["bcor"] = bcors

# -----------------------------------------------------------------------------
# Calculate RVs and RV correct
# -----------------------------------------------------------------------------
print("Compute RVs...")
rvs, e_rvs, teffs, chi2 = spec.do_all_template_matches(spectra_r_norm, 
                                        observations, ref_params, 
                                        ref_spec_norm,)# print_diagnostics=True)

observations["teff_fit"] = teffs
observations["rv"] = rvs
observations["e_rv"] = e_rvs
observations["chi2"] = chi2

# Create a new wl scale for each arm

# Blue arm
wl_min_b = 3500
wl_max_b = 5700
n_px_b = 2858
wl_per_pixel_b = (wl_max_b - wl_min_b) / n_px_b
wl_new_b = np.arange(wl_min_b, wl_max_b, wl_per_pixel_b)

# Red arm
wl_min_r = 5400
wl_max_r = 7000
n_px_r = 3637
wl_per_pixel_r = (wl_max_r - wl_min_r) / n_px_r 
wl_new_r = np.arange(wl_min_r, wl_max_r, wl_per_pixel_r) 

# RV correct the spectra
spec_rvcor_b = spec.correct_all_rvs(spectra_b_norm, observations, wl_new_b)
spec_rvcor_r = spec.correct_all_rvs(spectra_r_norm, observations, wl_new_r)

# Save RV corrected spectra
spec.save_pkl_spectra(observations, spec_rvcor_b, spec_rvcor_r, rv_corr=True)

# -----------------------------------------------------------------------------
# Crossmatch for science program and plot
# -----------------------------------------------------------------------------
catalogue = utils.load_crossmatch_catalogue(cat_type, cat_file)

# Find Gaia IDs
utils.do_id_crossmatch(observations, catalogue)

# Plot the spectra sorted by temperature
print("Plot Teff sorted summaries....")
pplt.plot_teff_sorted_spectra(spec_rvcor_r, observations, catalogue, "r")
pplt.plot_teff_sorted_spectra(spec_rvcor_b, observations, catalogue, "b")