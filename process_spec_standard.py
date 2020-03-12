"""Script to process 1D extracted/reduced science spectra, normalise, calculate
RVs for, and shift to rest frame. Note that whilst the normalisation code will
accept non-fluxed spectra, it has not been vetted and results cannot currently
be guaranteed. All spectra should be in the same format (e.g. fluxed), and
checked ahead of time for "bad" extractions without spectra present.

Data structure:
 - Science spectra stored in plumage/*/*.fits
 - Science spectra pickle files stored in plumage/*.pkl
 - Template spectra stored in plumage/templates/*.csv
 - Catalogue of observed stars stored in data/all_2m3_star_ids.csv
 - Catalogue of measured activity stored in data/activity.fits

 The specific template and catalogue files to load can be specified below.

 Template catalogues now have three files, where * is the unique label 
 associated with the tempalte grid:
  1 - template_params*.csv
  2 - template_spectra*.csv
  3 - template_wave*.csv

In late 2019 issues were encountered accessing online files related to the
International Earth Rotation and Reference Systems Service. This is required to
calculate barycentric corrections for the data. This astopy issue may prove a
usefule resource again if the issue reoccurs:
 - https://github.com/astropy/astropy/issues/8981
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import plumage.synthetic as synth
import plumage.spectra as spec
import plumage.plotting as pplt
import plumage.utils as utils
from astropy.table import Table
from astropy.io import fits

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_spectra = True                    # Whether to load in pickle spectra
label = "std"                      # If loading, which pickle of N spectra
disable_auto_max_age = False            # Useful if IERS broken

flux_corrected = True                   # If *all* data is fluxed
telluric_corrected = True               # If *all* data is telluric corr

spec_folder = "spectra/standard"

cat_type = "csv"                        # Crossmatch catalogue type
cat_file = "data/all_2m3_star_ids.csv"  # Crossmatch catalogue 

do_standard_crossmatch = True           # Whether to crossmatch lit standards
do_activity_crossmatch = True           # Whether to crossmatch activty cat
activity_file = "data/activity.fits"    # Catalogue of measured activity

ref_label = "576_tess_R7000_grid"       # The grid of reference spectra to use
#ref_label = "51_teff_only_R7000_rv_grid"

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
    print("Loading science spectra...")
    spectra_b = utils.load_spectra_fits("b", label)
    spectra_r = utils.load_spectra_fits("r", label)
    observations = utils.load_observations_fits(label)

else:
    # Do initial import
    print("Doing inital spectra import...")
    observations, spectra_b, spectra_r = spec.load_all_spectra(
        spec_folder, ext_sci=ext_sci)
    
    # Clean spectra
    spec.clean_spectra(spectra_b)
    spec.clean_spectra(spectra_r)

    # Save
    utils.save_observations_fits(observations, label)
    utils.save_spectra_fits(spectra_b, "b", label)
    utils.save_spectra_fits(spectra_r, "r", label)

# -----------------------------------------------------------------------------
# Normalise science and template spectra
# -----------------------------------------------------------------------------
# Note that the spectra are masked for telluric and emission regions before
# being normalised
print("Normalise blue science spectra...")
spectra_b_norm = spec.normalise_spectra(spectra_b, True)
print("Normalise red science spectra...")
spectra_r_norm = spec.normalise_spectra(spectra_r, True)

# Load in template spectra
print("Load in synthetic templates...")
ref_wave, ref_flux, ref_params = synth.load_synthetic_templates(ref_label)
ref_spec = spec.reformat_spectra(ref_wave, ref_flux)

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
# Calculate RVs and shift spectra to the rest frame
# -----------------------------------------------------------------------------
print("Compute RVs...")
all_nres, grid_rchi2 = spec.do_all_template_matches(
    spectra_r_norm, 
    observations, 
    ref_params, 
    ref_spec_norm,)

# Create a new wl scale for each arm
wl_new_b = spec.make_wl_scale(3500, 5700, 2858)
wl_new_r = spec.make_wl_scale(5400, 7000, 3637)

# RV correct the spectra
spec_rvcor_b = spec.correct_all_rvs(spectra_b_norm, observations, wl_new_b)
spec_rvcor_r = spec.correct_all_rvs(spectra_r_norm, observations, wl_new_r)

rv_label = label + "_rv_corr"

# Save rest frame normalised spectra
utils.save_spectra_fits(spec_rvcor_b, "b", rv_label)
utils.save_spectra_fits(spec_rvcor_r, "r", rv_label)

# -----------------------------------------------------------------------------
# Crossmatch for science program, activity, and chi^2 standard fit params 
# -----------------------------------------------------------------------------
# Crossmatch with observed catalogue for Gaia IDs
catalogue = utils.load_crossmatch_catalogue(cat_type, cat_file)
utils.do_id_crossmatch(observations, catalogue)

standards = utils.load_standards()  
std_params_all = utils.consolidate_standards(standards, force_unique=True)  

# Save the observation data
utils.save_observations_fits(observations, label)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
# Plot the spectra sorted by temperature
print("Plot Teff sorted summaries....")
pplt.plot_teff_sorted_spectra(spectra_b, observations, catalogue, "b",
                              suffix=label, normalise=True)
pplt.plot_teff_sorted_spectra(spectra_r, observations, catalogue, "r",
                              suffix=label, normalise=True)