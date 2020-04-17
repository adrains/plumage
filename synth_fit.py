from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import plumage.synthetic as synth
import plumage.utils as utils
import plumage.plotting as pplt
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Unique label of the fits file of spectra
label = "std"

# Where to load from and save to
spec_path = "spectra"
save_folder = "fits/std"

# Load science spectra and bad pixel masks
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)
bad_px_masks_b = utils.load_fits_image_hdu("bad_px", label, arm="b")
bad_px_masks_r = utils.load_fits_image_hdu("bad_px", label, arm="r")

# SNR metrics
snr_ratio = 3
snr_b_cutoff = 10

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
params_fit = []
fit_results = []
synth_fits_b = []
synth_fits_r = []
spec_dicts = []
rchi2 = []
both_arm_synth_fit = []

# For every star, do synthetic fitting
for ob_i in range(0, len(observations)):
    print("-"*40, "\n{}\n".format(ob_i), "-"*40)

    # Initialise parameters based on best fitting RV template
    params_init = (
        observations.iloc[ob_i]["teff_fit"],
        observations.iloc[ob_i]["logg_fit"],
        observations.iloc[ob_i]["feh_fit"],
        )

    # Check if we're going to fit with both red and blue specta. At low SNR, 
    # the measurement of blue SNR isn't reliable, so an approximate metric 
    # (based on comparison to the standard star set cooler than 4500 K) is that
    # blue SNR is ~3x less than red. We'll use this as an estimate of blue, and
    # consider SNR~25 a cut off for the blue spectra being included
    if observations.iloc[ob_i]["snr_r"]/snr_ratio > snr_b_cutoff:
        wave_b = spectra_b[ob_i, 0]
        spec_b = spectra_b[ob_i, 1]
        e_spec_b = spectra_b[ob_i, 2]
        bad_px_mask_b = bad_px_masks_b[ob_i]

    else:
        wave_b = None
        spec_b = None
        e_spec_b = None
        bad_px_mask_b = None

    # Do synthetic fit
    opt_res, spec_dict = synth.do_synthetic_fit(
        spectra_r[ob_i, 0], # Red wl
        spectra_r[ob_i, 1], # Red spec
        spectra_r[ob_i, 2], # Red uncertainties
        bad_px_masks_r[ob_i],
        params_init, 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"],
        wave_b=wave_b, 
        spec_b=spec_b, 
        e_spec_b=e_spec_b, 
        bad_px_mask_b=bad_px_mask_b,
        )

    # Record results
    params_fit.append(opt_res["x"])
    fit_results.append(opt_res)
    synth_fits_r.append(spec_dict["spec_synth_r"])
    spec_dicts.append(spec_dict)
    rchi2.append(np.sum(opt_res["fun"]**2) / (len(opt_res["fun"])-len(params_init)))

    # Append blue synthetic spectrum if used, otherwise an array of nans. Also
    # save a boolean indicating whether both arms were used in the fit
    if not spec_dict["spec_synth_b"] is None:
        synth_fits_b.append(spec_dict["spec_synth_b"])
        both_arm_synth_fit.append(True)
    else:
        synth_fits_b.append(np.ones_like(spectra_b[ob_i, 0])*np.nan)
        both_arm_synth_fit.append(False)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
# Update obs table with fits
params_fit = np.stack(params_fit)

observations["teff_synth"] = params_fit[:,0]
observations["logg_synth"] = params_fit[:,1]
observations["feh_synth"] = params_fit[:,2]
observations["rchi2_synth"] = np.array(rchi2)
observations["both_arm_synth_fit"] = np.array(both_arm_synth_fit)

utils.update_fits_obs_table(observations, label, path=spec_path)

# Save best fit synthetic spectra
synth_fits_b = np.array(synth_fits_b)
synth_fits_r = np.array(synth_fits_r)
utils.save_fits_image_hdu(synth_fits_b, "synth", label, path=spec_path, arm="b")
utils.save_fits_image_hdu(synth_fits_r, "synth", label, path=spec_path, arm="r")

# -----------------------------------------------------------------------------
# Diagnostic plotting
# -----------------------------------------------------------------------------
# Import reference catalogue
"""
info_cat = utils.load_info_cat(
    os.path.join("data", "{}_info.tsv".format(label)))

# Now do final plotting
pplt.plot_all_synthetic_fits(
    spectra_r, 
    synth_fits_r, 
    observations, 
    bad_px_masks, 
    label, 
    info_cat)
"""
pass