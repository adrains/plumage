"""Script to perform synthetic fits to observed spectra.

Uncertainty formalism from:
 1 - https://stackoverflow.com/questions/42388139/how-to-compute-standard-
     deviation-errors-with-scipy-optimize-least-squares
After correcting for mistakes and scaling by variance (per the method
employed in the RV fitting)
"""
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
label = "tess"

# Where to load from and save to
spec_path = "spectra"

# Load science spectra and bad pixel masks
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)
bad_px_masks_b = utils.load_fits_image_hdu("bad_px", label, arm="b")
bad_px_masks_r = utils.load_fits_image_hdu("bad_px", label, arm="r")

# SNR metrics
snr_ratio = 3
snr_b_cutoff = 10

# Whether to include photometry in fit
include_photometry = True
colour_bands = ['Rp-J', 'J-H', 'H-K']
e_colour_bands = ['e_Rp-J', 'e_J-H', 'e_H-K']

# Literature information (including photometry)
info_cat_path = "data/{}_info.tsv".format(label)
info_cat = utils.load_info_cat(info_cat_path, only_observed=True) 
only_fit_info_cat_stars = True

# Initialise settings for each band
band_settings_b = {
    "inst_res_pow":3000,
    "wl_min":3500,
    "wl_max":5700,
    "n_px":2858,
    "wl_per_px":0.77,
    "wl_broadening":0.77,
    "arm":"b",
    "grid":"B3000",
}
band_settings_r = {
    "inst_res_pow":7000,
    "wl_min":5400,
    "wl_max":7000,
    "n_px":3637,
    "wl_per_px":0.44,
    "wl_broadening":0.44,
    "arm":"r",
    "grid":"R7000",
}

# Whether to fix logg during fitting
fix_logg = True

if fix_logg:
    n_params = 2
else:
    n_params = 3

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
params_fit = []
e_params_fit = []
fit_results = []
synth_fits_b = []
synth_fits_r = []
rchi2 = []
both_arm_synth_fit = []
fit_used_colours = []

# initialise IDL
idl = synth.idl_init()

# For every star, do synthetic fitting
for ob_i in range(0, len(observations)):
    ln = "-"*40
    print("{}\n{} - {}\n{}".format(ln, ob_i, observations.iloc[ob_i]["id"],ln))

    # Match the star with its literature info
    star_info = info_cat[info_cat["source_id"]==observations.iloc[ob_i]["uid"]]
    if len(star_info) == 0:
        star_info = None
    elif len(star_info) > 0:
        star_info = star_info.iloc[0]

    # Check if we're only fitting for a subset of the standards
    if only_fit_info_cat_stars and star_info is None:
        params_fit.append(np.full(n_params, np.nan))
        e_params_fit.append(np.full(n_params, np.nan))
        fit_results.append(None)
        synth_fits_b.append(np.ones_like(spectra_b[ob_i, 0])*np.nan)
        synth_fits_r.append(np.ones_like(spectra_r[ob_i, 0])*np.nan)
        rchi2.append(np.nan)
        both_arm_synth_fit.append(False)
        fit_used_colours.append(False)

        continue
    
    # Now get the colours to be included in the fit if:
    #  A) We're including photometry in the fit and
    #  B) We actually have photometry
    if include_photometry and star_info is not None:
        colours = star_info[colour_bands].values.astype(float)
        e_colours = star_info[e_colour_bands].values.astype(float)
        fit_used_colours.append(True)
    else:
        colours = None
        e_colours = None
        fit_used_colours.append(False)

    # Initialise Teff and [Fe/H] and fix logg
    if fix_logg:
        params_init = (
            observations.iloc[ob_i]["teff_fit"],
            observations.iloc[ob_i]["feh_fit"],
            )
        logg = star_info["logg_m19"]
        e_logg = star_info["e_logg_m19"]
    
    # Intialise Teff, logg, and [Fe/H], and fit for logg
    else:
        params_init = (
            observations.iloc[ob_i]["teff_fit"],
            observations.iloc[ob_i]["logg_fit"],
            observations.iloc[ob_i]["feh_fit"],
            )
        logg = None

    # Setup temperature dependent wavelength masks for regions where the 
    # synthetic spectra are bad (e.g. missing opacities) at cool teffs
    bad_synth_px_mask_r = synth.make_synth_mask_for_bad_wl_regions(
        spectra_r[ob_i, 0], 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"], 
        observations.iloc[ob_i]["teff_fit"])

    bad_px_mask_r = np.logical_or(bad_px_masks_r[ob_i], bad_synth_px_mask_r)

    # Check if we're going to fit with both red and blue specta. At low SNR, 
    # the measurement of blue SNR isn't reliable, so an approximate metric 
    # (based on comparison to the standard star set cooler than 4500 K) is that
    # blue SNR is ~3x less than red. We'll use this as an estimate of blue, and
    # consider SNR~25 a cut off for the blue spectra being included
    if observations.iloc[ob_i]["snr_r"]/snr_ratio > snr_b_cutoff:
        wave_b = spectra_b[ob_i, 0]
        spec_b = spectra_b[ob_i, 1]
        e_spec_b = spectra_b[ob_i, 2]

        bad_synth_px_mask_b = synth.make_synth_mask_for_bad_wl_regions(
            spectra_b[ob_i, 0], 
            observations.iloc[ob_i]["rv"], 
            observations.iloc[ob_i]["bcor"], 
            observations.iloc[ob_i]["teff_fit"])

        bad_px_mask_b = np.logical_or(bad_px_masks_b[ob_i], bad_synth_px_mask_b)

        both_arm_synth_fit.append(True)

    # Not fitting to blue, pass in Nones
    else:
        wave_b = None
        spec_b = None
        e_spec_b = None
        bad_px_mask_b = None

        both_arm_synth_fit.append(False)

    # Do synthetic fit
    opt_res = synth.do_synthetic_fit(
        spectra_r[ob_i, 0], # Red wl
        spectra_r[ob_i, 1], # Red spec
        spectra_r[ob_i, 2], # Red uncertainties
        bad_px_masks_r[ob_i],
        params_init, 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"],
        idl,
        band_settings_r,
        logg=logg,
        band_settings_b=band_settings_b,
        wave_b=wave_b, 
        spec_b=spec_b, 
        e_spec_b=e_spec_b, 
        bad_px_mask_b=bad_px_mask_b,
        stellar_colours=colours,
        e_stellar_colours=e_colours,
        colour_bands=colour_bands,
        )

    # Record results
    params_fit.append(opt_res["x"])
    e_params_fit.append(opt_res["std"])
    fit_results.append(opt_res)
    synth_fits_r.append(opt_res["spec_synth_r"])
    rchi2.append(np.sum(opt_res["fun"]**2) / (len(opt_res["fun"])-len(params_init)))

    # Append blue synthetic spectrum if used, otherwise an array of nans
    if opt_res["spec_synth_b"] is not None:
        synth_fits_b.append(opt_res["spec_synth_b"])
    else:
        synth_fits_b.append(np.ones_like(spectra_b[ob_i, 0])*np.nan)

    if logg is None:
        teff, logg, feh = opt_res["x"]
        e_teff, e_logg, e_feh = opt_res["std"]
    else:
        teff, feh = opt_res["x"]
        e_teff, e_feh = opt_res["std"]

    print("\n---Result---")
    print("Teff = {:0.0f} +/- {:0.0f} K,".format(teff, e_teff), 
          "logg = {:0.2f} +/- {:0.2f},".format(logg, e_logg),
          "[Fe/H] = {:+0.2f} +/- {:0.2f}\n".format(feh, e_feh),)

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
# Update obs table with fits
params_fit = np.stack(params_fit)
e_params_fit = np.stack(e_params_fit)

observations["teff_synth"] = params_fit[:,0]
observations["e_teff_synth"] = e_params_fit[:,0]

if fix_logg:
    observations["feh_synth"] = params_fit[:,1]
    observations["e_feh_synth"] = e_params_fit[:,1]
else:
    observations["logg_synth"] = params_fit[:,1]
    observations["e_logg_synth"] = e_params_fit[:,1]
    observations["feh_synth"] = params_fit[:,2]
    observations["e_feh_synth"] = e_params_fit[:,2]

observations["rchi2_synth"] = np.array(rchi2)
observations["both_arm_synth_fit"] = np.array(both_arm_synth_fit)
observations["fit_used_colours"] = np.array(fit_used_colours)

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