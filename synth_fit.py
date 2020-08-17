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
from collections import OrderedDict                         

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Unique label of the fits file of spectra
label = "std"

# Where to load from and save to
spec_path = "spectra"

# Load science spectra and bad pixel masks
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)
bad_px_masks_b = utils.load_fits_image_hdu("bad_px", label, arm="b")
bad_px_masks_r = utils.load_fits_image_hdu("bad_px", label, arm="r")

# SNR metrics
snr_ratio = 3
snr_b_cutoff = 10

# Whether to use blue spectra in fit
use_blue_spectra = False

# Whether to include photometry in fit
include_photometry = False
colour_bands = np.array(['Rp-J', 'J-H', 'H-K'])
e_colour_bands = np.array(['e_Rp-J', 'e_J-H', 'e_H-K'])

# Scale factor for synthetic colour residuals
scale_fac = 1

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

# Whether to fix logg or teff during fitting
fit_for_params = OrderedDict([
    ("teff",False),
    ("logg",False),
    ("feh",True),])

#fit_for_params = [False, False, True]   #[teff, logg, feh]

n_params = np.sum(list(fit_for_params.values()))

teff_init_col = "teff_fit_rv"
logg_init_col = "logg_m19"
feh_init_col = "feh_fit_rv"

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
# Initialise results cols
teff_synth = np.full(len(observations), np.nan)
e_teff_synth = np.full(len(observations), np.nan)
logg_synth = np.full(len(observations), np.nan)
e_logg_synth = np.full(len(observations), np.nan)
feh_synth = np.full(len(observations), np.nan)
e_feh_synth = np.full(len(observations), np.nan)
fit_results = []
synth_fits_b = np.full(spectra_b[:,0,:].shape, np.nan)  
synth_fits_r = np.full(spectra_r[:,0,:].shape, np.nan)  
rchi2 = np.full(len(observations), np.nan)
both_arm_synth_fit = np.full(len(observations), False)
fit_used_colours = np.full(len(observations), False)
colours_used = np.full(len(observations), "000")

# initialise IDL
idl = synth.idl_init()

# Do table join
try:
    observations.rename(columns={"uid":"source_id"}, inplace=True)
except:
    print("failed")
obs_join = observations.join(info_cat, "source_id", rsuffix="_info")

# For every star, do synthetic fitting
for ob_i in range(0, len(observations)):
    ln = "-"*40
    print("{}\n{} - {}\n{}".format(ln, ob_i, observations.iloc[ob_i]["id"],ln))

    # Match the star with its literature info
    star_info = info_cat[info_cat.index==observations.iloc[ob_i]["source_id"]]
    if len(star_info) == 0:
        star_info = None
    elif len(star_info) > 0:
        star_info = star_info.iloc[0]

    # Check if we're only fitting for a subset of the standards
    if (only_fit_info_cat_stars and star_info is None) or np.isnan(star_info["teff_m15"]):
        fit_results.append(None)
        continue
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialise colours
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Now get the colours to be included in the fit if:
    #  A) We're including photometry in the fit and
    #  B) We actually have photometry
    if include_photometry and star_info is not None:
        # Make a mask for the colours to use, since some might be nans
        cmask = np.isfinite(star_info[colour_bands].values.astype(float))
        colours = star_info[colour_bands].values.astype(float)[cmask]
        e_colours = star_info[e_colour_bands].values.astype(float)[cmask]
        fit_used_colours[ob_i] = True

        # Now write flags indicating which colours were used
        flags = str(cmask.astype(int))[1:-1].replace(" ", "")
        colours_used[ob_i] = flags
    else:
        cmask = np.array([False, False, False])
        colours = None
        e_colours = None

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialise params
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params_init = {
        "teff":observations.iloc[ob_i][teff_init_col],
        "logg":star_info[logg_init_col],
        "feh":observations.iloc[ob_i][feh_init_col],
    }

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Wavelength masks
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setup temperature dependent wavelength masks for regions where the 
    # synthetic spectra are bad (e.g. missing opacities) at cool teffs
    bad_synth_px_mask_r = synth.make_synth_mask_for_bad_wl_regions(
        spectra_r[ob_i, 0], 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"], 
        observations.iloc[ob_i][teff_init_col])

    bad_px_mask_r = np.logical_or(bad_px_masks_r[ob_i], bad_synth_px_mask_r)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setup blue/red spectra
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check if we're going to fit with both red and blue specta. At low SNR, 
    # the measurement of blue SNR isn't reliable, so an approximate metric 
    # (based on comparison to the standard star set cooler than 4500 K) is that
    # blue SNR is ~3x less than red. We'll use this as an estimate of blue, and
    # consider SNR~25 a cut off for the blue spectra being included
    if (use_blue_spectra 
        and observations.iloc[ob_i]["snr_r"]/snr_ratio > snr_b_cutoff):
        wave_b = spectra_b[ob_i, 0]
        spec_b = spectra_b[ob_i, 1]
        e_spec_b = spectra_b[ob_i, 2]

        bad_synth_px_mask_b = synth.make_synth_mask_for_bad_wl_regions(
            spectra_b[ob_i, 0], 
            observations.iloc[ob_i]["rv"], 
            observations.iloc[ob_i]["bcor"], 
            observations.iloc[ob_i][teff_init_col])

        bad_px_mask_b = np.logical_or(bad_px_masks_b[ob_i], bad_synth_px_mask_b)

        both_arm_synth_fit[ob_i] = True

    # Not fitting to blue, pass in Nones for everything except the wavelength
    # scale. This is so we can generate a synthetic spectrum for the blue still
    else:
        wave_b = spectra_b[ob_i, 0]
        spec_b = None
        e_spec_b = None
        bad_px_mask_b = None

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Do synthetic fit
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        fit_for_params=fit_for_params,
        band_settings_b=band_settings_b,
        wave_b=wave_b, 
        spec_b=spec_b, 
        e_spec_b=e_spec_b, 
        bad_px_mask_b=bad_px_mask_b,
        stellar_colours=colours,
        e_stellar_colours=e_colours,
        colour_bands=colour_bands[cmask],   # Mask the colours to what we have
        scale_fac=scale_fac,)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sort out results
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    teff_synth[ob_i] = opt_res["teff"]
    e_teff_synth[ob_i] = opt_res["e_teff"] # TODO

    logg_synth[ob_i] = opt_res["logg"]
    e_logg_synth[ob_i] = opt_res["e_logg"] # TODO

    feh_synth[ob_i] = opt_res["feh"]
    e_feh_synth[ob_i] = opt_res["e_feh"] # TODO

    fit_results.append(opt_res)
    synth_fits_r[ob_i] = opt_res["spec_synth_r"]
    rchi2[ob_i] = opt_res["rchi2"]

    # Update blue synthetic spectrum if used
    if opt_res["spec_synth_b"] is not None:
        synth_fits_b[ob_i] = opt_res["spec_synth_b"]

    print("\n---Result---")
    print("Teff = {:0.0f} +/- {:0.0f} K,".format(opt_res["teff"], 
                                                 opt_res["e_teff"]), 
          "logg = {:0.2f} +/- {:0.2f},".format(opt_res["logg"], 
                                               opt_res["e_logg"]), 
          "[Fe/H] = {:+0.2f} +/- {:0.2f}\n".format(opt_res["feh"], 
                                                   opt_res["e_feh"]),)

    # Print observed stellar colours
    synth_colours = opt_res["synth_colours"]

    if synth_colours is not None:
        for cband, csynth in zip(colour_bands[cmask], synth_colours):
            print("{} = {:0.3f}, ".format(cband, csynth), end="")
        print("\n")

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
# Update obs table with fits
observations["teff_synth"] = teff_synth
observations["e_teff_synth"] = e_teff_synth
observations["logg_synth"] = logg_synth
observations["e_logg_synth"] = e_logg_synth
observations["feh_synth"] = feh_synth
observations["e_feh_synth"] = e_feh_synth

observations["rchi2_synth"] = rchi2
observations["both_arm_synth_fit"] = both_arm_synth_fit
observations["fit_used_colours"] = fit_used_colours
observations["colours_used"] = colours_used
observations["colour_resid_scale_factor"] = scale_fac

utils.save_fits_table("OBS_TAB", observations, label, path=spec_path)

# Save best fit synthetic spectra
synth_fits_b = np.array(synth_fits_b)
synth_fits_r = np.array(synth_fits_r)
utils.save_fits_image_hdu(synth_fits_b, "synth", label, path=spec_path, arm="b")
utils.save_fits_image_hdu(synth_fits_r, "synth", label, path=spec_path, arm="r")