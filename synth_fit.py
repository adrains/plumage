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
import plumage.parameters as params
import matplotlib.pyplot as plt
from collections import OrderedDict

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

# Whether to use blue spectra in fit
use_blue_spectra = False

# Whether to include photometry in fit
include_photometry = True

filter_defs = np.array([
    # [filt, use, error, mag_col, e_mag_col],
    ["Bp", True, 1.04, "Bp_mag_offset", "e_Bp_mag"],
    ["Rp", True, 1.02, "Rp_mag", "e_Rp_mag"],
    ["J", True, 1.0, "J_mag", "e_J_mag"],
    ["H", True, 1.0, "H_mag", "e_H_mag"],
    ["K", True, 1.0, "K_mag", "e_K_mag"],
    ["v", False, 2.0, "v_psf", "e_v_psf"],
    ["g", False, 1.2, "g_psf", "e_g_psf"],
    ["r", True, 1.02 , "r_psf", "e_r_psf"],
    ["i", True, 1.02, "i_psf", "e_i_psf"],
    ["z", True, 1.02, "z_psf", "e_z_psf"],
], dtype=object)

# Mask out unused filters
filter_defs = filter_defs[filter_defs[:,1].astype(bool)]

# Construct arrays from columns of filters we are using
filters = filter_defs[:,0].astype(str)
band_model_uncertainties = 2.5*np.log10(filter_defs[:,2].astype(float))
phot_band_cols = filter_defs[:,3].astype(str)
e_phot_band_cols = filter_defs[:,4].astype(str)

# Scale factor for synthetic colour residuals
phot_scale_fac = 1

# Literature information (including photometry)
info_cat_path = "data/{}_info.tsv".format(label)
info_cat = utils.load_info_cat(info_cat_path, in_paper=True, only_observed=True)

skymapper_phot = pd.read_csv( 
    "data/rains_all_gaia_ids_matchfinal.csv", 
    sep=",", 
    dtype={"source_id":str}, 
    header=0)
skymapper_phot.set_index("source_id", inplace=True)  

info_cat = info_cat.join(skymapper_phot, "source_id", how="left", rsuffix="_") 

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
    ("teff",True),
    ("logg",False),
    ("feh",False),
    ("Mbol",True),])

#fit_for_params = [False, False, True]   #[teff, logg, feh]

n_params = np.sum(list(fit_for_params.values()))

teff_init_col = "teff_fit_rv"
logg_init_col = "logg_m19"
feh_init_col = "feh_fit_rv"

# Masking of bad model wavelength regions
cutoff_temp = 8000
mask_blue = True
mask_missing_opacities = True
mask_tio = False
mask_sodium_wings = False
low_cutoff = None
high_cutoff = None

def calc_bp_offset(bp_rp,):
    """
    """
    bp_offset = np.zeros_like(bp_rp)

    bp_offset[bp_rp > 3] = 2.5*np.log10(3**-1 * bp_rp[bp_rp > 3] + 0.05)

    mask = np.logical_and(bp_rp > 2, bp_rp <= 3)

    bp_offset[mask] = 2.5*np.log10(0.05 * bp_rp[mask] + 0.9)

    return bp_offset

# Offset Bp to synthetic equivalent
info_cat["Bp_mag_offset"] = info_cat["Bp_mag"] - calc_bp_offset(info_cat["Bp-Rp"])

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
mbol_synth = np.full(len(observations), np.nan)
e_mbol_synth = np.full(len(observations), np.nan)
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

    # Match the star with its literature info, continue if we don't have any
    source_id = observations.iloc[ob_i].name

    if source_id in info_cat.index:
        star_info = info_cat.loc[source_id]
    else:
        fit_results.append(None)
        continue
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialise colours
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Now get the colours to be included in the fit if:
    #  A) We're including photometry in the fit and
    #  B) We actually have photometry
    if include_photometry:
        # Make a mask for the colours to use, since some might be nans
        phot_mask = np.isfinite(star_info[phot_band_cols].values.astype(float))
        photometry = star_info[phot_band_cols].values.astype(float)[phot_mask]
        e_photometry = star_info[e_phot_band_cols].values.astype(float)[phot_mask]
        e_phot_model = band_model_uncertainties[phot_mask]
        fit_used_colours[ob_i] = True

        # Add model uncertainty in quadrature
        e_photometry = np.sqrt(e_photometry**2 + e_phot_model**2)

        # Now write flags indicating which colours were used
        flags = str(phot_mask.astype(int))[1:-1].replace(" ", "")
        colours_used[ob_i] = flags
    else:
        phot_mask = np.zeros(len(filters)).astype(bool)
        photometry = None
        e_photometry = None

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialise params
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params_init = {
        "teff":observations.iloc[ob_i][teff_init_col],
        "logg":star_info[logg_init_col],
        "feh":observations.iloc[ob_i][feh_init_col],
        "Mbol":0,
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
        observations.iloc[ob_i][teff_init_col],
        cutoff_temp=cutoff_temp,
        mask_blue=mask_blue,
        mask_missing_opacities=mask_missing_opacities,
        mask_tio=mask_tio,
        mask_sodium_wings=mask_sodium_wings,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,)

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
            observations.iloc[ob_i][teff_init_col],
            cutoff_temp=cutoff_temp,
            mask_blue=mask_blue,
            mask_missing_opacities=mask_missing_opacities,
            mask_tio=mask_tio,
            mask_sodium_wings=mask_sodium_wings,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,)

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
        stellar_phot=photometry,
        e_stellar_phot=e_photometry,
        phot_bands=filters[phot_mask],   # Mask the colours to what we have
        phot_scale_fac=phot_scale_fac,
        fit_for_resid_norm_fac=True,)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sort out results
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    teff_synth[ob_i] = opt_res["teff"]
    e_teff_synth[ob_i] = opt_res["e_teff"] # TODO

    logg_synth[ob_i] = opt_res["logg"]
    e_logg_synth[ob_i] = opt_res["e_logg"] # TODO

    feh_synth[ob_i] = opt_res["feh"]
    e_feh_synth[ob_i] = opt_res["e_feh"] # TODO

    mbol_synth[ob_i] = opt_res["Mbol"]
    e_mbol_synth[ob_i] = opt_res["e_Mbol"] # TODO

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
                                                   opt_res["e_feh"]),
          "m_bol = {:0.2f} +/- {:0.2f}\n".format(opt_res["Mbol"], 
                                                   opt_res["e_Mbol"]),)

    # Print observed stellar colours
    synth_phot = opt_res["synth_phot"]
    synth_bc = opt_res["synth_bc"]

    if synth_phot is not None:
        for filt, psynth in zip(filters[phot_mask], synth_phot):
            print("{} = {:0.3f}, ".format(filt, psynth), end="")
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
observations["mbol_synth"] = mbol_synth
observations["e_mbol_synth"] = e_mbol_synth

observations["rchi2_synth"] = rchi2
observations["both_arm_synth_fit"] = both_arm_synth_fit
observations["fit_used_colours"] = fit_used_colours
observations["colours_used"] = colours_used
#observations["colour_resid_scale_factor"] = scale_fac

# Now calculate final params
fbol, e_fbol = params.calc_f_bol_from_mbol(
    mbol_synth, 
    e_mbol_synth)

rad, e_rad = params.calc_radii(
    teff_synth,
    e_teff_synth,
    fbol,
    e_fbol, 
    obs_join["dist"],
    obs_join["e_dist"],)

L_star, e_L_star = params.calc_L_star(
    fbol,
    e_fbol,obs_join["dist"],
    obs_join["e_dist"])

observations["fbol"] = fbol
observations["e_fbol"] = e_fbol
observations["radius"] = rad
observations["e_radius"] = e_rad
observations["L_star"] = L_star
observations["e_L_star"] = e_L_star

# Determine limb darkening coefficients
ldc_ak = params.get_claret17_limb_darkening_coeff(
    observations["teff_synth"], 
    observations["logg_synth"], 
    observations["feh_synth"])

ldd_cols = ["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]

for ldc_i, ldd_col in enumerate(ldd_cols):
    observations[ldd_col] = ldc_ak[:,ldc_i]

utils.save_fits_table("OBS_TAB", observations, label, path=spec_path)

# Save best fit synthetic spectra
synth_fits_b = np.array(synth_fits_b)
synth_fits_r = np.array(synth_fits_r)
utils.save_fits_image_hdu(synth_fits_b, "synth", label, path=spec_path, arm="b")
utils.save_fits_image_hdu(synth_fits_r, "synth", label, path=spec_path, arm="r")