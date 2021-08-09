import numpy as np
import plumage.utils as utils
import plumage.spectra as spec

spec_path = "spectra"
label = "std"

# Import observation table and blue/red spectra
obs = utils.load_fits_table("OBS_TAB", label, path="spectra")

wave_b = utils.load_fits_image_hdu("wave", label, arm="b")
spec_b = utils.load_fits_image_hdu("spec", label, arm="b")
e_spec_b = utils.load_fits_image_hdu("sigma", label, arm="b")

wave_r = utils.load_fits_image_hdu("wave", label, arm="r")
spec_r = utils.load_fits_image_hdu("spec", label, arm="r")
e_spec_r = utils.load_fits_image_hdu("sigma", label, arm="r")

# Put in rest frame
wave_rf_b = wave_b
wave_rf_r = wave_r

spec_rf_b, e_spec_rf_b = spec.correct_all_rvs(
    wave_b, spec_b, e_spec_b, obs, wave_rf_b)

spec_rf_r, e_spec_rf_r = spec.correct_all_rvs(
    wave_r, spec_r, e_spec_r, obs, wave_rf_r)

# Combine arms
wl_br, spec_br, e_spec_br = spec.merge_wifes_arms_all(
    wave_rf_b,
    spec_rf_b,
    e_spec_rf_b,
    wave_rf_r,
    spec_rf_r,
    e_spec_rf_r)

# Save this single spectrum
utils.save_fits_image_hdu(wl_br, "rest_frame_wave", label, arm="br")
utils.save_fits_image_hdu(spec_br, "rest_frame_spec", label, arm="br")
utils.save_fits_image_hdu(e_spec_br, "rest_frame_sigma", label, arm="br")

# Normalise
spec_rf_norm_b, e_spec_rf_norm_b = spec.normalise_spectra(
    wave_rf_b, spec_rf_b, e_spec_rf_b)

spec_rf_norm_r, e_spec_rf_norm_r = spec.normalise_spectra(
    wave_rf_r, spec_rf_r, e_spec_rf_r)

# Save both sets as extra fits HDUs
utils.save_fits_image_hdu(wave_rf_b, "rest_frame_wave", label, arm="b")
utils.save_fits_image_hdu(spec_rf_b, "rest_frame_spec", label, arm="b")
utils.save_fits_image_hdu(e_spec_rf_b, "rest_frame_sigma", label, arm="b")
utils.save_fits_image_hdu(spec_rf_norm_b, "rest_frame_spec_norm", label, arm="b")
utils.save_fits_image_hdu(e_spec_rf_norm_b, "rest_frame_sigma_norm", label, arm="b")

utils.save_fits_image_hdu(wave_rf_r, "rest_frame_wave", label, arm="r")
utils.save_fits_image_hdu(spec_rf_r, "rest_frame_spec", label, arm="r")
utils.save_fits_image_hdu(e_spec_rf_r, "rest_frame_sigma", label, arm="r")
utils.save_fits_image_hdu(spec_rf_norm_r, "rest_frame_spec_norm", label, arm="r")
utils.save_fits_image_hdu(e_spec_rf_norm_r, "rest_frame_sigma_norm", label, arm="r")