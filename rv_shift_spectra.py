import numpy as np
import plumage.utils as utils
import plumage.spectra as spec

spec_path = "spectra"
label = "tess"

# Import
spec_b, spec_r, obs = utils.load_fits(label, path=spec_path)

# Put in rest frame
wave_rf_b = spec_b[0,0,:]
wave_rf_r = spec_r[0,0,:]

spec_rf_b = spec.correct_all_rvs(spec_b, obs, wave_rf_b)
spec_rf_r = spec.correct_all_rvs(spec_r, obs, wave_rf_r)

# Normalise
spec_rf_norm_b = spec.normalise_spectra(spec_rf_b, True)
spec_rf_norm_r = spec.normalise_spectra(spec_rf_r, True)

# Save both sets as extra fits HDUs
utils.save_fits_image_hdu(wave_rf_b, "rest_frame_wave", label, arm="b")
utils.save_fits_image_hdu(spec_rf_b[:,1], "rest_frame_spec", label, arm="b")
utils.save_fits_image_hdu(spec_rf_b[:,2], "rest_frame_sigma", label, arm="b")
utils.save_fits_image_hdu(spec_rf_norm_b[:,1], "rest_frame_spec_norm", label, arm="b")
utils.save_fits_image_hdu(spec_rf_norm_b[:,2], "rest_frame_sigma_norm", label, arm="b")

utils.save_fits_image_hdu(wave_rf_r, "rest_frame_wave", label, arm="r")
utils.save_fits_image_hdu(spec_rf_r[:,1], "rest_frame_spec", label, arm="r")
utils.save_fits_image_hdu(spec_rf_r[:,2], "rest_frame_sigma", label, arm="r")
utils.save_fits_image_hdu(spec_rf_norm_r[:,1], "rest_frame_spec_norm", label, arm="r")
utils.save_fits_image_hdu(spec_rf_norm_r[:,2], "rest_frame_sigma_norm", label, arm="r")