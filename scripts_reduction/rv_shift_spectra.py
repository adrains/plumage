"""Script to RV shift spectra to the rest frame. This is necessary to use
spectra with the Cannon.
"""
import plumage.utils as pu
import plumage.spectra as ps

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
spec_path = "spectra"
label = "planet_mk"

# Set to True if we've previously used the 'modern' ID crossmatch in
# scripts_reduction/import_spectra.py and our index column is 'source_id_dr3'
do_use_dr3_id = True

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Import observation table and blue/red spectra
obs = pu.load_fits_table(
    "OBS_TAB", label, path="spectra", do_use_dr3_id=do_use_dr3_id)

wave_b = pu.load_fits_image_hdu("wave", label, arm="b")
spec_b = pu.load_fits_image_hdu("spec", label, arm="b")
e_spec_b = pu.load_fits_image_hdu("sigma", label, arm="b")

wave_r = pu.load_fits_image_hdu("wave", label, arm="r")
spec_r = pu.load_fits_image_hdu("spec", label, arm="r")
e_spec_r = pu.load_fits_image_hdu("sigma", label, arm="r")

#------------------------------------------------------------------------------
# RV shifting
#------------------------------------------------------------------------------
# Adopt 'final' wavelength scales
wave_rf_b = wave_b
wave_rf_r = wave_r

# RV shift blue arm to the rest frame
spec_rf_b, e_spec_rf_b = ps.correct_all_rvs(
    wave_b, spec_b, e_spec_b, obs, wave_rf_b)

# RV shift red arm to the rest frame
spec_rf_r, e_spec_rf_r = ps.correct_all_rvs(
    wave_r, spec_r, e_spec_r, obs, wave_rf_r)

#------------------------------------------------------------------------------
# Merging arms
#------------------------------------------------------------------------------
# Combine blue and red arms
wl_br, spec_br, e_spec_br = ps.merge_wifes_arms_all(
    wave_rf_b,
    spec_rf_b,
    e_spec_rf_b,
    wave_rf_r,
    spec_rf_r,
    e_spec_rf_r)

# Save this merged spectrum
pu.save_fits_image_hdu(wl_br, "rest_frame_wave", label, arm="br")
pu.save_fits_image_hdu(spec_br, "rest_frame_spec", label, arm="br")
pu.save_fits_image_hdu(e_spec_br, "rest_frame_sigma", label, arm="br")

#------------------------------------------------------------------------------
# Polynomial normalisation + saving each arm
#------------------------------------------------------------------------------
# Polynomial normalisation
spec_rf_norm_b, e_spec_rf_norm_b = ps.normalise_spectra(
    wave_rf_b, spec_rf_b, e_spec_rf_b)

spec_rf_norm_r, e_spec_rf_norm_r = ps.normalise_spectra(
    wave_rf_r, spec_rf_r, e_spec_rf_r)

# Save both sets as extra fits HDUs
pu.save_fits_image_hdu(wave_rf_b, "rest_frame_wave", label, arm="b")
pu.save_fits_image_hdu(spec_rf_b, "rest_frame_spec", label, arm="b")
pu.save_fits_image_hdu(e_spec_rf_b, "rest_frame_sigma", label, arm="b")
pu.save_fits_image_hdu(spec_rf_norm_b, "rest_frame_spec_norm", label, arm="b")
pu.save_fits_image_hdu(
    e_spec_rf_norm_b, "rest_frame_sigma_norm", label, arm="b")

pu.save_fits_image_hdu(wave_rf_r, "rest_frame_wave", label, arm="r")
pu.save_fits_image_hdu(spec_rf_r, "rest_frame_spec", label, arm="r")
pu.save_fits_image_hdu(e_spec_rf_r, "rest_frame_sigma", label, arm="r")
pu.save_fits_image_hdu(spec_rf_norm_r, "rest_frame_spec_norm", label, arm="r")
pu.save_fits_image_hdu(
    e_spec_rf_norm_r, "rest_frame_sigma_norm", label, arm="r")