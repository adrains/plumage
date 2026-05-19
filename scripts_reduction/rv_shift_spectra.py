"""Script to RV shift spectra to the rest frame. This is necessary to use
spectra with the Cannon.
"""
import plumage.utils as pu
import plumage.spectra as ps
import stannon.stannon as stannon

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
spec_path = "spectra"
label = "cannon_mk"

# Set to True if we've previously used the 'modern' ID crossmatch in
# scripts_reduction/import_spectra.py and our index column is 'source_id_dr3'
do_use_dr3_id = True

# Minimumum and maximum wavelengths to be used when normalising (applicable
# later when running the Cannon, mostly because the coolest stars don't have
# good SNR < 400 nm.
wl_min = 4000
wl_max = 7000

# Normalisation - using using a Gaussian smoothed version of the spectrum. Only
# wavelengths > than wl_min_normalisation will be considered during either 
# approach to avoid low-SNR blue pixels for the coolest stars.
wl_min_normalisation = 4000
wl_broadening = 50

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
# Gaussian normalisation
#------------------------------------------------------------------------------
fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wl_br,
        spectra=spec_br,
        e_spectra=e_spec_br,
        wl_min_model=wl_min,
        wl_max_model=wl_max,
        wl_min_normalisation=wl_min_normalisation,
        wl_broadening=wl_broadening,
        do_gaussian_spectra_normalisation=True,)

# Save both sets as extra fits HDUs
pu.save_fits_image_hdu(fluxes_norm, "rest_frame_spec_norm", label, arm="br")
pu.save_fits_image_hdu(ivars_norm, "rest_frame_ivars_norm", label, arm="br")