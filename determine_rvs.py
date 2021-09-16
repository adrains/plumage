"""Script to perform RV fits to R7000 WiFeS spectra
"""
import os
from tqdm import tqdm
import plumage.synthetic as synth
import plumage.spectra as spec
import plumage.utils as utils
import plumage.plotting as pplt

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Unique label of the fits file of spectra
label = "cannon"

# Where to load from and save to
spec_path = "spectra"
fits_save_path = "spectra"

# The grid of reference spectra to use
ref_label_b = "53_teff_only_B3000_grid"
ref_label_r = "51_teff_only_R7000_rv_grid"

# Load science observations
observations = utils.load_fits_table("OBS_TAB", label, path=spec_path)

# Load in science spectra
wave_b = utils.load_fits_image_hdu("wave", label, arm="b")
spec_b = utils.load_fits_image_hdu("spec", label, arm="b")
e_spec_b = utils.load_fits_image_hdu("sigma", label, arm="b")

wave_r = utils.load_fits_image_hdu("wave", label, arm="r")
spec_r = utils.load_fits_image_hdu("spec", label, arm="r")
e_spec_r = utils.load_fits_image_hdu("sigma", label, arm="r")

# Whether to do plotting
make_rv_diagnostic_plots = False
plot_teff_sorted_spectra = False

# -----------------------------------------------------------------------------
# Normalise science and template spectra
# -----------------------------------------------------------------------------
# Note that the spectra are masked for telluric and emission regions whilst
# computing polynomial fit to do normalisation
print("Normalise science spectra...")
spec_b_norm, e_spec_b_norm = spec.normalise_spectra(wave_b, spec_b, e_spec_b)
spec_r_norm, e_spec_r_norm = spec.normalise_spectra(wave_r, spec_r, e_spec_r)

# Load in template spectra
print("Load in synthetic templates...")
# Blue
ref_wave_b, ref_flux_b, ref_params_b = synth.load_synthetic_templates(ref_label_b)
ref_spec_norm_b = spec.normalise_spectra(ref_wave_b, ref_flux_b)

# Red
ref_wave_r, ref_flux_r, ref_params_r = synth.load_synthetic_templates(ref_label_r)
ref_spec_norm_r = spec.normalise_spectra(ref_wave_r, ref_flux_r,)

# -----------------------------------------------------------------------------
# Calculate RVs
# -----------------------------------------------------------------------------
print("Fitting RVs to blue and red spectra...")
# Blue
nres_b, rchi2_b, bad_px_masks_b, info_dicts_b = spec.do_all_template_matches(
    wave_b,
    spec_b_norm,
    e_spec_b_norm,
    observations,
    ref_params_b,
    ref_wave_b,
    ref_spec_norm_b,
    "b",
    save_column_ext="b")

# Red
nres_r, rchi2_r, bad_px_masks_r, info_dicts_r = spec.do_all_template_matches(
    wave_r,
    spec_r_norm,
    e_spec_r_norm,
    observations, 
    ref_params_r,
    ref_wave_r,
    ref_spec_norm_r,
    "r")

# Save, but now with RV
utils.save_fits_table("OBS_TAB", observations, label, path=spec_path)
utils.save_fits_image_hdu(bad_px_masks_b, "bad_px", label, arm="b")
utils.save_fits_image_hdu(bad_px_masks_r, "bad_px", label, arm="r")