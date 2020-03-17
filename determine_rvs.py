"""Script to perform RV fits to R7000 WiFeS spectra
"""
import plumage.synthetic as synth
import plumage.spectra as spec
import plumage.utils as utils

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
label = "tess"                      # If loading, which pickle of N spectra
spec_path = "spectra"

ref_label = "576_tess_R7000_grid"       # The grid of reference spectra to use
#ref_label = "51_teff_only_R7000_rv_grid"

# Load science spectra
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)

# -----------------------------------------------------------------------------
# Normalise science and template spectra
# -----------------------------------------------------------------------------
# Note that the spectra are masked for telluric and emission regions whilst
# computing polynomial fit to do normalisation
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
# Calculate RVs
# -----------------------------------------------------------------------------
print("Compute RVs...")
all_nres, grid_rchi2 = spec.do_all_template_matches(
    spectra_r_norm, 
    observations, 
    ref_params, 
    ref_spec_norm,)

# Save, but now with RV
utils.save_fits(spectra_b, spectra_r, observations, label, path=fits_save_path)