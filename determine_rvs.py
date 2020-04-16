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
label = "std"

# Where to load from and save to
spec_path = "spectra"
fits_save_path = "spectra"

# The grid of reference spectra to use
#ref_label = "576_tess_R7000_grid"       
#ref_label = "795_full_R7000_rv_grid"
ref_label_b = "53_teff_only_B3000_grid"
ref_label_r = "51_teff_only_R7000_rv_grid"

# Load science spectra
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)

# Whether to do plotting
make_rv_diagnostic_plots = True
plot_teff_sorted_spectra = False

# -----------------------------------------------------------------------------
# Normalise science and template spectra
# -----------------------------------------------------------------------------
# Note that the spectra are masked for telluric and emission regions whilst
# computing polynomial fit to do normalisation
print("Normalise science spectra...")
spectra_b_norm = spec.normalise_spectra(spectra_b, True)
spectra_r_norm = spec.normalise_spectra(spectra_r, True)

# Load in template spectra
print("Load in synthetic templates...")
# Blue
ref_wave_b, ref_flux_b, ref_params_b = synth.load_synthetic_templates(ref_label_b)
ref_spec_b = spec.reformat_spectra(ref_wave_b, ref_flux_b)
ref_spec_norm_b = spec.normalise_spectra(ref_spec_b)

# Red
ref_wave_r, ref_flux_r, ref_params_r = synth.load_synthetic_templates(ref_label_r)
ref_spec_r = spec.reformat_spectra(ref_wave_r, ref_flux_r)
ref_spec_norm_r = spec.normalise_spectra(ref_spec_r)

# -----------------------------------------------------------------------------
# Calculate RVs
# -----------------------------------------------------------------------------
print("Fitting RVs to blue and red spectra...")
# Blue
nres_b, rchi2_b, bad_px_masks_b, info_dicts_b = spec.do_all_template_matches(
    spectra_b_norm, 
    observations, 
    ref_params_b, 
    ref_spec_norm_b,
    save_column_ext="b")

# Red
nres_r, rchi2_r, bad_px_masks_r, info_dicts_r = spec.do_all_template_matches(
    spectra_r_norm, 
    observations, 
    ref_params_r, 
    ref_spec_norm_r,)


# Save, but now with RV
utils.save_fits(spectra_b, spectra_r, observations, label, path=fits_save_path)
utils.save_fits_image_hdu(bad_px_masks_b, "bad_px", label, arm="b")
utils.save_fits_image_hdu(bad_px_masks_r, "bad_px", label, arm="r")

# TODO: save rv corrected spectra as fits HDUs

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
# Make diagnostic plots
if make_rv_diagnostic_plots:
    # Make new directory
    plot_dir = "plots/rv_fits_%s" % label
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    for star_i in tqdm(range(len(observations)), desc="Plotting Diagnostics"):
        # Blue
        pplt.plot_synthetic_fit(
          spectra_b_norm[star_i, 0], 
          spectra_b_norm[star_i, 1], 
          spectra_b_norm[star_i, 2], 
          info_dicts_b[star_i]["template_spec"], 
          bad_px_masks_b[star_i],
          observations.iloc[star_i],
          ["teff_fit_b", "logg_fit_b", "feh_fit_b"],
          "%i_%s_b" % (star_i, observations.iloc[star_i]["id"]), 
          os.path.join(plot_dir, "%03i_b.pdf" % star_i),
          save_fig=True,
          arm="b")

        # Red
        pplt.plot_synthetic_fit(
          spectra_r_norm[star_i, 0], 
          spectra_r_norm[star_i, 1], 
          spectra_r_norm[star_i, 2], 
          info_dicts_r[star_i]["template_spec"], 
          bad_px_masks_r[star_i],
          observations.iloc[star_i],
          ["teff_fit", "logg_fit", "feh_fit"],
          "%i_%s_r" % (star_i, observations.iloc[star_i]["id"]), 
          os.path.join(plot_dir, "%03i_r.pdf" % star_i),
          save_fig=True,
          arm="r")

    # Merge plots
    pplt.merge_spectra_pdfs(
        os.path.join(plot_dir,"???_b.pdf"), 
        os.path.join(plot_dir, "rv_diagnostics_%s_b.pdf" % label))
    
    pplt.merge_spectra_pdfs(
        os.path.join(plot_dir,"???_r.pdf"), 
        os.path.join(plot_dir, "rv_diagnostics_%s_r.pdf" % label))

# Plot the spectra sorted by temperature
if plot_teff_sorted_spectra:
    print("Plot Teff sorted summaries....")
    pplt.plot_teff_sorted_spectra(spectra_b, observations, "b",
                                suffix=label, normalise=True)
    pplt.plot_teff_sorted_spectra(spectra_r, observations, "r",
                                suffix=label, normalise=True)