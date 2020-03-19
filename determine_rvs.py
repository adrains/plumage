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
label = "std"                      # If loading, which pickle of N spectra
spec_path = "spectra"
fits_save_path = "spectra"

#ref_label = "576_tess_R7000_grid"       # The grid of reference spectra to use
#ref_label = "795_full_R7000_rv_grid"
ref_label = "51_teff_only_R7000_rv_grid"

# Load science spectra
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)

# Whether to do diagnostic plotting
do_plotting = True

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
all_nres, grid_rchi2, info_dicts = spec.do_all_template_matches(
    spectra_r_norm, 
    observations, 
    ref_params, 
    ref_spec_norm,)

# Save, but now with RV
utils.save_fits(spectra_b, spectra_r, observations, label, path=fits_save_path)

# Make diagnostic plots
if do_plotting:
    # Make new directory
    plot_dir = "plots/rv_fits_%s" % label
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    for star_i in tqdm(range(len(observations)), desc="Plotting Diagnostics"):
        pplt.plot_synthetic_fit(
          spectra_r_norm[star_i, 0], 
          spectra_r_norm[star_i, 1], 
          spectra_r_norm[star_i, 2], 
          info_dicts[star_i]["template_spec"], 
          (observations.iloc[star_i]["teff_fit"], 
          observations.iloc[star_i]["logg_fit"],
          observations.iloc[star_i]["feh_fit"],),
          "%i_%s" % (star_i, observations.iloc[star_i]["id"]), 
          os.path.join(plot_dir, "%03i.pdf" % star_i),
          info_dicts[star_i]["bad_px_mask"],
          observations.iloc[star_i]["rv"],
          observations.iloc[star_i]["e_rv"])

    # Merge plots
    pplt.merge_spectra_pdfs(
        os.path.join(plot_dir,"???.pdf"), 
        os.path.join(plot_dir, "rv_diagnostics_%s.pdf" % label))