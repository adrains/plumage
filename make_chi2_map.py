"""
"""
import numpy as np
import pandas as pd
import plumage.synthetic as synth
import plumage.utils as utils
import plumage.plotting as pplt
import matplotlib.pyplot as plt

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

# Whether to use blue spectra in fit
use_blue_spectra = False

# Whether to include photometry in fit
include_photometry = True
colour_bands = np.array(['Bp-Rp', 'Rp-J', 'J-H', 'H-K'])
e_colour_bands = np.array(['e_Bp-Rp', 'e_Rp-J', 'e_J-H', 'e_H-K'])

use_bprp_colour = False

if not use_bprp_colour:
    colour_bands = colour_bands[1:]
    e_colour_bands = e_colour_bands[1:]

# Scale factor for synthetic colour residuals
scale_fac = 100

# Literature information (including photometry)
info_cat_path = "data/{}_info.tsv".format(label)
info_cat = utils.load_info_cat(info_cat_path, only_observed=True)

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

#source_id = "3902785109124370432"   # Gl 471
#source_id = "1244644727396803584"   # Gl 525
#source_id = "3796072592206250624"   # Gl 447
#source_id = "3101920046552857728"   # Gl 250 B (CPM)
#source_id = "3195919322830293760"   # Gl 166 C (CPM)

teff_span = 500
feh_span = 1.2
n_fits = 1000

if label == "std":
    teff_col = "teff_m15"
    feh_col = "feh_m15"
    id_col = "ID"
    skip = True

else:
    teff_col = "teff_fit_rv"
    feh_col = "feh_fit_rv"
    id_col = "TOI"
    skip = False

# Masking
mask_blue = True
mask_missing_opacities = True
mask_tio = True
mask_sodium_wings = True
low_cutoff = 6400
high_cutoff = 6600

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
# initialise IDL
idl = synth.idl_init()

# Do table join
try:
    observations.rename(columns={"uid":"source_id"}, inplace=True)
except:
    print("failed")
obs_join = observations.join(info_cat, "source_id", rsuffix="_info")
obs_join.set_index("source_id", inplace=True) 

total_stars = np.sum(~np.isnan(obs_join[teff_col].values))
counter = 1

results_dict = {}

# Run for all stars
for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
    # Only run on stars with Mann+15 parameters
    if skip and np.isnan(star_info["teff_m15"]):
        continue
    
    # Otherwise print the star ID and proceed
    print("\n{}/{} - Generating chi^2 map for {}".format(
        counter, total_stars, star_info[id_col]))
    counter += 1

    # Make bad pixel mask
    # Setup temperature dependent wavelength masks for regions where the 
    # synthetic spectra are bad (e.g. missing opacities) at cool teffs
    bad_synth_px_mask_r = synth.make_synth_mask_for_bad_wl_regions(
        spectra_r[star_i, 0], 
        star_info["rv"], 
        star_info["bcor"],
        star_info[teff_col],
        mask_blue=mask_blue,
        mask_missing_opacities=mask_missing_opacities,
        mask_tio=mask_tio,
        mask_sodium_wings=mask_sodium_wings,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,)

    bad_px_mask_r = np.logical_or(bad_px_masks_r[star_i], bad_synth_px_mask_r)

    # Initialise colours
    if include_photometry:
        # Make a mask for the colours to use, since some might be nans
        cmask = np.isfinite(star_info[colour_bands].values.astype(float))
        colours = star_info[colour_bands].values.astype(float)[cmask]
        e_colours = star_info[e_colour_bands].values.astype(float)[cmask]

    else:
        cmask = np.array([False, False, False])
        colours = None
        e_colours = None

    # Make map
    teffs, fehs, rchi2s = synth.make_chi2_map(
        star_info[teff_col],
        star_info["logg_m19"],
        star_info[feh_col],
        spectra_r[star_i, 0], # Red wl
        spectra_r[star_i, 1], # Red spec
        spectra_r[star_i, 2], # Red uncertainties
        bad_px_mask_r,
        star_info["rv"], 
        star_info["bcor"],
        idl,
        band_settings_r,
        band_settings_b=band_settings_b,
        wave_b=None, #wave_b, 
        spec_b=None, #spec_b, 
        e_spec_b=None, #e_spec_b, 
        bad_px_mask_b=None, #bad_px_mask_b,
        stellar_colours=colours,
        e_stellar_colours=e_colours,
        colour_bands=colour_bands[cmask],  # Mask the colours to what we have
        teff_span=teff_span,
        feh_span=feh_span, 
        n_fits=n_fits,
        scale_fac=scale_fac,)

    # Save
    results_dict[source_id] = [teffs, fehs, rchi2s]

    # Get uncertainties
    if label == "std":
        e_teff = star_info["e_{}".format(teff_col)]
        e_feh = star_info["e_{}".format(feh_col)]
    else:
        e_teff = 0
        e_feh = 0

    # Plot the chi^2 map
    pplt.plot_chi2_map(
        star_info[teff_col],
        e_teff,
        star_info[feh_col],
        e_feh,
        teffs,
        fehs,
        rchi2s,
        levels=10,
        feh_slice_step=0.04,
        do_log10_rchi2s=False,
        use_n_log_levels=True,
        star_id=star_info[id_col],
        source_id=source_id,
        save_path="plots/chi2_maps_{}_phot_x{:0.0f}".format(label, scale_fac),
        used_phot=include_photometry,
        phot_scale=scale_fac,)