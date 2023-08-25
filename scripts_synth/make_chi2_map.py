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

g2m_filts = np.array(['Bp', 'Rp', 'J', 'H', 'K'])
g2m_phot_band_cols = np.array(["{}_mag".format(filt) for filt in g2m_filts])
e_g2m_phot_band_cols = np.array(["e_{}_mag".format(filt) for filt in g2m_filts])

sm_filts = np.array(['v', 'g', 'r', 'i', 'z',])
sm_phot_band_cols = np.array(["{}_psf".format(filt) for filt in sm_filts])
e_sm_phot_band_cols = np.array(["e_{}_psf".format(filt) for filt in sm_filts])

filters = np.concatenate((g2m_filts, sm_filts))
phot_band_cols = np.concatenate((g2m_phot_band_cols, sm_phot_band_cols))
e_phot_band_cols = np.concatenate((e_g2m_phot_band_cols, e_sm_phot_band_cols))

# Model uncertainties
band_model_uncertainties = 2.5*np.log10(
    [1.15, 1.05, 0, 0, 0, 2, 1.2, 1.05, 1.05, 1.05])
band_model_uncertainties[~np.isfinite(band_model_uncertainties)] = 0

# Bp, Rp, J, H, K, v, g, r, i, z
phot_mask = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 1]).astype(bool)
filters = filters[phot_mask]
phot_band_cols = phot_band_cols[phot_mask]
e_phot_band_cols = e_phot_band_cols[phot_mask]
band_model_uncertainties = band_model_uncertainties[phot_mask]

# Scale factor for synthetic colour residuals
phot_scale_fac = 1

# Literature information (including photometry)
info_cat_path = "data/{}_info.tsv".format(label)
info_cat = utils.load_info_cat(info_cat_path, only_observed=True)

skymapper_phot = pd.read_csv( 
    "data/rains_all_gaia_ids_matchfinal.csv", 
    sep=",", 
    dtype={"source_id":str}, 
    header=0)
skymapper_phot.set_index("source_id", inplace=True)  

info_cat = info_cat.join(skymapper_phot, "source_id", how="inner", rsuffix="_") 

# Mask for testing
#mask = np.isfinite(info_cat["i_psf"])
#info_cat = info_cat[mask]

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

teff_span = 600
feh_span = 1.2
n_fits = 3

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
mask_tio = False
mask_sodium_wings = False

# Ca
low_cutoff = 6350
high_cutoff = 6500

# TiO
low_cutoff = 6700
high_cutoff = 6850

# All
low_cutoff = None
high_cutoff = None

map_desc = "mbol_ars_median_norm_5_pc"

scale_residuals = {"blue":True,"red":True,"phot":True}

drive = "priv"

cutoff_temp = 6000

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
# initialise IDL
idl = synth.idl_init(drive=drive)

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
        print("Skipping, no literature info available")
        continue

    #if source_id == "2603090003484152064":
    #    continue
    
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
        cutoff_temp=cutoff_temp,
        mask_blue=mask_blue,
        mask_missing_opacities=mask_missing_opacities,
        mask_tio=mask_tio,
        mask_sodium_wings=mask_sodium_wings,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,)

    bad_px_mask_r = np.logical_or(bad_px_masks_r[star_i], bad_synth_px_mask_r)

    # Setup Blue
    if use_blue_spectra:
        wave_b = spectra_b[star_i, 0]
        spec_b = spectra_b[star_i, 1]
        e_spec_b = spectra_b[star_i, 2]

        bad_synth_px_mask_b = synth.make_synth_mask_for_bad_wl_regions(
            spectra_b[star_i, 0], 
            star_info["rv"], 
            star_info["bcor"],
            star_info[teff_col],
            mask_blue=mask_blue,
            mask_missing_opacities=mask_missing_opacities,
            mask_tio=mask_tio,
            mask_sodium_wings=mask_sodium_wings,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,)

        bad_px_mask_b = np.logical_or(bad_px_masks_b[star_i], bad_synth_px_mask_b)

    else:
        wave_b = None
        spec_b = None
        e_spec_b = None
        bad_px_mask_b = None

    # Initialise colours
    if include_photometry:
        # Make a mask for the colours to use, since some might be nans
        phot_mask = np.isfinite(star_info[phot_band_cols].values.astype(float))
        photometry = star_info[phot_band_cols].values.astype(float)[phot_mask]
        e_photometry = star_info[e_phot_band_cols].values.astype(float)[phot_mask]
        e_phot_model = band_model_uncertainties[phot_mask]

        # Add model uncertainty in quadrature
        e_photometry = np.sqrt(e_photometry**2 + e_phot_model**2)

    else:
        phot_mask = np.array([False, False, False])
        photometry = None
        e_photometry = None

    # Make map
    chi2_map_dict = synth.make_chi2_map(
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
        wave_b=spectra_b[star_i, 0], #wave_b, 
        spec_b=spec_b, 
        e_spec_b=e_spec_b, 
        bad_px_mask_b=bad_px_mask_b,
        stellar_phot=photometry,
        e_stellar_phot=e_photometry,
        phot_bands=filters[phot_mask],  # Mask photometry to what we have
        teff_span=teff_span,
        feh_span=feh_span, 
        n_fits=n_fits,
        phot_scale_fac=phot_scale_fac,
        scale_residuals=scale_residuals,)

    # Save
    results_dict[source_id] = chi2_map_dict

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
        chi2_map_dict,
        filters[phot_mask],
        n_levels=20,
        star_id=star_info[id_col],
        source_id=source_id,
        save_path="plots/chi2_maps_{}_{}_phot_x{:0.0f}".format(
            label, map_desc, phot_scale_fac),
        used_phot=include_photometry,
        phot_scale_fac=phot_scale_fac,)

    #import pdb
    #pdb.set_trace()