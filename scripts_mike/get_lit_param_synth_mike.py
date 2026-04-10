"""Script to generate synthetic spectra at the literature values for stellar
benchmarks observed with MIKE.
"""
import numpy as np
import plumage.utils as pu
import plumage.synthetic as synth
import plumage.spectra as spec
import stannon.utils as su

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

# -----------------------------------------------------------------------------
# Setup & Settings
# -----------------------------------------------------------------------------
# Whether to shift the resulting synthetic spectrum to the stellar frame
do_rv_shift_to_stellar_frame = True

if not do_rv_shift_to_stellar_frame:
    hdu_name = "synth_lit"
else:
    hdu_name = "rest_frame_synth_lit"

# Importing observations
label = "cannon"
spec_path = "spectra"

wave_r = pu.load_fits_image_hdu(
    extension="rest_frame_wave",
    label=ms.fits_label,
    fn_base=ms.fits_fn_base,
    path=ms.spec_synth_fits_folder,
    arm="r",)
    
spectra_r = pu.load_fits_image_hdu(
    extension="rest_frame_spec_norm",
    label=ms.fits_label,
    fn_base=ms.fits_fn_base,
    path=ms.spec_synth_fits_folder,
    arm="r",)

(n_star, n_px) = spectra_r.shape

# Initlise empty arrays to hold synthetic spectra at lit params
#all_synth_lit_b = []
all_synth_lit_r = np.full((n_star, ms.spec_synth_n_px), np.nan)

# Intitialise wavelength scale values
band_settings_r = {
    "inst_res_pow":ms.spec_synth_mike_resolving_power_r,
    "wl_min":ms.spec_synth_wl_min,
    "wl_max":ms.spec_synth_wl_max,
    "n_px":ms.spec_synth_n_px,
    "wl_per_px":ms.spec_synth_wl_per_px,
    "wl_broadening":ms.spec_synth_wl_broadening,
    "arm":"r",
    "grid":ms.spec_synth_grid,}

# Load results table
obs_join = pu.load_fits_table(
    extension="CANNON_INFO",
    label=ms.fits_label,
    fn_base=ms.fits_fn_base,
    path=ms.spec_synth_fits_folder,)

# Initialise IDL
idl = synth.idl_init()

# -----------------------------------------------------------------------------
# Generating Synthetic Spectra
# -----------------------------------------------------------------------------
# For every spectrum in observations, we want to get a spectrum at the 
# lit values *if* the star has each of Teff, logg, and [Fe/H].
for star_i, (sid, obs_info) in enumerate(obs_join.iterrows()):
    if not obs_info["has_complete_label_set"]:
        #all_synth_lit_b.append(np.full(band_settings_b["n_px"], np.nan))
        #all_synth_lit_r.append(np.full(band_settings_r["n_px"], np.nan))
        print("{}/{}: skipping".format(star_i+1, n_star,))
        continue

    # Grab the teff, logg, and [Fe/H] for this star
    params = obs_info[[
        "label_adopt_teff", "label_adopt_logg", "label_adopt_Fe_H"]
        ].values.astype(float)

    # Don't proceed if any of these parameters are NaNs
    if np.sum(np.isnan(params)) > 0:
        #all_synth_lit_b.append(np.full(band_settings_b["n_px"], np.nan))
        #all_synth_lit_r.append(np.full(band_settings_r["n_px"], np.nan))
        print("{}/{}: skipping".format(star_i+1, n_star,))
        continue
    
    print("{}/{}: {} {}".format(star_i+1, n_star, sid, params))

    # Get RV and bcor for shifting synth spec
    if not do_rv_shift_to_stellar_frame:
        rv = obs_info["rv_r"]
        bcor = obs_info["bcor"]
    else:
        rv = 0
        bcor = 0

    # Get science wavelength scales
    #wave_sci_b = spectra_b[star_i, 0]
    #wave_sci_r = spectra_r[star_i, 0]
    """
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Blue
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # We have all three stellar params, now get a synthetic spectrum
    wave_synth_b, spec_synth_b = synth.get_idl_spectrum(
        idl, 
        params[0],  # teff
        params[1],  # logg
        params[2],  # [Fe/H]
        band_settings_b["wl_min"], 
        band_settings_b["wl_max"], 
        ipres=band_settings_b["inst_res_pow"],
        grid=band_settings_b["grid"],
        resolution=band_settings_b["wl_per_px"],
        norm="abs",
        do_resample=True, 
        wl_per_px=band_settings_b["wl_broadening"],
        rv_bcor=(rv-bcor),)

    # Normalise spectra by wavelength region
    spec_synth_b = spec.norm_spec_by_wl_region(wave_sci_b, spec_synth_b, "b")
    """
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Red
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # We have all three stellar params, now get a synthetic spectrum
    wave_synth_r, spec_synth_r = synth.get_idl_spectrum(
        idl, 
        params[0],  # teff
        params[1],  # logg
        params[2],  # [Fe/H]
        band_settings_r["wl_min"], 
        band_settings_r["wl_max"], 
        ipres=band_settings_r["inst_res_pow"],
        grid=band_settings_r["grid"],
        resolution=band_settings_r["wl_per_px"],
        norm="abs",
        do_resample=True, 
        wl_per_px=band_settings_r["wl_broadening"],
        rv_bcor=(rv-bcor),)

    # HACK: off by one pixel error
    wave_synth_r = wave_synth_r[:-1]
    spec_synth_r = spec_synth_r[:-1]

    # Normalise spectra by wavelength region
    spec_synth_r = spec.norm_spec_by_wl_region(wave_r, spec_synth_r, "r")

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Wrapping up
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Now should have both blue and red synthetic spectra at the lit values
    #all_synth_lit_b.append(spec_synth_b)
    all_synth_lit_r[star_i] = spec_synth_r

# All done, save these
#all_synth_lit_b = np.stack(all_synth_lit_b)
#all_synth_lit_r = np.stack(all_synth_lit_r)

# Save to the fits file
#pu.save_fits_image_hdu(
#    data=all_synth_lit_b,
#    extension=hdu_name,
#    label=label,
#    path=spec_path,
#    arm="b")

pu.save_fits_image_hdu(
    data=all_synth_lit_r,
    extension=hdu_name,
    fn_base="mike_spectra",
    label=ms.spec_synth_fits_label,
    path=ms.spec_synth_fits_folder,
    arm="r")

# Also merge into a single arm and save
#wl_br, spec_br, e_spec_br = spec.merge_wifes_arms_all(
#    wl_b=wave_synth_b,
#    spec_b=all_synth_lit_b,
#    e_spec_b=np.ones_like(all_synth_lit_b),
#    wl_r=wave_synth_r,
#    spec_r=all_synth_lit_r,
#    e_spec_r=np.ones_like(all_synth_lit_r))

# Save this single spectrum
#pu.save_fits_image_hdu(spec_br, hdu_name, label, arm="br")