"""Script to generate synthetic spectra at the literature values for stellar
standards.

TODO: now requires the first half of train_stannon.py to be run. Generalise.
"""
import numpy as np
import plumage.utils as pu
import plumage.synthetic as synth
import plumage.spectra as spec

# -----------------------------------------------------------------------------
# Setup & Settings
# -----------------------------------------------------------------------------
# Whether to shift the resulting synthetic spectrum to the stellar frame
do_rv_shift_to_stellar_frame = False

if do_rv_shift_to_stellar_frame:
    hdu_name = "synth_lit"
else:
    hdu_name = "rest_frame_synth_lit"

# Importing observations
label = "cannon"
spec_path = "spectra"

spectra_b, spectra_r, observations = pu.load_fits(label, path=spec_path)

# Initlise empty arrays to hold synthetic spectra at lit params
all_synth_lit_b = []
all_synth_lit_r = []

# Intitialise wavelength scale values
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

# Load results table
obs_join = pu.load_fits_table("CANNON_INFO", "cannon")

# Initialise IDL
idl = synth.idl_init()

# -----------------------------------------------------------------------------
# Generating Synthetic Spectra
# -----------------------------------------------------------------------------
# For every spectrum in observations, we want to get a spectrum at the 
# lit values *if* the star has each of Teff, logg, and [Fe/H].
for star_i, (sid, obs_info) in enumerate(obs_join.iterrows()):
    if not obs_info["has_complete_label_set"]:
        all_synth_lit_b.append(np.full(band_settings_b["n_px"], np.nan))
        all_synth_lit_r.append(np.full(band_settings_r["n_px"], np.nan))
        continue

    # Grab the teff, logg, and [Fe/H] for this star
    params = obs_info[[
        "label_adopt_teff", "label_adopt_logg", "label_adopt_feh"]
        ].values.astype(float)

    # Don't proceed if any of these parameters are NaNs
    if np.sum(np.isnan(params)) > 0:
        all_synth_lit_b.append(np.full(band_settings_b["n_px"], np.nan))
        all_synth_lit_r.append(np.full(band_settings_r["n_px"], np.nan))
        continue
    
    # Get RV and bcor for shifting synth spec
    if do_rv_shift_to_stellar_frame:
        rv = obs_info["rv"]
        bcor = obs_info["bcor"]
    else:
        rv = 0
        bcor = 0

    # Get science wavelength scales
    wave_sci_b = spectra_b[star_i, 0]
    wave_sci_r = spectra_r[star_i, 0]

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
        rv_bcor=(rv-bcor),
        )

    # Normalise spectra by wavelength region
    spec_synth_r = spec.norm_spec_by_wl_region(wave_sci_r, spec_synth_r, "r")

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Wrapping up
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Now should have both blue and red synthetic spectra at the lit values
    all_synth_lit_b.append(spec_synth_b)
    all_synth_lit_r.append(spec_synth_r)

# All done, save these
all_synth_lit_b = np.stack(all_synth_lit_b)
all_synth_lit_r = np.stack(all_synth_lit_r)

# Save to the fits file
pu.save_fits_image_hdu(
    data=all_synth_lit_b,
    extension=hdu_name,
    label=label,
    path=spec_path,
    arm="b")

pu.save_fits_image_hdu(
    data=all_synth_lit_r,
    extension=hdu_name,
    label=label,
    path=spec_path,
    arm="r")

# Also merge into a single arm and save
wl_br, spec_br, e_spec_br = spec.merge_wifes_arms_all(
    wl_b=wave_synth_b,
    spec_b=all_synth_lit_b,
    e_spec_b=np.ones_like(all_synth_lit_b),
    wl_r=wave_synth_r,
    spec_r=all_synth_lit_r,
    e_spec_r=np.ones_like(all_synth_lit_r))

# Save this single spectrum
pu.save_fits_image_hdu(spec_br, hdu_name, label, arm="br")