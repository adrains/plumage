"""Script to generate synthetic spectra at the literature values for stellar
standards.
"""
import numpy as np
import plumage.utils as utils
import plumage.synthetic as synth
import plumage.spectra as spec
from astropy import constants as const
from scipy.interpolate import InterpolatedUnivariateSpline as ius

label = "std"
spec_path = "spectra"

std_info = utils.load_info_cat("data/std_info.tsv", in_paper=True)
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)

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

# Intialise columns to use for two sets of reference stars
mann15_cols = ["teff_m15", "logg_m19", "feh_m15"]
ra12_cols = ["teff_ra12", "logg_m19", "feh_ra12"]
int_cols = ["teff_int", "logg_m19"]
other_cols = ["teff_other", "logg_other", "feh_other"]

# Initialise IDL
idl = synth.idl_init()

# For every spectrum in observations, we want to get a spectrum at the 
# lit values *if* the star has each of Teff, logg, and [Fe/H] in std_info
for star_i, (sid, obs_info) in enumerate(observations.iterrows()):
    # Find this ID in std_info
    if sid not in std_info.index:
        all_synth_lit_b.append(np.full(band_settings_b["n_px"], np.nan))
        all_synth_lit_r.append(np.full(band_settings_r["n_px"], np.nan))
        continue
    else:
        star_info = std_info.loc[sid]

    # Check Mann+15
    if np.isfinite(np.sum(star_info[mann15_cols].values)):
        params = star_info[mann15_cols]
    
    # Rojas-Ayala+12
    elif np.isfinite(np.sum(star_info[ra12_cols].values)):
        params = star_info[ra12_cols]
    
    # Interferometry
    elif np.isfinite(np.sum(star_info[int_cols].values)):
        params = np.concatenate((star_info[int_cols], [0]))   # Assume [Fe/H]=0

    # Other
    elif np.isfinite(np.sum(star_info[other_cols].values)):
        params = star_info[other_cols]

    # Any other condition, we don't have a complete set of params so continue
    else:
        all_synth_lit_b.append(np.full(band_settings_b["n_px"], np.nan))
        all_synth_lit_r.append(np.full(band_settings_r["n_px"], np.nan))
        continue
    
    # Get RV and bcor for shifting synth spec
    rv = obs_info["rv"]
    bcor = obs_info["bcor"]

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
        rv_bcor=(rv-bcor),
        )

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
utils.save_fits_image_hdu(all_synth_lit_b, "synth_lit", label, path=spec_path, arm="b")
utils.save_fits_image_hdu(all_synth_lit_r, "synth_lit", label, path=spec_path, arm="r")