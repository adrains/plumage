"""Function to generate synthetic spectra at the literature values for stellar
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

std_info = utils.load_info_cat("data/std_info.tsv")
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)

# Initlise empty arrays to hold synthetic spectra at lit params
all_synth_lit_b = []
all_synth_lit_r = []

# Intitialise wavelength scale values
res_b = 3000
wl_min_b = 3500
wl_max_b = 5700
n_px_b = 2858
wl_per_px_b = (wl_max_b - wl_min_b) / n_px_b

res_r = 7000
wl_min_r = 5400
wl_max_r = 7000
n_px_r = 3637
wl_per_px_r = (wl_max_r - wl_min_r) / n_px_r

# Initialise IDL
idl = synth.idl_init()

# For every spectrum in observations, we want to get a spectrum at the 
# lit values *if* the star has each of Teff, logg, and [Fe/H] in std_info
for star_i, obs_info in observations.iterrows():
    # Get the gaia id
    uid = obs_info["uid"]

    # Find this ID in std_info
    star_info = std_info[std_info["source_id"]==uid]

    # Can't continue if the star isn't in star_info, or it has an incomplete
    # set of parameters
    if (len(star_info) == 0 
        or not np.isfinite(
            np.sum(star_info.iloc[0][["teff", "logg", "feh"]].values))):
        # Before continuing, first save nan arrays
        all_synth_lit_b.append(np.full(n_px_b, np.nan))
        all_synth_lit_r.append(np.full(n_px_r, np.nan))

        continue

    # All good to go, just save variables for convenience
    star_info = star_info.iloc[0]
    params = star_info[["teff", "logg", "feh"]]
    
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
        params[0], 
        params[1], 
        params[2], 
        wl_min_b, 
        wl_max_b, 
        res_b, 
        norm="abs",
        do_resample=True, 
        wl_per_pixel=wl_per_px_b,
        )

    # The grid we put our new synthetic spectrum on should be put in the same
    # RV frame as the science spectrum
    wave_rv_scale_b = 1 - (rv - bcor)/(const.c.si.value/1000)
    ref_spec_interp_b = ius(wave_synth_b, spec_synth_b)

    wave_synth_b = wave_sci_b * wave_rv_scale_b
    spec_synth_b = ref_spec_interp_b(wave_synth_b)

    # Normalise spectra by wavelength region
    spec_synth_b = spec.norm_spec_by_wl_region(wave_sci_b, spec_synth_b, "b")

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Red
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # We have all three stellar params, now get a synthetic spectrum
    wave_synth_r, spec_synth_r = synth.get_idl_spectrum(
        idl, 
        params[0], 
        params[1], 
        params[2], 
        wl_min_r, 
        wl_max_r, 
        res_r, 
        norm="abs",
        do_resample=True, 
        wl_per_pixel=wl_per_px_r,
        )

    # The grid we put our new synthetic spectrum on should be put in the same
    # RV frame as the science spectrum
    wave_rv_scale_r = 1 - (rv - bcor)/(const.c.si.value/1000)
    ref_spec_interp_r = ius(wave_synth_r, spec_synth_r)

    wave_synth_r = wave_sci_r * wave_rv_scale_r
    spec_synth_r = ref_spec_interp_r(wave_synth_r)

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