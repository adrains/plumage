"""
Regions:

Blue:
 - 3700-3900
 - 5000-5200

Red:
 - 5600-5800
 - 6600-6800
"""
import numpy as np
import plumage.utils as utils
import plumage.synthetic as synth
import astropy.constants as const
from scipy.optimize import least_squares
from scipy.interpolate import InterpolatedUnivariateSpline as ius

def compute_resid(
    res, 
    params,
    wave_sci, 
    spec_sci, 
    e_spec_sci, 
    bad_px_mask,
    rv, 
    bcor, 
    idl,
    wl_min,
    wl_max,
    wl_per_px,
    ):
    """
    """

    wave_synth, spec_synth = synth.get_idl_spectrum(
        idl, 
        params[0], 
        params[1], 
        params[2], 
        wl_min, 
        wl_max, 
        res[0], 
        norm="abs",
        do_resample=True, 
        wl_per_pixel=wl_per_px,
        )

    # The grid we put our new synthetic spectrum on should be put in the same
    # RV frame as the science spectrum
    wave_rv_scale = 1 - (rv - bcor)/(const.c.si.value/1000)
    ref_spec_interp = ius(wave_synth, spec_synth)

    wave_synth = wave_sci * wave_rv_scale
    spec_synth = ref_spec_interp(wave_synth)

    # Normalise spectra by median before fit
    e_spec_sci /= np.nanmedian(spec_sci)
    spec_sci /= np.nanmedian(spec_sci)
    spec_synth /= np.nanmedian(spec_synth)

    # Calculate the residual
    resid_vect = (spec_sci[~bad_px_mask] 
                  - spec_synth[~bad_px_mask]) / e_spec_sci[~bad_px_mask]

    if not np.isfinite(np.sum(resid_vect)):
        resid_vect = np.ones_like(resid_vect) * 1E30

    return resid_vect


def fit_for_res(
    idl,
    params,
    wave, 
    spec_sci, 
    e_spec_sci, 
    bad_px_mask,
    rv, 
    bcor,
    wl_min,
    wl_max,
    wl_per_px,
    res,):
    """
    """
    # Setup fit settings
    args = (params, wave, spec_sci, e_spec_sci, bad_px_mask, rv, bcor , idl,
            wl_min, wl_max, wl_per_px)
    bounds = ((1000,), (100000,))
    scale = (1,)
    step = (1,)
    
    # Make sure initial teff guess isn't out of bounds, assign default if so
    #res = res
    res = np.array([res])

    # Do fit
    optimize_result = least_squares(
        compute_resid, 
        res, 
        jac="3-point",
        bounds=bounds,
        x_scale=scale,
        diff_step=step,
        args=args, 
    )

    return optimize_result


def fit_for_many_res(
    idl, 
    params_all, 
    spectra,
    bad_px_masks, 
    observations, 
    wl_mins,
    wl_maxs,
    star_is,
    res,):
    """
    """
    res_fits_all = []

    for star_i, params in zip(star_is, params_all):

        print("\n{}".format(observations.iloc[star_i]["id"]))
        res_fits = []

        for wl_min, wl_max in zip(wl_mins, wl_maxs):

            wl_mask = np.logical_and(spectra[star_i,0] > wl_min, 
                                    spectra[star_i,0] < wl_max)

            wl_per_px = (wl_max - wl_min) / len(spectra[star_i, 0][wl_mask])

            opt_res = fit_for_res(
                idl,
                params,
                spectra[star_i, 0][wl_mask], 
                spectra[star_i, 1][wl_mask], 
                spectra[star_i, 2][wl_mask], 
                bad_px_masks[star_i][wl_mask],
                observations.iloc[star_i]["rv"], 
                observations.iloc[star_i]["bcor"],
                wl_min,
                wl_max,
                wl_per_px,
                res,)

            # Calculate uncertainties
            jac = opt_res["jac"]
            residuals = opt_res["fun"]
            cov = np.linalg.inv(jac.T.dot(jac))
            std = np.sqrt(np.diagonal(cov)) * np.nanvar(residuals)

            print("{:0.0f}-{:0.0f} A --> R~{:0.0f}+/-{:0.0f}".format(
                wl_min, wl_max, float(opt_res["x"]), float(std)))

            res_fits.append([float(opt_res["x"]), float(std)])

        res_fits_all.append(res_fits)

    rf = np.array(res_fits_all)

    mn = rf[:,:,0]
    std = rf[:,:,1]

    return mn, std

# Import
label = "std"
spec_path = "spectra"

spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path) 
bad_px_masks_b = utils.load_fits_image_hdu("bad_px", label, arm="b") 
bad_px_masks_r = utils.load_fits_image_hdu("bad_px", label, arm="r") 

row_is = [
    12,     # eps Eri, 5164707970261630080
    39,     # eps Ind, 6412595290592307840
    7,      # omi2 Eri, 3195919528988725120
    41,     # del Pav, 6427464325637241728
    3,      # tau Cet, 2452378776434276992
    157,    # HD 131977, 6232511606838403968
]

# Initialise IDL
idl = synth.idl_init()

params_all = [
    [5052, 4.62, -0.13],
    [4649, 4.61, -0.06],
    [5126, 4.55, -0.19],
    [5571, 4.28, 0.33],
    [5347, 4.53, -0.55],
    [4505, 4.76, 0.12]]

print("-"*40, 'Fitting Blue')

# Do blue fit
wl_step = 400
wl_mins_b = np.arange(3600, 5400, wl_step)
wl_maxs_b = wl_mins_b + wl_step

blue_res, e_blue_res = fit_for_many_res(
    idl, 
    params_all, 
    spectra_b,
    bad_px_masks_b, 
    observations, 
    wl_mins_b,
    wl_maxs_b,
    row_is,
    3000,)

w_res_b = np.average(blue_res, axis=0, weights=e_blue_res**-2)
w_e_res_b = (np.sum(e_blue_res**-2,axis=0)**-1)**0.5 

print("-"*40, 'Fitting Red')

# Do red fit
wl_step = 400
wl_mins_r = np.arange(5400, 6850, wl_step)
wl_maxs_r = wl_mins_r + wl_step

red_res, e_red_res = fit_for_many_res(
    idl, 
    params_all, 
    spectra_r,
    bad_px_masks_r, 
    observations, 
    wl_mins_r,
    wl_maxs_r,
    row_is,
    7000,)

w_res_r = np.average(red_res, axis=0, weights=e_red_res**-2)
w_e_res_r = (np.sum(e_red_res**-2,axis=0)**-1)**0.5 