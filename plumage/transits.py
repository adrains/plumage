"""Module to interface with TESS light curves and fit
"""
import os
import numpy as np
import lightkurve as lk
import batman as bm
from tqdm import tqdm
from scipy.optimize import least_squares

# Ensure the lightcurves folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "lightcurves"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)


def my_custom_corrector_func(lc):
    corrected_lc = lc.normalize().flatten(window_length=401)
    return corrected_lc

def download_target_px_file(tic_id):
    """
    """
    sr = lk.search_tesscut("TIC {}".format(tic_id))
    lcfs = sr.download_all()
    tpf = lcfs[0]

    aperture_mask = tpf.create_threshold_mask(threshold=7)
    lc = tpf.to_lightcurve(aperture_mask=aperture_mask)     

    return lc

def download_lc_all_sectors(tic_id, source_id, save_fits=True, 
                            save_path="lightcurves"):
    """
    https://docs.lightkurve.org/tutorials/01-lightcurve-files.html

    SAP: Simple Aperture Photometry
    PDCSAP: Pre-search Data Conditioning SAP
    """
    # Search for light curve results for our star
    search_res = lk.search_lightcurvefile("TIC {}".format(tic_id), 
                                          mission="TESS")

    # If we didn't get as match with the TIC ID, try the source ID
    if len(search_res) == 0:
        search_res = lk.search_lightcurvefile("Gaia DR2 {}".format(source_id), 
                                              mission="TESS")

    # Raise exception if we still don't have a match
    if len(search_res) == 0:
        raise(ValueError("Target not resolved"))
    
    # Download all the light curves
    print("Downloading light curves for {} sectors...".format(len(search_res)))
    lcfs = search_res.download_all()
    stitched_lc = lcfs.PDCSAP_FLUX.stitch()#corrector_func=my_custom_corrector_func)

    # Save the light curve
    save_file = os.path.join(save_path, "tess_lc_tic_{}.fits".format(tic_id))
    stitched_lc.to_fits(path=save_file, overwrite=True)

    return stitched_lc


def load_light_curve(tic_id, path="lightcurves", prefix="tess_lc_tic"):
    """Function to load a fits light curve fits file in

    Parameters
    ----------
    tic_id: int
        TIC id of the target.
    
    path: str
        Path to directory where fits light curves are saved.

    prefix: str, default: 'tess_lc_tic'
        Prefix of the light curve fits file, format is taken to be
        '<prefix>_{}.fits'.format(tic_id)

    Returns
    -------
    lc: lightkurve.lightcurve.TessLightCurve
        Loaded light curve.

    Raises
    ------
    FileNotFoundError:
        Raised if no light curve file exists for the given TIC ID.
    """
    # Load in the light curve initially as a TessLightCurveFile, then convert
    path = os.path.join(path, prefix + "_{}.fits".format(tic_id))
    lcf = lk.lightcurvefile.TessLightCurveFile(path)
    lcf.targetid = tic_id
    lc = lcf.FLUX
    
    return lc


def load_all_light_curves(tic_ids):
    """
    Parameters
    ----------
    tic_ids: int array
        TIC IDs of all target.

    Returns
    -------
    light_curves: array of lightkurve.lightcurve.TessLightCurve
        Loaded light curves.
    """
    light_curves = []
    unsuccessful = []

    # Load in light curves for for the TIC IDs specified, returning Non for 
    # those IDs without corresponding files
    for tic_id in tqdm(tic_ids, desc="Light curves loaded"):
        try:
            lc = load_light_curve(tic_id)
            light_curves.append(lc)
        except FileNotFoundError:
            light_curves.append(None)
            unsuccessful.append(tic_id)
    
    print("No light curve files found for TICs {}".format(unsuccessful))

    return light_curves

# -----------------------------------------------------------------------------
# Light curve fitting
# -----------------------------------------------------------------------------
def compute_lc_resid(params, folded_lc, t0, period, ldm, ldc, transit_dur, 
                     return_dict):
    """
    """
    # Initialise transit model. Note that since we've already phase folded the
    # light curve, t0 is at 0, and the period is 1
    bm_params = bm.TransitParams()
    bm_params.t0 = 0                  # time of inferior conjunction (days)
    bm_params.per = 1                 # orbital period (days)
    bm_params.rp = params[0]          # planet radius (stellar radii)
    bm_params.a = params[1]           # semi-major axis (stellar radii)
    bm_params.inc = params[2]         # orbital inclination (degrees)
    bm_params.ecc = 0                 # eccentricity
    bm_params.w = 0                   # longitude of periastron (degrees)
    bm_params.limb_dark = ldm         # limb darkening model
    bm_params.u = ldc                 # limb darkening coeff [u1, u2, u3, u4]

    xx = folded_lc.time# * period

    bm_model = bm.TransitModel(bm_params, xx)
    lc_model = bm_model.light_curve(bm_params)

    # Clean flux and uncertainties
    flux = folded_lc.flux
    e_flux = folded_lc.flux_err

    bad_flux_mask = ~np.isfinite(flux)

    e_flux[bad_flux_mask] = np.inf
    flux[bad_flux_mask] = 1

    # Calculate scaled residuals
    resid = (folded_lc.flux - lc_model) / folded_lc.flux_err

    # Save best fit transit model
    return_dict["bm_params"] = bm_params
    return_dict["bm_model"] = bm_model
    return_dict["lc_model"] = lc_model

    # Only consider the transit duration itself, x1 on either side (so 3x 
    # transit duration in total). Need to convert the transit duration to units
    # of phase
    td = transit_dur / period

    mask = np.logical_and(
        folded_lc.time > -1.5*td, 
        folded_lc.time < 1.5*td)

    resid[~mask] = 0

    return resid


def fit_light_curve(light_curve, t0, period, transit_dur, ld_coeff, 
                    ld_model="nonlinear"):
    """
    """
    BTJD_OFFSET = 2457000

    if t0 > BTJD_OFFSET:
        t0 -= BTJD_OFFSET

    # Phase fold the light curve
    folded_lc = light_curve.remove_outliers(sigma=6).fold(period=period, t0=t0)

    return_dict = {}

    # Initialise transit params. Ftting for: [rp, semi-major axis, inclination]
    # and assuming a circular orbit, so eccentricity=1, & longitude periastron
    # isn't relevant
    init_params = np.array([0.1, 10, 90,])
    bounds = ((0, 0, 60,), (1, 10000, 120,))

    args = (folded_lc, t0, period, ld_model, ld_coeff, transit_dur, return_dict)

    #scale = (1, 1, 1)
    #step = (10, 0.1, 0.1)

    # Do fit
    opt_res = least_squares(
        compute_lc_resid, 
        init_params, 
        jac="3-point",
        bounds=bounds,
        #x_scale=scale,
        #diff_step=step,
        args=args, 
    )

    print("Rp/R* = {:0.4f}".format(opt_res["x"][0]))
    print("a = {:0.4f}".format(opt_res["x"][1]))
    print("i = {:0.4f}".format(opt_res["x"][2]))
    print("e = {:0.4f}".format(opt_res["x"][3]))
    print("w = {:0.4f}".format(opt_res["x"][4]))

    # Calculate parameter uncertanties
    #jac = opt_res["jac"]
    #res = opt_res["fun"]
    #cov = np.linalg.inv(jac.T.dot(jac))
    #opt_res["std"] = np.sqrt(np.diagonal(cov)) * np.nanvar(res)

    return opt_res, return_dict

