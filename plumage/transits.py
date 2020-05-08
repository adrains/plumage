"""Module to interface with TESS light curves and fit
"""
import os
import numpy as np
import lightkurve as lk
import batman as bm
from scipy.optimize import least_squares

# Ensure the lightcurves folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "lightcurves"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)


def my_custom_corrector_func(lc):
    corrected_lc = lc.normalize().flatten(window_length=401)
    return corrected_lc

def download_lc_all_sectors(tic_id, save_fits=True, save_path="lightcurves"):
    """
    https://docs.lightkurve.org/tutorials/01-lightcurve-files.html

    SAP: Simple Aperture Photometry
    PDCSAP: Pre-search Data Conditioning SAP
    """
    # Search for light curve results for our star
    search_res = lk.search_lightcurvefile("TIC {}".format(tic_id), 
                                          mission="TESS")

    # Raise exception if we don't match with a target
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


def download_lc_all_tics(tic_ids):
    """
    """


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
    bm_params.t0 = 0                      # time of inferior conjunction (days)
    bm_params.per = 1                 # orbital period (days)
    bm_params.rp = params[0]               # planet radius (stellar radii)
    bm_params.a = params[1]                # semi-major axis (stellar radii)
    bm_params.inc = params[2]              # orbital inclination (degrees)
    bm_params.ecc = params[3]              # eccentricity
    bm_params.w = params[4]                # longitude of periastron (degrees)
    bm_params.limb_dark = ldm              # limb darkening model
    bm_params.u = ldc                      # limb darkening coeff [u1, u2, u3, u4]

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
    btjd_offset = 2457000

    if t0 > btjd_offset:
        t0 -= btjd_offset

    # Phase fold the light curve
    folded_lc = light_curve.remove_outliers(sigma=6).fold(period=period, t0=t0)

    return_dict = {}

    # Initialise transit params
    # [rp, semi-major axis, inclination, eccentricity, longitude periastron]
    init_params = np.array([0.1, 10, 90, 0, 90])
    bounds = ((0, 0, 60, 0, 0), (1, 10000, 120, 1, 360))

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

