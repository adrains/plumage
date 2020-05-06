"""Module to interface with TESS light curves and fit
"""
import os
import numpy as np
import lightkurve as lk
import batman as bm

# Ensure the lightcurves folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "lightcurves"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

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
    stitched_lc = lcfs.PDCSAP_FLUX.stitch()

    # Save the light curve
    save_file = os.path.join(save_path, "tess_lc_tic_{}.fits".format(tic_id))
    stitched_lc.to_fits(path=save_file, overwrite=True)

    return stitched_lc


# -----------------------------------------------------------------------------
# Light curve fitting
# -----------------------------------------------------------------------------
def compute_lc_resid(params, bjds, lc_flux, e_lc_flux, t0, period, ldm, ldc):
    """
    """
    # Initialise transit model
    params = batman.TransitParams()
    params.t0 = t0                      # time of inferior conjunction (days)
    params.per = period                 # orbital period (days)
    params.rp = params[0]               # planet radius (stellar radii)
    params.a = params[1]                # semi-major axis (stellar radii)
    params.inc = params[2]              # orbital inclination (degrees)
    params.ecc = params[3]              # eccentricity
    params.w = params[4]                # longitude of periastron (degrees)
    params.limb_dark = ldm              # limb darkening model
    params.u = ldc                      # limb darkening coeff [u1, u2, u3, u4]

    lc_model = batman.TransitModel(params, bjds)

    resid = (lc_flux - lc_model) / e_lc_flux

    return resid


def fit_light_curve(bjds, lc_flux, e_lc_flux, t0, period, ld_coeff, ld_model="quadratic"):
    """
    """
    # Phase fold the light curve
    fslc = slc.fold(period=1.80081, t0=efi.loc[200]["Epoch (BJD)"]-2457000)

    # Initialise transit params
    # [rp, semi-major axis, inclination, eccentricity, longitude periastron]
    params = np.array([0.1, 10, 90, 0, 90])
    args = (bjds, lc_data, e_lc_data, t0, period, ld_model, ld_coeff)

    bounds = ((0, 0, 0, 0, 0), (1, 10000, 360, 1, 360))
    #scale = (1, 1, 1)
    #step = (10, 0.1, 0.1)

    # Do fit
    optimize_result = least_squares(
        compute_lc_resid, 
        params, 
        jac="3-point",
        bounds=bounds,
        #x_scale=scale,
        #diff_step=step,
        args=args, 
    )

    # Calculate parameter uncertanties

    # Prep for LS fitting
    pass