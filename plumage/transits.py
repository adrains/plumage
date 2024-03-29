"""Module to interface with TESS light curves and fit
"""
import os
import numpy as np
import lightkurve as lk
try:
    import batman as bm
except:
    pass
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle 
from scipy.optimize import least_squares
from collections import OrderedDict

# Ensure the lightcurves folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "lightcurves"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# Convert between BJD and TESS BJD
BTJD_OFFSET = 2457000

class BatmanError(Exception):
    """Exception to throw when batman raises a generic exception. Mostly trying
    to catch this:

    'Exception: Convergence failure in calculation of scale factor for 
    integration step size'
    
    https://github.com/lkreidberg/batman/issues/27
    """
    pass

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

def download_lc_all_sectors(
    tic_id,
    source_id,
    save_fits=True, 
    save_path="lightcurves",
    do_binning=True,
    bin_fac=2,
    base_cadence=2/24/60,):
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

    # Before doing anything, grab the sectors
    sectors = [lc.sector for lc in lcfs.data]

    # If we're binning, do it here for each sector before stitching together
    if do_binning:
        binned_lc = []

        for lc_i, lc in enumerate(lcfs.PDCSAP_FLUX.data):
            total_time = lc.time[-1] - lc.time[0]
            total_elapsed_cadences = total_time / base_cadence
            n_bins = int(total_elapsed_cadences / bin_fac)
            binned_lc.append(lc.bin(bins=n_bins))

            suffix = "_binning_{}x".format(bin_fac)

        # Now make a new lightkurve collection
        lkc = lk.collections.LightCurveCollection(binned_lc)

        # Stitch together our new collection of binned light curves
        stitched_lc = lkc.stitch().remove_nans()

    else:
        # No binning, so no suffix
        suffix = "_binning_0x"

        # Just stitch together exisiting light curves
        stitched_lc = lcfs.PDCSAP_FLUX.stitch().remove_nans()

    # Save the light curve
    save_file = os.path.join(
        save_path, 
        "tess_lc_tic_{}{}.fits".format(tic_id, suffix))
    stitched_lc.to_fits(path=save_file, overwrite=True)

    return stitched_lc, sectors


def load_light_curve(
    tic_id,
    path="lightcurves",
    prefix="tess_lc_tic",
    bin_fac=2,):
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
    path = os.path.join(
        path, 
        prefix + "_{}_binning_{}x.fits".format(tic_id, bin_fac))
    lcf = lk.lightcurvefile.TessLightCurveFile(path)
    lcf.targetid = tic_id
    lc = lcf.FLUX
    
    return lc


def load_all_light_curves(tic_ids, bin_fac,):
    """Load in light curves, remove nans, and save to dict with TIC ID as key.

    Parameters
    ----------
    tic_ids: int array
        TIC IDs of all target.

    Returns
    -------
    light_curves: array of lightkurve.lightcurve.TessLightCurve
        Loaded light curves.
    """
    light_curves = {}
    unsuccessful = []

    # Load in light curves for for the TIC IDs specified, returning Non for 
    # those IDs without corresponding files
    desc = "Loading light curves, binning {}x".format(bin_fac)
    for tic_id in tqdm(tic_ids, desc=desc):
        try:
            lc = load_light_curve(tic_id, bin_fac=bin_fac,)
            light_curves[tic_id] = lc
        except FileNotFoundError:
            light_curves[tic_id] = None
            unsuccessful.append(tic_id)
    
    print("No light curve files found for TICs {}".format(unsuccessful))

    # Also load the sectors
    sector_list = np.loadtxt(
        "lightcurves/tess_lc_sectors_binning_x{}.tsv".format(bin_fac),
        delimiter="\t",
        dtype=str)

    # And convert to dict
    sector_dict = {int(tic): sectors for tic, sectors in sector_list}

    return light_curves, sector_dict


def get_sectors(tic,):
    """Queries for the number of sectors available for a given TIC
    """
    # Query
    search_res = lk.search_lightcurvefile("TIC {}".format(tic), mission="TESS")

    # Grab table actually containing info
    sectors = list(search_res.table["observation"])

    # Take just the sector numbers and sort
    sectors = [int(sector.split(" ")[-1]) for sector in sectors]

    sector_str = format_sectors(sectors)

    return sectors, sector_str


def format_sectors(sector_list):
    """Format sectors into succinct string
    """
    # Prepare
    sector_list = list(set(sector_list))
    sector_list.sort()

    # Format the sectors succinctly, e.g. 1-4, 28
    sector_str = str(sector_list[0])
    prev_sector = sector_list[0]
    holdover = False

    for sector in sector_list[1:]:
        if sector == prev_sector + 1:
            holdover = True
        elif holdover:
            sector_str += "-{},{}".format(prev_sector,sector)
            holdover = False
        else:
            sector_str += ",{}".format(sector)

        prev_sector = sector

    if holdover:
        sector_str += "-{}".format(prev_sector)

    return sector_str

# -----------------------------------------------------------------------------
# Light curve fitting
# -----------------------------------------------------------------------------
def compute_lc_resid_for_period_fitting(
    params,
    params_fit_keys,
    params_fixed,
    clean_lc,
    bm_lightcurve,
    trans_dur,
    n_trans_dur,
    verbose,):
    """Fold light curve on given period, and compute residuals

    Parameters
    ----------
    params_fit_keys: string list
        List of parameters that are to be fitted for, either:
         - 'period'
         - 't0'

    params_fixed: dict
        Dictionary pairing of parameter ('period', 't0') for those
        parameters that are fixed during fitting. Will contain those parameters
        not in params_fit_keys.
    """
    # Unpack params, first period
    Ti = np.argwhere(params_fit_keys=="period")
    period = params[int(Ti)] if len(Ti) > 0 else params_fixed["period"]

    # Now t0, the epoch
    ei = np.argwhere(params_fit_keys=="t0")
    t0 = params[int(ei)] if len(ei) > 0 else params_fixed["t0"]

    # Fold
    folded_lc = clean_lc.fold(period=period, t0=t0)

    # Clean flux and uncertainties
    flux = folded_lc.flux
    e_flux = folded_lc.flux_err

    bad_flux_mask = ~np.isfinite(flux)

    e_flux[bad_flux_mask] = np.inf
    flux[bad_flux_mask] = 1

    # Calculate scaled residuals
    resid = (flux - bm_lightcurve) / e_flux

    # Only consider the transit duration itself. Need to convert the transit 
    # duration to units of phase
    td = trans_dur / period

    mask = np.logical_and(
        folded_lc.time > -0.5*n_trans_dur*td, 
        folded_lc.time < 0.5*n_trans_dur*td)

    resid[~mask] = 0

    # Print
    if verbose:
        chi2 = np.sum(resid**2) / (len(resid)-len(params))
        print("T = {:0.6f}, t0 = {:0.6f}, chi^2 = {:0.4f}".format(
            period, t0, chi2))

    return resid


def fit_period(
    t0_init,
    e_t0_init,
    period_init,
    e_period_init,
    sma_rstar,
    inclination,
    ld_model,
    ld_coeff,
    trans_dur,
    clean_lc,
    n_trans_dur,
    bounds_frac=0.01,
    rp_r_star=None,
    verbose=True,
    do_plot=False,
    dt_step_fac=10,
    fit_for_period=True,
    fit_for_t0=False,):
    """Fit for the transit orbital period
    """
    # Convert between BJD and TESS BJD
    if t0_init > BTJD_OFFSET:
        t0_init -= BTJD_OFFSET
    
    # Setup which parameters we're fitting for. Must fit for at least one.
    if fit_for_period or fit_for_t0:
        fit_for_params = OrderedDict([
            ("period", fit_for_period),
            ("t0", fit_for_t0),])
    else:
        raise ValueError("Must fit for at least one of period or t0!")

    # Print an update, with our step size in seconds
    if verbose:
        period_text = "Fitting T (+/-{:0.2f} sec)".format(
                e_period_init*24*3600/dt_step_fac)
        t0_text = "T0 ({:0.2f} sec)".format(
                e_t0_init*24*3600/dt_step_fac)

        if fit_for_period and fit_for_t0:
            print(period_text, "and", t0_text)
        elif fit_for_period:
            print(period_text)
        else:
            print(t0_text)

    # Intial folding lightcurve
    folded_lc_init = clean_lc.fold(period=period_init, t0=t0_init)

    transit_mask = np.logical_and(
        folded_lc_init.time < trans_dur/period_init/8,
        folded_lc_init.time > -trans_dur/period_init/8)

    # If we haven't been given a guess for Rp/R*, use the depth
    if rp_r_star is None:
        depth = 1-np.nanmean(folded_lc_init.flux[transit_mask])

        if depth < 0:
            rp_r_star = 0.01
        else:
            rp_r_star = np.sqrt(depth)

    # Create initial lightcurve model
    bm_params, bm_model, bm_lightcurve = initialise_bm_model(
        t0=0,
        period=1,
        rp_rstar=rp_r_star,
        sma_rstar=sma_rstar,
        inclination=inclination,
        ecc=0,
        omega=0,
        ld_model=ld_model,
        ld_coeff=ld_coeff,
        time=folded_lc_init.time,)

    # Set +/- bounds
    pm_period = period_init*bounds_frac
    pm_t0 = t0_init*bounds_frac
    bounds = np.array([
        (period_init-pm_period, t0_init-pm_t0), 
        (period_init+pm_period,  t0_init+pm_t0)])

    # Now sort out our fitted for and fixed parameters
    param_mask = list(fit_for_params.values())

    # Initialise param initial guess *to be fit for* as a list, save keys
    params = {"period":period_init, "t0":t0_init}
    params_init = [params[pp] for pp in params if fit_for_params[pp]]
    params_init_keys = np.array([pp for pp in params if fit_for_params[pp]])

    # Keep the rest in dictionary form
    params_fixed = {pp:params[pp] for pp in params if not fit_for_params[pp]}

    args = (params_init_keys, params_fixed, clean_lc, bm_lightcurve, trans_dur,
            n_trans_dur, verbose)

    # Set the step size to be dt_step_fac x smaller than our uncertainty
    period_step_frac = (e_period_init / dt_step_fac) / period_init
    t0_step_frac = (e_t0_init / dt_step_fac) / t0_init
    step = np.array([period_step_frac, t0_step_frac])

    # Do fit
    opt_res = least_squares(
        compute_lc_resid_for_period_fitting, 
        params_init, 
        jac="3-point",
        bounds=bounds[:,param_mask],
        #x_scale=scale,
        diff_step=step[param_mask],
        args=args, 
    )

    # Calculate uncertainties
    jac = opt_res["jac"]
    res = opt_res["fun"]
    
    # TODO remove this equivalent code from inside the residual function
    td = trans_dur / period_init
    transit_window_mask = np.logical_and(
        folded_lc_init.time > -0.5*n_trans_dur*td, 
        folded_lc_init.time < 0.5*n_trans_dur*td)

    # Calculate RMS to scale uncertainties by
    rms = np.sqrt(np.sum(res**2)/np.sum(transit_window_mask))

    cov = np.linalg.inv(jac.T.dot(jac))
    std = np.sqrt(np.diagonal(cov)) * rms #np.nanvar(res)

    opt_res["cov"] = cov
    opt_res["std"] = std

    # Plot
    if do_plot:
        period = opt_res["x"][0] if fit_for_params["period"] else period_init

        if fit_for_params["period"]:
            t0 = opt_res["x"][1] if fit_for_params["t0"] else t0_init
        else:
            t0 = opt_res["x"][0] if fit_for_params["t0"] else t0_init

        folded_lc = clean_lc.fold(period=period, t0=t0)
        plt.close("all")
        ax = folded_lc.errorbar() 
        folded_lc.scatter(ax=ax) 
        ax.plot(folded_lc.time, bm_lightcurve, "r-")

        import pdb
        pdb.set_trace()

    return opt_res


def initialise_bm_model(t0, period, rp_rstar, sma_rstar, inclination, ecc, 
    omega, ld_model, ld_coeff, time, fac=1e-4):
    """Generates batman params, model, and light curve given orbital 
    parameters. Note that units can either be in time or phase, so long as they
    are consistent.

    Parameters
    ----------
    t0: float
        Epoch (i.e time of transit). Should be set to zero if using units of 
        phase.
    
    period: float
        Period of transit. Should be set to one if using units of phase.

    rp_star: float
        Planetary radii in units of stellar radius.

    sma_rstar: float
        Semi-major axis in units of stellar radius.

    inclination: float
        Inclination in degrees, with 90 deg being edge on,

    ecc: float
        Eccentricity, with 0 indicating a circular orbit.

    omega: float
        Longitude of periastron in degrees. Note that this parameter is only
        relevant for eccentric orbits.

    ld_model: str
        Limb darkening model compatible with batman.

    ld_coeff: float array
        Vector of limb darkening coefficients associated with ld_model.

    time: float array
        Observed time vector to generate model transit at. In units of either
        days or phase.

    Returns
    -------
    bm_params: batman.transitmodel.TransitParams
        Batman transit parameter + time object.

    bm_model: batman.transitmodel.TransitModel
        Batman transit model.

    bm_lightcurve: float array
        Batman transit model fluxes associated with time.
    """
    # Initialise transit model
    bm_params = bm.TransitParams()
    bm_params.t0 = t0               # time of inferior conjunction (days/phase)
    bm_params.per = period          # orbital period (days/phase)
    bm_params.rp = rp_rstar         # planet radius (stellar radii)
    bm_params.a = sma_rstar         # semi-major axis (stellar radii)
    bm_params.inc = inclination     # orbital inclination (degrees)
    bm_params.ecc = ecc             # eccentricity
    bm_params.w = omega             # longitude of periastron (degrees)
    bm_params.limb_dark = ld_model  # limb darkening model
    bm_params.u = ld_coeff          # limb darkening coeff [u1, u2, u3, u4]

    # Batman unfortunately doesn't have custom exceptions, and is known to just
    # raise the default exception, making it difficult to catch. By wrapping
    # calls as such, we can raise (and thus catch) our own batman exceptions.
    try:
        bm_model = bm.TransitModel(bm_params, time, )
        bm_lightcurve = bm_model.light_curve(bm_params)
    except:
        print("Batman error, forcing fac={}".format(fac))
        bm_model = bm.TransitModel(bm_params, time, fac=fac)
        bm_lightcurve = bm_model.light_curve(bm_params)

        #raise BatmanError(
        #    "Batman unhappy with model generation at specified parameters.")

    return bm_params, bm_model, bm_lightcurve


def compute_lc_resid(
    params, 
    folded_lc, 
    t0, 
    period, 
    ld_model, 
    ld_coeff, 
    trans_dur, 
    sma_rstar, 
    e_sma_rstar, 
    verbose=True, 
    n_trans_dur=2):
    """Computes residuals between the folded light curve provided, and a batman
    model transit light curve generated using params.

    Parameters
    ----------
    params: float array
        Array of form [rp_rstar, sma_rstar, inclination].
    
    folded_lc: lightkurve.lightcurve.FoldedLightCurve
        TESS light curve folded about the planet transit epoch and period. Note
        that this means the new 'period' is 1, and the 'epoch' (i.e. time of
        transit is 0).

    t0: float
        Epoch, i.e. time of first transit, in days.
    
    period: float
        Period of the planet in days.

    ld_model: str
        Limb darkening model compatible with batman.

    ld_coeff: float array
        Vector of limb darkening coefficients associated with ld_model.

    trans_dur: float
        Transit duration in days.
    
    sma_rstar, e_sma_rstar: float
        Prior on the stellar radius scaled semi-major axis and its uncertainty 
        from the stellar mass, orbital period, and stellar radius.
    
    verbose: boolean, default: True
        Whether to print information about each iteration of fitting.

    n_trans_dur: float, default: 2
        Specifies the window width, in units of transit duration, that  
        residuals should be computed for. e.g. a value of 2 means that there
        is 0.5x transit_dur on either side of the transit worth of points.
        
    Returns
    -------
    resid: float array
        Vector of uncertainty weighted residuals, set to zero outside the 
        region set by trans_dur and n_trans_dur.
    """
    # Extract params
    rp_rstar_guess = params[0]
    sma_rstar_guess = params[1]
    inclination_guess = np.arcsin(params[2])*180/np.pi

    # Initialise transit model. Note that since we've already phase folded the
    # light curve, t0 is at 0, and the period is 1. We're also assuming a 
    # circular orbit, so e=0, and omega=0 (but is irrelevant)
    bm_params, bm_model, bm_lightcurve = initialise_bm_model(
        t0=0, 
        period=1, 
        rp_rstar=rp_rstar_guess, 
        sma_rstar=sma_rstar_guess, 
        inclination=inclination_guess, 
        ecc=0, 
        omega=0, 
        ld_model=ld_model, 
        ld_coeff=ld_coeff, 
        time=folded_lc.time,)

    # Check to ensure there the planet is still transiting
    is_transiting = int(np.sum(bm_lightcurve)) != len(bm_lightcurve)

    # Clean flux and uncertainties
    flux = folded_lc.flux
    e_flux = folded_lc.flux_err

    bad_flux_mask = ~np.isfinite(flux)

    e_flux[bad_flux_mask] = np.inf
    flux[bad_flux_mask] = 1

    # Calculate scaled residuals
    resid = (flux - bm_lightcurve) / e_flux

    # Only consider the transit duration itself. Need to convert the transit 
    # duration to units of phase
    td = trans_dur / period

    mask = np.logical_and(
        folded_lc.time > -0.5*n_trans_dur*td, 
        folded_lc.time < 0.5*n_trans_dur*td)

    resid[~mask] = 0

    # Penalise the fit if the planet is not transiting
    if not is_transiting:
        resid *= 10

    # Add our prior on the SMA to the residual vector
    sma_resid = [(sma_rstar - sma_rstar_guess) / e_sma_rstar]
    resid = np.concatenate((resid, sma_resid))

    # Print updates
    if verbose:
        print("Rp/R* = {:0.5f}, a = {:0.05f} [{:0.5f}+/-{:0.5f}], i = {:0.05f}".format(
            rp_rstar_guess, sma_rstar_guess, sma_rstar, e_sma_rstar, 
            inclination_guess), end="")
        
        rchi2 = np.sum(resid**2) / (np.sum(mask)-len(params))
        print("\t--> rchi^2 = {:0.5f}".format(rchi2), end="")

        if not is_transiting:
            print("\t [WARNING - planet is no longer transiting]")
        else:
            print("")

    return resid


def fit_light_curve(
    light_curve, 
    t0,
    e_t0_init,
    period,
    e_period_init, 
    trans_dur, 
    ld_coeff, 
    sma_rstar, 
    e_sma_rstar,
    mask,
    ld_model="nonlinear", 
    flatten_frac=0.1, 
    outlier_sig=6,
    niters_flat=5,
    t_min=1/24,
    verbose=True, 
    n_trans_dur=2,
    bin_lightcurve=False,
    break_tolerance=100,
    do_period_and_t0_ls_fit=False,
    fitting_iterations=2,
    force_window_length_to_min=True,
    break_tol_days=0.5,
    fit_for_period=True,
    fit_for_t0=False,
    dt_step_fac=10,
    do_period_fit_plot=False,
    n_trans_dur_period_fit=4,):
    """Perform least squares fitting on the provided light curve to determine
    Rp/R*, a/R*, and inclination.

    Parameters
    ----------
    light_curve: lightkurve.lightcurve.TessLightCurve
        Lightkurve object containing TESS time-series photometry from NASA.

    t0: float
        Time of first transit in Barycentric Julian Day (BJD).

    period: float
        Transit period in days.

    transit_dur: float
        Transit duration in days.

    ld_coeff: float array
        Vector of limb darkening coefficients of form ld_model.

    sma_rstar, e_sma_rstar: float
        Prior on the stellar radius scaled semi-major axis and its uncertainty 
        from the stellar mass, orbital period, and stellar radius.

    ld_model: str, default: 'nonlinear'
        Kind of limb darkening model to use.

    flat_frac: float, default: 0.1
        Fractional length of the lightcurve to use as the Savitzky-Golay 
        smoothing window length - e.g. 0.1 is 10% of the total light curve.

    outlier_sig: float, default: outlier_sig
        Sigma value beyond which to remove outliers when cleaning light curve.

    Returns
    -------
    opt_res: dict
        Dictionary of least squares fit results.
    """
    # Convert between BJD and TESS BJD
    if t0 > BTJD_OFFSET:
        t0 -= BTJD_OFFSET

    # Before cleaning however, we should mask out the transit signal itself
    #clean_lc = light_curve.remove_outliers(sigma=outlier_sig)

    # Get window size for smoothing (note mask inversion)
    window_length = determine_window_size(
        light_curve,
        t0,
        period,
        ~mask,
        t_min,
        force_window_length_to_min=force_window_length_to_min,)

    # Set break tolerance
    cadence = np.median(light_curve.time[1:] - light_curve.time[:-1])
    break_tolerance = int(break_tol_days / cadence)

    clean_lc, flat_lc_trend = light_curve.flatten(
        window_length=window_length,
        return_trend=True,
        niters=niters_flat,
        break_tolerance=break_tolerance,
        mask=mask)

    #import pdb
    #pdb.set_trace()

    # If we aren't fitting the period, only do a single iteration
    if not do_period_and_t0_ls_fit:
        print("Not fitting for period or t0")
        fitting_iterations = 1
    
    # Initialise
    rp_r_star = None
    inc_deg = 90
    sin_inc = np.sin(inc_deg*np.pi/180)

    for fit_i in range(fitting_iterations):
        print("\nFitting iteration #{:0.0f}".format(fit_i))

        # If the lightcurve has data from sectors widely spaced in time (more
        # than a year), fit for the period as the given values are often wrong.
        # Only do this however once we have already done a lightcurve fit to
        # get the general shape - i.e. if fitting for the period, this happens
        # on iterations 2+
        if do_period_and_t0_ls_fit and fit_i >= 1:
            opt_res_period_fit = fit_period(
                t0_init=t0,
                e_t0_init=e_t0_init,
                period_init=period,
                e_period_init=e_period_init,
                sma_rstar=sma_rstar,
                inclination=inc_deg,
                ld_model=ld_model,
                ld_coeff=ld_coeff,
                trans_dur=trans_dur, 
                clean_lc=clean_lc, 
                n_trans_dur=n_trans_dur_period_fit,
                rp_r_star=rp_r_star,
                fit_for_period=fit_for_period,
                fit_for_t0=fit_for_t0,
                dt_step_fac=dt_step_fac,
                do_plot=do_period_fit_plot,) 
            
            # If fit period, extract
            if fit_for_period:
                new_period = float(opt_res_period_fit["x"][0])
                e_period = opt_res_period_fit["std"][0]

                # Print update
                delta_period = (period - new_period) * 24 * 3600
                print("Period: {:0.6f} --> {:0.6f} days [{:+0.4f} sec]".format(
                    period, new_period, delta_period))

                # Save
                period = new_period
            
            # If fit for t0 extract
            if fit_for_t0:
                # Set the index
                if fit_for_period:
                    t0_i = 1
                else:
                    t0_i = 0
                
                new_t0 = float(opt_res_period_fit["x"][t0_i])
                e_t0 = opt_res_period_fit["std"][t0_i]
                
                # Print update
                delta_t0 = (t0 - new_t0) * 24 * 3600
                print("t0: {:0.6f} --> {:0.6f} days [{:+0.4f} sec]".format(
                    t0, new_t0, delta_t0))

                t0 = new_t0

        # Phase fold the light curve
        folded_lc = clean_lc.fold(period=period, t0=t0)

        # If this is our first pass through, have a guess at Rp/R_*. 
        # Otherwise use the fitted value from the last iteration
        if rp_r_star is None:
            transit_mask = np.logical_and(
                folded_lc.time < trans_dur/period/8,
                folded_lc.time > -trans_dur/period/8)
                
            depth = 1-np.nanmean(folded_lc.flux[transit_mask])

            if depth < 0:
                rp_r_star = 0.01
            else:
                rp_r_star = np.sqrt(depth)

        # Initialise transit params. Ftting for: 
        # [rp, semi-major axis, sin(inclination)]
        # We're assuming a circular orbit, so eccentricity=1, & longitude 
        # periastron isn't relevant
        init_params = np.array([rp_r_star, sma_rstar, sin_inc,])
        bounds = ((0.001, 0.001, 0,), (1, 10000, 1,))

        args = (folded_lc, t0, period, ld_model, ld_coeff, trans_dur, 
                sma_rstar, e_sma_rstar, verbose, n_trans_dur,)

        #scale = (1, 1, 1)
        step = (0.005, 0.005, 0.000001)

        # Do fit
        print("\n Running lightcurve fit:")
        opt_res = least_squares(
            compute_lc_resid, 
            init_params, 
            jac="3-point",
            bounds=bounds,
            #x_scale=scale,
            diff_step=step,
            args=args, 
        )

        # Extract params from fit so we can either iterate or continue
        rp_r_star = opt_res["x"][0]
        #sma_rstar = opt_res["x"][1]
        
        # Convert inclination back to angle
        sin_inc = opt_res["x"][2]
        inc_deg = np.arcsin(sin_inc)*180/np.pi

    # TODO remove this equivalent code from inside the residual function
    td = trans_dur / period
    transit_window_mask = np.logical_and(
        folded_lc.time > -0.5*n_trans_dur*td, 
        folded_lc.time < 0.5*n_trans_dur*td)

    # Calculate uncertainties
    jac = opt_res["jac"]
    res = opt_res["fun"]
    
    # Calculate RMS to scale uncertainties by
    rms = np.sqrt(np.sum(res**2)/np.sum(transit_window_mask))

    # If jacobian is entirely 0, or if anything other than the inclination 
    # column is zero, assume singular, something went wrong entire with the 
    # fit. Set statistical uncertainties to nan
    if np.sum(jac) == 0 or np.sum(jac[:,0]) == 0 or np.sum(jac[:,1]) == 0:
        print("\nWarning, singular matrix!")
        std = np.full(3, np.nan)

    # If just the inclination axis of the jacobian is 0, then ignore this when
    # computing uncertainties and set nan
    elif np.sum(jac[:,2]) == 0:
        cov = np.linalg.inv(jac[:,:2].T.dot(jac[:,:2]))
        std = np.sqrt(np.diagonal(cov)) * rms #np.nanvar(res)
        std = np.concatenate((std, np.atleast_1d(np.nan)))

    # Everthing is behaving normally!
    else:
        cov = np.linalg.inv(jac.T.dot(jac))
        std = np.sqrt(np.diagonal(cov)) * rms #np.nanvar(res)

    # Uncertainty is d(sin(i))/d(i) * sigma sin(i)
    # where d(sin(i))/d(i) = 1 / sqrt(1-x**2)
    e_inc_deg = std[2] / np.sqrt(1-sin_inc**2)

    opt_res["x"][2] = inc_deg
    std[2] = e_inc_deg

    # Generate and save batman model at optimal params
    bm_params, bm_model, bm_lightcurve = initialise_bm_model(
        t0=0, 
        period=1, 
        rp_rstar=opt_res["x"][0], 
        sma_rstar=opt_res["x"][1], 
        inclination=opt_res["x"][2], 
        ecc=0, 
        omega=0, 
        ld_model=ld_model, 
        ld_coeff=ld_coeff, 
        time=folded_lc.time,)

    # Calculate rchi^2
    rchi2 = np.sum(opt_res["fun"]**2) / (np.sum(~(opt_res["fun"] == 0))-3)

    # Add period and t0 fits to dict
    if do_period_and_t0_ls_fit and fit_for_period:
        opt_res["period_fit"] = period
        opt_res["e_period_fit"] = e_period
    else:
        opt_res["period_fit"] = np.nan
        opt_res["e_period_fit"] = np.nan
    
    if do_period_and_t0_ls_fit and fit_for_t0:
        opt_res["t0_fit"] = t0
        opt_res["e_t0_fit"] = e_t0
    else:
        opt_res["t0_fit"] = np.nan
        opt_res["e_t0_fit"] = np.nan

    # And add everything else
    opt_res["clean_lc"] = clean_lc
    opt_res["folded_lc"] = folded_lc
    opt_res["window_length"] = window_length
    opt_res["niters_flat"] = niters_flat
    opt_res["flat_lc_trend"] = flat_lc_trend
    opt_res["rchi2"] = rchi2
    opt_res["std"] = std
    opt_res["bm_params"] = bm_params
    opt_res["bm_model"] = bm_model
    opt_res["bm_lightcurve"] = bm_lightcurve

    return opt_res


def make_transit_mask_single_period(
    light_curve, 
    t0, 
    period, 
    trans_dur, 
    plot_check=False):
    """

    Parameters
    ----------
    light_curve: lightkurve.lightcurve.TessLightCurve
        Lightkurve object containing TESS time-series photometry from NASA.

    t0: float
        Time of first transit in Barycentric Julian Day (BJD).

    period: float
        Transit period in days.

    transit_dur: float
        Transit duration in days.

    plot_check: bool, default: False
        Indicates whether to plot a diagnostic check on the mask

    Returns
    -------
    mask: bool array
        True where transits occur
    """
    # Work in modulo time of the period
    mod_time = (light_curve.time - t0) % period

    # t0 is the centre of the transit, so we have to grab the times and the 
    # beginning *end* the end of the modulo period (the second and first halves
    # of the transit respectively)
    trans_2nd_half = mod_time < (trans_dur)
    trans_1st_half = mod_time > (period-trans_dur)

    mask = np.logical_or(trans_2nd_half, trans_1st_half)

    if plot_check:
        plt.close("all")
        plt.plot(light_curve.time, light_curve.flux, linewidth=0.1)
        plt.plot(light_curve.time[mask], light_curve.flux[mask], linewidth=0.1)

    return mask


def make_transit_mask_all_periods(light_curve, toi_info, tic_id):
    """
    """
    selected_tois = toi_info[toi_info["TIC"]==tic_id]

    mask = np.full(len(light_curve.time), False)

    for star_i, toi_row in selected_tois.iterrows():
        # Only contribute to mask if we have non-nan parameters
        if np.isnan(toi_row["Duration (hours)"]):
            print("Skipped TOI {} when making mask".format(toi_row.name))
            continue

        mask = np.logical_or(
            mask,
            make_transit_mask_single_period(
                light_curve,
                toi_row["Transit Epoch (BJD)"]-BTJD_OFFSET,
                toi_row["Period (days)"],
                toi_row["Duration (hours)"]/24,)
        )

        if np.sum(mask) == 0:
            raise Exception("Mask is entirely zero")
    
    return mask


def determine_window_size(light_curve, t0, period, mask, t_min=1/24,
    show_diagnostic_plot=False, force_window_length_to_min=False,):
    """
    mask: bool arrays
        True where we should use.
    """
    # Get cadence of observations
    cadence = np.median(light_curve.time[1:] - light_curve.time[:-1])

    # Use periodogram if we're not forcing to the minimum
    if not force_window_length_to_min:
        # Get periodogram
        freq, power = LombScargle(light_curve.time[mask], light_curve.flux[mask]).autopower()

        # Consider only those frequencies lower than t_min days
        freq_mask = freq < t_min**-1
        stellar_period = 1 / freq[np.argmax(power[freq_mask])]

        # Calculate window length
        window_length = int(stellar_period / cadence / 4)

        print("Fitted stellar period for window: {:0.2f} days".format(
            stellar_period))

    # Otherwise just do based on minimum
    else:
        #print("Window from provided t_min")
        window_length = int(t_min / cadence)

    if show_diagnostic_plot and not force_window_length_to_min:
        plt.plot(1/freq, power)
    
    if window_length % 2 == 0:
        window_length += 1

    return window_length


def flatten_and_clean_lc(
    light_curve,
    toi_info,
    tic,
    toi,
    break_tol_days,
    t0,
    period,
    t_min,
    force_window_length_to_min,):
    """Flattens and cleans the given light curve
    """
    # Get mask for all transits with this system
    mask = make_transit_mask_all_periods(
        light_curve, 
        toi_info, 
        tic)

    # Flatten
    cadence = np.median(light_curve.time[1:] - light_curve.time[:-1])
    break_tolerance = int(break_tol_days / cadence)

    window_length = determine_window_size(
        light_curve,
        t0,
        period,
        ~mask,
        t_min,
        force_window_length_to_min=force_window_length_to_min,)

    clean_lc, flat_lc_trend = light_curve.flatten(
        window_length=window_length,
        return_trend=True,
        niters=int(toi_info.loc[toi]["niters_flat"]),
        break_tolerance=break_tolerance,
        mask=mask)

    return clean_lc, flat_lc_trend