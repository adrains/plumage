"""Functions to work with MIKE spectra.

TODO: there should be a dedicated spectra module, with e.g.:
    plumage.spectra.wifes and plumage.spectra.mike
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.ticker as plticker
from scipy.optimize import least_squares
from astropy.stats import sigma_clip
from numpy.polynomial.polynomial import Polynomial
from PyAstronomy.pyasl import instrBroadGaussFast
import plumage.utils_mike as pum

HEADERS_TO_EXTRACT = ["SITENAME", "SITEALT", "SITELAT", "SITELONG", "TELESCOP",
    "OBSERVER", "DATE-OBS", "UT-DATE", "UT-START", "LC-START", "INSTRUME",
    "RA-D", "DEC-D", "AIRMASS", "OBJECT", "EXPTIME", "NLOOPS", "BINNING",
    "SPEED", "SLITSIZE",]

# -----------------------------------------------------------------------------
# Utilities functions
# -----------------------------------------------------------------------------
def interpolate_wavelength_scale(
    wave_old,
    spec_old,
    wave_new,
    rv_shift,
    interpolation_method="cubic",):
    """Interpolates spectra to a new wavelength scale, and optionally applies
    an added Doppler shift to e.g. perform a barycentric correction.

    Parameters
    ----------
    wave_old, spec_old, wave_new: 2D float array
        Old wavelength and spectra arrays, plus new wavelength scale, all of
        shape [n_order, n_px].

    rv_shift: float
        RV shift to apply as a doppler shift when interpolating to the new
        wavelength scale. No shift is done if this value is 0.

    interpolation_method: str, default: "cubic"
        Default interpolation method to use with scipy.interp1d. Can be one of: 
        ['linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', or 'next'].

    Returns
    -------
    spec_new: 2D float array
        Interpolated spectra array, of shape [n_order, n_px].
    """
    # Initialise new flux array
    (n_order, n_px) = wave_old.shape
    spec_new = np.full((n_order, n_px), np.nan)

    # Interpolate spectra onto new wavelength scale order-by-order
    for order_i in range(n_order):
        # Create interpolator for only non-nan px
        is_finite = np.isfinite(spec_old[order_i])

        # If we have an order that is entirely NaN, skip interpolating it
        if np.sum(is_finite) == 0:
            spec_new[order_i] = spec_old[order_i]
            continue

        # Otherwise proceed with the interpolation
        interpolate_spec = interp1d(
            x=wave_old[order_i][is_finite],
            y=spec_old[order_i][is_finite],
            kind=interpolation_method,
            bounds_error=False,
            assume_sorted=True,)
        
        # [Optional] Apply a Doppler shift when doing the interpolation
        if rv_shift != 0:
            ds = rv_shift / const.c.cgs.to(u.km/u.s).value
            spec_new[order_i] = interpolate_spec(wave_new[order_i]*(1-ds))
            
        # Otherwise interpolate as normal
        else:
            spec_new[order_i] = interpolate_spec(wave_new[order_i])

    return spec_new


def compute_barycentric_correction(
    site_lat,
    site_long,
    site_alt,
    ras,
    decs,
    times,
    exps,):
    """Compute the barycentric corrections for a set of stars

    Parameters
    ----------
    ras: string array
        Array of right ascensions in string form: "HH:MM:SS.S".
    
    decs: string array
        Array of declinations in string form: "DD:MM:SS.S".

    times: string/float array
        Array of times in MJD format.

    site: string
        The site name to look up its coordinates.

    disable_auto_max_age: boolean, defaults to False
        Useful only when IERS server is not working.

    Returns
    -------
    bcors: float array
        Array of barycentric corrections in km/s.
    """
    # Get the location
    loc = EarthLocation.from_geodetic(
        lon=site_lat, lat=site_long, height=site_alt)

    # Initialise barycentric correction array
    bcors = []

    # Calculate the barycentric correction for every star
    for ra, dec, time, exp in zip(ras, decs, times, exps):
        sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

        # Get the *mid-point* of the observeration
        time = Time(time, format="isot") + 0.5*float(exp)*u.second
        barycorr = sc.radial_velocity_correction(obstime=time, location=loc)  
        bcors.append(barycorr.to(u.km/u.s).value)

    return bcors


def unify_wavelength_scale_per_exp(exp_dict,):
    """Unifies the dimensions of the wavelength scales for exposures of the
    same targets taken in sequence in the event one or more orders did not
    extract properly. Missing stellar orders are populated with NaNs, whereas
    the wavelength scale and any calibration fluxes are copied from the
    reference exposure which has all orders extracted.

    exp_dict is updated in place.

    HACK: currently we use the mean order wavelength (rather than the order
    number itself) to check for missing orders. This seems to work for adjacent
    exposures, but it would be more rigorous to use the order number.

    Parameters
    ----------
    all_exp_dict: dict
        Dictionary representation of MIKE fits files, with keys for relevant
        header keywords, the wavelength scale, and each of the seven (sky, obj,
        sigma, snr, lamp, flat, obj / flat) data-cubes. Each key corresponds to
        a list of length the number of exposures.
    """
    # These are [n_order, n_px] arrays we need to check the shapes of
    spec_keywords = ["wave", "spec_sky", "spec_star", "sigma_star", "snr_star", 
                     "spec_lamp", "spec_flat", "spec_star_norm"]

    # These are the arrays that, if missing, we adopt from our reference exp
    spec_keywords_to_dup = ["wave", "spec_sky", "spec_lamp", "spec_flat",]

    # Grab the number of exposures for reference
    n_exp = len(exp_dict["wave"])

    # Check all wavelength scales
    wl_shapes = [exp_dict["wave"][exp_i].shape for exp_i in range(n_exp)]

    # If we only have one shape, all wavelength scales are the same, return
    if len(set(wl_shapes)) == 1:
        print("\tOrders all equal, continuing")
        return

    # Determine the reference wavelength scale to adopt based on which has the
    # most px. 
    total_px = np.prod(wl_shapes, axis=1)
    ref_exp_i = np.argmax(total_px)
    wave_ref = exp_dict["wave"][ref_exp_i].copy()

    # Compute (rounded) order wavelengths for this selected exposure
    orders_ref = exp_dict["orders"][ref_exp_i]
    order_means_ref = (np.round(np.mean(wave_ref/5, axis=1))*5).astype(int)

    # Loop over all exposures and determine which order/s are missing
    for exp_i in range(n_exp):
        # Skip the reference exposure
        if exp_i == ref_exp_i:
            continue
        
        print("\tCorrecting for missing order on {} {}, {} --> {}".format(
            exp_dict["DATE-OBS"][exp_i],
            exp_dict["OBJECT"][exp_i],
            wl_shapes[exp_i],
            wave_ref.shape,
            ),)

        wave_ith = exp_dict["wave"][exp_i]

        # Compute (rounded) order wavelength scales for this exposure
        order_means = (np.round(np.mean(wave_ith/5, axis=1))*5).astype(int)
        
        # Determine which orders are missing
        matched_orders = np.isin(order_means_ref, order_means)
        
        # For all [n_order, n_px] arrays for this exposure, adopt the values
        # from this exposure where possible. For missing orders we either
        # simply adopt the reference values (for the wavelength scale and
        # calibration data), or add a dummy empty order full of NaNs (for the
        # stellar spectrum).
        for spec_key in spec_keywords:
            # Grab the arrays for the reference and current exposures
            array_ref = exp_dict[spec_key][ref_exp_i]
            array_ith = exp_dict[spec_key][exp_i]

            # Make a new array with the reference shape, and copy over the
            # values from the overlapping orders. 
            spec_array_new = np.full_like(wave_ref, np.nan)
            spec_array_new[matched_orders,:] = array_ith

            # For the wavelength scale and calibration data, duplicate the
            # missing order/s rather than simply leaving them NaN as we do for
            # the star.
            if spec_key in spec_keywords_to_dup:
                spec_array_new[~matched_orders,:] = \
                    array_ref[~matched_orders,:]

            # Save the new array back to the list
            exp_dict[spec_key][exp_i] = spec_array_new

        # Update the orders
        exp_dict["orders"][exp_i] = orders_ref

    # All done, everything has been updated in place


def unify_wavelength_scales_global(info_dict, return_new_shape=False,):
    """Unifies the dimensions of the wavelength scales given a set of MIKE
    observations taken on multiple nights and multiple targets, since it is
    possible for a) different numbers of orders to extract based on the blaze
    angle, and b) sometimes orders fail to extract.  Missing stellar orders are
    populated with NaNs, whereas the wavelength scale and any calibration
    fluxes are copied from the reference exposure which has all orders
    extracted. info_dict is updated in place, and orders are sorted in
    ascending order.

    Very similar in function to unify_wavelength_scale_per_exp, but there's a
    greater variety of differences when considering all MIKE data, vs just
    adjacent sets of exposures.

    Parameters
    ----------
    info_dict: dict
        Dictionary representation of MIKE fits files (either for the blue or
        red arm), with (source_id, date) keys linked to a dictionary containing
        relevant header keywords, order numbers, the wavelength scale, fluxes,
        and uncertanties.

    return_new_shape: bool, default: False
        Whether or not to return the adopted shape, [n_order, n_px].

    Returns
    -------
    adopted_wl_shape: float array
        Optional return value, the shape of the unified wavelength scale as
        [n_order, n_px].
    """
    # Count unique orders to initialise
    unique_order_nums = []

    n_px_list = []

    for obj in info_dict.keys():
        if info_dict[obj] is not None:
            unique_order_nums += list(info_dict[obj]["orders"])

            n_px_list.append(info_dict[obj]["wave"].shape[1])

    # Sanity checking
    assert len(set(n_px_list)) == 1
    
    # Select or sort the orders our data encompasses
    unique_order_nums = np.array(list(set(unique_order_nums)))
    unique_order_nums.sort()

    # Grab adopted dimensions
    n_px = n_px_list[0]
    n_order = len(unique_order_nums)

    # Loop over all objects and expand where necessary
    for obj in info_dict.keys():
        array_keys = ["wave", "spec_star_norm", "sigma_star_norm"]
        
        # Only proceed if we actually have data for this arm (None if not)
        if (info_dict[obj] is not None 
            and info_dict[obj]["wave"].shape != (n_order, n_px)):
            orders = info_dict[obj]["orders"]
            matched_orders = np.isin(unique_order_nums, orders)

            for ak in array_keys:
                vec_old = info_dict[obj][ak]
                vec_new = np.full((n_order, n_px), np.nan)

                vec_new[matched_orders,:] = vec_old

                # Save the new array back to the list
                info_dict[obj][ak] = vec_new

            # Update order list
            info_dict[obj]["orders"] = unique_order_nums

        else:
            continue

    # Everything updated in place, but optionally return shape
    if return_new_shape:
        return (n_order, n_px)


# -----------------------------------------------------------------------------
# Reading fits files
# -----------------------------------------------------------------------------
def load_single_mike_fits(fn):
    """Function to extract relevant header information + the datacube from a
    MIKE fits file, create the wavelength scale, and export as a dict.

    Parameters
    ----------
    fn: str
        Filename to open.

    Returns
    -------
    data_dict: dict
        Dictionary representation of MIKE fits file, with keys for relevant
        header keywords, the wavelength scale, and each of the seven (sky, obj,
        sigma, snr, lamp, flat, obj / flat) data-cubes.
    """
    data_dict = {}

    with fits.open(fn) as ff:
        # Grab data, header, and dimensions for convenience
        data = ff["PRIMARY"].data
        header = ff["PRIMARY"].header
        (n_kind, n_order, n_px) = data.shape

        # Save relevant header information to dict
        for keyword in HEADERS_TO_EXTRACT:
            data_dict[keyword] = header[keyword]

        # ---------------------------------------------------------------------
        # Reconstruct wavelength scale
        # ---------------------------------------------------------------------
        # *Annoyingly* MIKE saves the wavelength information from each order
        # as a concatenated and wrapped set of miscellaneous strings right at
        # the end of the header. These are labelled 'WAT2_0XX' where these
        # contain 'wtype=multispec specX = "<>"'. There is one specX = "<>" for
        # each echelle order, where an example <> is: 
        # 
        #   '37 1 0 9148.4828539850878 0.065204998854824225 4096 0.000000 \
        #    2022.391862 2030.116138'
        #
        # Critically, however, when these are concatenated together and broken
        # up over multiple headers, trailing spaces are removed, which means
        # we need to perform an awkward reconstruction of each order while
        # knowing something about the structure we expect. Fortunately, we ony
        # care about elements 0 (order number), 3 (min wavelength), and 4 
        # (delta wavelength), so we just have to make sure up to there is
        # correct:

        # Check how many misc string header entries we need to parse
        for wi in range(1, 1000):
            if "WAT2_{:03.0f}".format(wi) in ff["PRIMARY"].header:
                continue
            else:
                max_i = wi
                break
        
        # Extract and combine all misc headers
        order_info_str_comb = "".join(
                [header["WAT2_{:03.0f}".format(i)] for i in range(1, max_i)])
        
        # By splitting on a " character we separate the orders
        order_info_str_split = order_info_str_comb.split('"')

        # And by taking every second string we take just the numerical info
        raw_strings = order_info_str_split[1::2]

        # Initialise output wavelength scale info
        orders = np.zeros(n_order).astype(int)
        wave_min = np.zeros(n_order)
        wave_max = np.zeros(n_order)
        wave_delta = np.zeros(n_order)
        waves = np.zeros((n_order, n_px))

        # These are the locations we want to insert spaces in case they've
        # been dropped.
        SPACE_LOC = [2, 4, 6]

        # Initialise output set of 'repaired' order information
        repaired_strings = []

        # Loop over and repair all orders
        for order_i, order_str in enumerate(raw_strings):
            # First thing we need to do is 'repair' the string, and the easiest
            # way to insert the missing spaces is to convert the string to a
            # list which is mutable.
            order_str_list = list(order_str)

            # Most orders are two digit numbers, but for orders that are three
            # digits we need to increment by 1 when checking the location of
            # spaces. We check this by looking for numbers starting with 10.
            if order_str[:2] == "10":
                xi = 1
            else:
                xi = 0

            # Insert spaces where required
            for space_i in SPACE_LOC:
                if order_str_list[space_i+xi] != " ":
                    order_str_list.insert(space_i, " ")

            # Check that there is a space before delta wavelength. We'll use
            # the second decimal point (which belongs to delta wave) as a ref.
            dp_ii = np.argwhere(np.array(order_str_list) == ".")
            space_i = dp_ii[1,0] - 2

            if order_str_list[space_i] != " ":
                order_str_list.insert(space_i+1, " ")

            # Convert back to string format
            order_str_new = "".join(order_str_list)

            # Save for output if needed
            repaired_strings.append(order_str_new)

            # Now finally split on spaces to get access to the numberical info
            split_str = order_str_new.split(" ")

            # Extract order and wavelength scale information
            orders[order_i] = int(split_str[0])
            wave_min[order_i] = float(split_str[3])
            wave_delta[order_i] = float(split_str[4])
            wave_max[order_i] =  wave_min[order_i] + n_px*wave_delta[order_i]

            wave = np.arange(
                wave_min[order_i], wave_max[order_i], wave_delta[order_i])
            
            # HACK: it's possible the wavelength scale is larger than 4096 px
            # (presumably due to floating point uncertainties?) so we only want
            # to take up to 4096.
            if len(wave) > 4096:
                #print(
                # "warning, min + max + delta give {} px".format(len(wave)))
                wave = wave[:4096]

            waves[order_i] = wave

        # Store arrays
        data_dict["orders"] = orders
        data_dict["wave"] = waves
        data_dict["spec_sky"] = data[0]
        data_dict["spec_star"] = data[1]
        data_dict["sigma_star"] = data[2]
        data_dict["snr_star"] = data[3]
        data_dict["spec_lamp"] = data[4]
        data_dict["spec_flat"] = data[5]
        data_dict["spec_star_norm"] = data[6]

    return data_dict


def load_and_combine_single_star(filenames, plot_folder, normalise_by_flat):
    """Load in a list of fits files corresponding to reduced MIKE data of a
    *single* arm, and co-add the spectra.

    Parameters
    ----------
    filenames: str list
        List of filenames of reduced MIKE fits files to open.

    plot_folder: str, default: ''
        Base folder to save diagnostic reduction plots to.

    Returns
    -------
    all_exp_dict: dict
        Dictionary representation of MIKE fits files (either for the blue or
        red arm), with keys for relevant header keywords, the wavelength scale,
        and each of the seven (sky, obj, sigma, snr, lamp, flat, obj / flat)
        data-cubes. Each key corresponds to a list of length the number of
        exposures.

    obj_dict: dict
        As above, but for the co-added spectra. Spectra and exposure times are
        co-added, uncertainties added in quadrature, and all other header
        values are taken from the median exposure in time.
    """
    # -------------------------------------------------------------------------
    # Read in all separate exposures provided
    # -------------------------------------------------------------------------
    all_exp_dict = {}

    # Read data from all exposures
    for fn_i, fn in enumerate(filenames):
        exp_dict = load_single_mike_fits(fn)

        # Diagnostic Plot
        plot_fn = "{}_{}_{}_{}.pdf".format(
            exp_dict["UT-DATE"],
            exp_dict["OBJECT"],
            exp_dict["INSTRUME"].split("-")[-1],
            fn_i)
        
        plot_title = "{}, {}, {}, SNR~{:0.0f}, # order = {}, exp {}/{}".format(
            exp_dict["DATE-OBS"],
            exp_dict["OBJECT"],
            exp_dict["INSTRUME"].split("-")[-1],
            np.nanmean(exp_dict["snr_star"]),
            exp_dict["wave"].shape[0],
            fn_i+1,
            len(filenames),)

        visualise_mike_spectra(
            exp_dict, 
            plot_folder,
            plot_fn,
            plot_title,
            title_colour="r" if "Red" in plot_title else "b",)

        # Save everything to one dictionary for all exposures
        for key in exp_dict.keys():
            if key in all_exp_dict.keys():
                all_exp_dict[key].append(exp_dict[key])
            else:
                all_exp_dict[key] = [exp_dict[key]]

    # -------------------------------------------------------------------------
    # Input checking
    # -------------------------------------------------------------------------
    # Ensure that these headers are the same for all exposures
    CONSTANT_HEADERS = ["SITENAME", "SITEALT", "SITELAT", "SITELONG", 
        "TELESCOP", "OBSERVER", "INSTRUME", "OBJECT", "BINNING", "SPEED",
        "SLITSIZE",]

    for header in CONSTANT_HEADERS:
        if len(set(all_exp_dict[header])) != 1:
            raise Exception(
                "{} is not consistent for all exposures.".format(header))

    # Ensure we have a consistent number of orders
    unify_wavelength_scale_per_exp(all_exp_dict)

    # -------------------------------------------------------------------------
    # Flat field normalisation
    # -------------------------------------------------------------------------
    # [Optional ]Normalise all stellar fluxes + sigmas by the flat field
    if normalise_by_flat:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning) 
            # Normalise flat, setting zero sensitivity pixels to nan to avoid inf
            flat = np.stack(all_exp_dict["spec_flat"])
            flat[flat == 0] = np.nan

            flat_max = np.nanmax(flat, axis=2)
            flat /= np.broadcast_to(flat_max[:,:,None], flat.shape)

            all_exp_dict["spec_star_norm"] = \
                np.stack(all_exp_dict["spec_star"]) / flat
            all_exp_dict["sigma_star_norm"] = \
                np.stack(all_exp_dict["sigma_star"]) / flat
            
    else:
        all_exp_dict["spec_star_norm"] = np.stack(all_exp_dict["spec_star"])
        all_exp_dict["sigma_star_norm"] = np.stack(all_exp_dict["sigma_star"])

    # -------------------------------------------------------------------------
    # Computing barycentric corrections for all exposures
    # -------------------------------------------------------------------------
    # Compute barycentric velocity
    bcors = compute_barycentric_correction(
        site_lat=all_exp_dict["SITELAT"][0],
        site_long=all_exp_dict["SITELONG"][0],
        site_alt=all_exp_dict["SITEALT"][0],
        ras=all_exp_dict["RA-D"],
        decs=all_exp_dict["DEC-D"],
        times=all_exp_dict["DATE-OBS"],
        exps=all_exp_dict["EXPTIME"],)
    
    all_exp_dict["BCOR"] = bcors

    # -------------------------------------------------------------------------
    # Interpolate spectra to new wavelength scale, co-add
    # -------------------------------------------------------------------------
    n_exp = len(all_exp_dict["EXPTIME"])

    # Select reference exposure, from which we adopt the wavelength scale
    ref_exp = np.median(np.arange(n_exp)).astype(int)

    bcor_rel = bcors - bcors[ref_exp]
    wave_new = all_exp_dict["wave"][ref_exp]

    all_exp_dict["spec_star_norm_regridded"] = []
    all_exp_dict["sigma_star_norm_regridded"] = []
    
    # Intialise output dict
    obj_dict = {"wave":wave_new}

    for exp_i in range(n_exp):
        # Interpolate fluxes
        spec_new = interpolate_wavelength_scale(
            wave_old=all_exp_dict["wave"][exp_i],
            spec_old=all_exp_dict["spec_star_norm"][exp_i],
            wave_new=wave_new,
            rv_shift=bcor_rel[exp_i],)
        
        all_exp_dict["spec_star_norm_regridded"].append(spec_new)

        # Interpolate uncertainties
        sigma_new = interpolate_wavelength_scale(
            wave_old=all_exp_dict["wave"][exp_i],
            spec_old=all_exp_dict["sigma_star_norm"][exp_i],
            wave_new=wave_new,
            rv_shift=bcor_rel[exp_i],)
        
        all_exp_dict["sigma_star_norm_regridded"].append(sigma_new)

    # Convert to numpy array
    obj_dict["spec_star_norm"] = \
        np.stack(all_exp_dict["spec_star_norm_regridded"])
    obj_dict["sigma_star_norm"] = \
        np.stack(all_exp_dict["sigma_star_norm_regridded"])

    # Create a bad px mask?
    pass

    # Co-add. Currently we average all exposures, which for the uncertainties
    # means adding in quadrature and dividing by the number of exposures. This
    # doesn't preserve 'total counts' in the same way taking one monolithic
    # exposure would, but this *should* preserve SNR, which is more important.
    obj_dict["spec_star_norm"] = \
        np.nansum(obj_dict["spec_star_norm"], axis=0) / n_exp
    obj_dict["sigma_star_norm"] = \
        np.nansum(obj_dict["sigma_star_norm"]**2, axis=0)**0.5 / n_exp

    # -------------------------------------------------------------------------
    # Select and combine other header info
    # -------------------------------------------------------------------------
    # For all headers we simply take the value from the reference exposure,
    # with the sole exception of the exppsure time, which we sum.
    for header in HEADERS_TO_EXTRACT:
        if header != "EXPTIME":
            obj_dict[header] = all_exp_dict[header][ref_exp]
        else:
            obj_dict[header] = np.sum(all_exp_dict[header])

    # Add the reference barycentric velocity
    obj_dict["BCOR"] = all_exp_dict["BCOR"][ref_exp]

    # Finally add the order numbers
    obj_dict["orders"] = all_exp_dict["orders"][ref_exp]

    return all_exp_dict, obj_dict


def load_all_mike_fits(
    spectra_folder="spectra/",
    id_crossmatch_fn="data/mike_id_cross_match.tsv",
    plot_folder="",
    normalise_by_flat=True,):
    """Load in all fits cubes containing 1D spectra to extract both the spectra
    and key details of the observations for many targets or nights of data.

    Parameters
    ----------
    spectra_folder: string
        Root directory of nightly extracted 1D fits cubes.

    id_crossmatch_fn: default: 'data/mike_id_cross_match.tsv'

    plot_folder: str, default: ''
        Base folder to save diagnostic reduction plots to.

    Returns
    -------
    blue_dict, red_dict: dict
        Dictionary representation of MIKE fits files (one for each spectrograph
        arm), with (source_id, date) keys linked to a dictionary containing
        relevant header keywords, order numbers, the wavelength scale, fluxes,
        and uncertanties.
    """
    # Import ID crossmatch file
    id_cm_df = pd.read_csv(
        filepath_or_buffer=id_crossmatch_fn,
        sep="\t",
        index_col="obj",
        dtype={"source_id":str},)

    # Get all fits filenames
    spectra_wildcard = os.path.join(spectra_folder, "*.fits")
    fits_files = glob.glob(spectra_wildcard)
    fits_files.sort()
    fits_files = np.array(fits_files)

    n_spec = len(fits_files)

    # Group fits files per star. Although the object name is in the filename,
    # it's not reliably formatted, so the simple approach is to just grab the
    # information straight from the fits headers instead.
    obj_names = np.array([fits.getval(fn, "OBJECT") for fn in fits_files])
    det_arms = np.array([fits.getval(fn, "INSTRUME") for fn in fits_files])
    exp_times = np.array([fits.getval(fn, "EXPTIME") for fn in fits_files])
    n_loops = np.array([fits.getval(fn, "NLOOPS") for fn in fits_files])
    date_obs = np.array([fits.getval(fn, "DATE-OBS") for fn in fits_files])
    ut_dates = np.array([fits.getval(fn, "UT-DATE") for fn in fits_files])

    # Round exposure times
    exp_times = np.round(exp_times, decimals=0).astype(int)

    # Create a boolean mask for the blue or red detectors
    is_red  = np.array(det_arms) == "MIKE-Red"

    # Grab unique object names
    unique_objs = set(obj_names)
    
    # Initialise objects to store all of this
    blue_dict = {}
    red_dict = {}

    # Loop over all objects
    for obj in unique_objs:
        # Crossmatch ID and replace with Gaia DR3 source id
        if obj not in id_cm_df.index:
            print("\nMissing crossmatch info for {}".format(obj))
            continue

        sid = id_cm_df.loc[obj]["source_id"]
        kind = id_cm_df.loc[obj]["kind"]
        is_spphot = id_cm_df.loc[obj]["is_sphot"]

        # Also loop over dates observed
        ut_dates_obj = set(ut_dates[obj_names == obj])
        for ut_date in ut_dates_obj:
            print("\n{} ({})...".format(obj, ut_date))
            # Read in blue data
            is_obj_blue_arm_date = np.all(np.stack(
                (obj_names == obj, ~is_red, ut_dates == ut_date)), axis=0)

            if np.sum(is_obj_blue_arm_date) > 0:
                obj_blue_fns = fits_files[is_obj_blue_arm_date]
                print("\n\tBlue:\n\t", "\n\t".join(obj_blue_fns), sep="",)
                exp_dict_b, obj_dict_b = load_and_combine_single_star(
                    obj_blue_fns, plot_folder, normalise_by_flat)
                
                # Store target classification that we get from the crossmatch
                obj_dict_b["kind"] = kind
                obj_dict_b["is_spphot"] = is_spphot

            else:
                exp_dict_b = None
                obj_dict_b = None

            # Read in red data
            is_obj_red_arm_date = np.all(np.stack(
                (obj_names == obj, is_red, ut_dates == ut_date)), axis=0)

            if np.sum(is_obj_red_arm_date) > 0:
                obj_red_fns = fits_files[is_obj_red_arm_date]
                print("\n\tRed:\n\t", "\n\t".join(obj_red_fns), sep="",)
                exp_dict_r, obj_dict_r = load_and_combine_single_star(
                    obj_red_fns, plot_folder, normalise_by_flat)
                
                # Store target classification that we get from the crossmatch
                obj_dict_r["kind"] = kind
                obj_dict_r["is_spphot"] = is_spphot

            else:
                exp_dict_r = None
                obj_dict_r = None

            # Star extracted combined dicts for each exposure (distinguishing
            # per night) to allow for duplicate targets.
            blue_dict[(sid, ut_date)] = obj_dict_b
            red_dict[(sid, ut_date)] = obj_dict_r

    return blue_dict, red_dict


def collate_mike_obs(blue_dict, red_dict):
    """Takes extraced dictionaries of each observation, and compiles these into
    one unified pandas DataFrame and sets of arrays.

    Parameters
    ----------
    blue_dict, red_dict: dict
        Dictionary representation of MIKE fits files (one for each spectrograph
        arm), with (source_id, date) keys linked to a dictionary containing
        relevant header keywords, order numbers, the wavelength scale, fluxes,
        and uncertanties.

    Returns
    -------
    obs_dict: dict
        Dictionary containing the compiled MIKE data, with the following keys:
            'obs_info' - pandas DataFrame with info from fits headers
            'orders_b' - blue echelle orders, shape [n_order]
            'wave_b'   - blue wavelength scales, shape [n_star, n_order, n_px]
            'spec_b'   - blue spectra, shape [n_star, n_order, n_px]
            'sigma_b'  - blue uncertainties, shape [n_star, n_order, n_px]
            'orders_r' - red echelle orders, shape [n_order]
            'wave_r'   - red wavelength scales, shape [n_star, n_order, n_px]
            'spec_r'   - red spectra, shape [n_star, n_order, n_px]
            'sigma_r'  - red uncertainties, shape [n_star, n_order, n_px]
    """
    # Get a list of the unique (obj, date) pairs
    obj_all = set(list(blue_dict.keys()) + list(red_dict.keys()))

    n_obj = len(obj_all)

    # -------------------------------------------------------------------------
    # Create wave, spec, and sigma arrays
    # -------------------------------------------------------------------------
    # Unify wavelength scales by adding dummy orders
    (n_order_b, n_px_b) = unify_wavelength_scales_global(blue_dict, True)
    (n_order_r, n_px_r) = unify_wavelength_scales_global(red_dict, True)

    #(n_order_b, n_px_b) = blue_dict[blue_dict.keys()]

    # Store all spectra into 2x 3D arrays of shape [n_obs, n_rder, n_px]
    wave_b = np.full((n_obj, n_order_b, n_px_b), np.nan)
    wave_r = np.full((n_obj, n_order_r, n_px_r), np.nan)

    spec_b = np.full((n_obj, n_order_b, n_px_b), np.nan)
    spec_r = np.full((n_obj, n_order_r, n_px_r), np.nan)

    sigma_b = np.full((n_obj, n_order_b, n_px_b), np.nan)
    sigma_r = np.full((n_obj, n_order_r, n_px_r), np.nan)

    for obj_i, obj in enumerate(obj_all):
        if blue_dict[obj] is not None:
            wave_b[obj_i] = blue_dict[obj]["wave"]
            spec_b[obj_i] = blue_dict[obj]["spec_star_norm"]
            sigma_b[obj_i] = blue_dict[obj]["sigma_star_norm"]
            orders_b = blue_dict[obj]["orders"]

        if red_dict[obj] is not None:
            wave_r[obj_i] = red_dict[obj]["wave"]
            spec_r[obj_i] = red_dict[obj]["spec_star_norm"]
            sigma_r[obj_i] = red_dict[obj]["sigma_star_norm"]
            orders_r = red_dict[obj]["orders"]

    # -------------------------------------------------------------------------
    # Create DataFrame by combining B/R info
    # -------------------------------------------------------------------------
    # Column definitions, one set is common between arms, the othar are per-arm
    cols_common = ["source_id", "kind", "is_spphot", "object", "observer",
         "ut_date", "ut_start",  "lc_start", "airmass", "slit_size", "bcor",] 
    
    cols_base_br = ["has", "exp_time", "n_loops", "binning", "speed", "snr",]

    cols_br = []
    for col_base in cols_base_br:
        cols_br.append(col_base + "_b")
        cols_br.append(col_base + "_r")

    # Initialise our dataframe with relevant columns and NaN values
    cols_all = cols_common + cols_br

    n_cols = len(cols_all)
    data = np.full((n_obj, n_cols), np.nan)

    obs_info = pd.DataFrame(data=data, columns=cols_all)

    # Store all header information in a pandas dataframe
    for obj_i, obj in enumerate(obj_all):
        # Store obj key values
        obs_info.loc[obj_i, "source_id"] = obj[0]

        # Loop over both arms and populate dataframe
        for arm, arm_dict in zip(("b", "r"), (blue_dict, red_dict)):
            if arm_dict[obj] is None:
                # Arm specific params
                obs_info.loc[obj_i, "has_" + arm] = False
                obs_info.loc[obj_i, "exp_time_" + arm] = np.nan
                obs_info.loc[obj_i, "n_loops_" + arm] = np.nan
                obs_info.loc[obj_i, "binning_" + arm] = ""
                obs_info.loc[obj_i, "speed_" + arm] = ""
                continue
            
            # Constant params
            obs_info.loc[obj_i, "kind"] = arm_dict[obj]["kind"]
            obs_info.loc[obj_i, "is_spphot"] = arm_dict[obj]["is_spphot"]
            obs_info.loc[obj_i, "object"] = arm_dict[obj]["OBJECT"]
            obs_info.loc[obj_i, "observer"] = arm_dict[obj]["OBSERVER"]
            obs_info.loc[obj_i, "ut_date"] = arm_dict[obj]["UT-DATE"]
            obs_info.loc[obj_i, "ut_start"] = arm_dict[obj]["UT-START"]
            obs_info.loc[obj_i, "lc_start"] = arm_dict[obj]["LC-START"]
            obs_info.loc[obj_i, "airmass"] = arm_dict[obj]["AIRMASS"]
            obs_info.loc[obj_i, "slit_size"] = arm_dict[obj]["SLITSIZE"]
            obs_info.loc[obj_i, "bcor"] = arm_dict[obj]["BCOR"]

            # Arm specfic params
            obs_info.loc[obj_i, "has_" + arm] = True
            obs_info.loc[obj_i, "exp_time_" + arm] = arm_dict[obj]["EXPTIME"]
            obs_info.loc[obj_i, "n_loops_" + arm] = arm_dict[obj]["NLOOPS"]
            obs_info.loc[obj_i, "binning_" + arm] = arm_dict[obj]["SLITSIZE"]
            obs_info.loc[obj_i, "speed_" + arm] = arm_dict[obj]["SPEED"]

            # Ignore divide by zero/NaN errors when calculating SNR
            with warnings.catch_warnings():
                msg1 = "divide by zero encountered in true_divide"
                msg2 = "invalid value encountered in true_divide"
                warnings.filterwarnings(action='ignore', message=msg1)
                warnings.filterwarnings(action='ignore', message=msg2)

                snr = np.round(np.nanmedian(
                    arm_dict[obj]["spec_star_norm"]
                    / arm_dict[obj]["sigma_star_norm"]))
            
                obs_info.loc[obj_i, "snr_" + arm] = snr

    # Column unit definition to avoid astropy complaining later
    obs_info["has_b"] = obs_info["has_b"].values.astype(bool)
    obs_info["has_r"] = obs_info["has_r"].values.astype(bool)
    obs_info["is_spphot"] = obs_info["is_spphot"].values.astype(bool)

    obs_dict = {
        "obs_info":obs_info,
        "orders_b":orders_b,
        "wave_b":wave_b,
        "spec_b":spec_b,
        "sigma_b":sigma_r,
        "orders_r":orders_r,
        "wave_r":wave_r,
        "spec_r":spec_r,
        "sigma_r":sigma_r,}

    return obs_dict


def visualise_mike_spectra(
    data_dict,
    plot_folder,
    plot_fn,
    plot_title,
    title_colour,):
    """Visualises extracted MIKE spectra, plotting panels for sky, star, sigma,
    SNR, lamp, flat, and normalised star.

    Parameters
    ----------
    data_dict: dict
        Dictionary representation of MIKE fits file, with keys for relevant
        header keywords, the wavelength scale, and each of the seven (sky, obj,
        sigma, snr, lamp, flat, obj / flat) data-cubes.

    plot_folder: str
        Folder to save plots to.

    plot_fn: str
        Filename of the saved plot.

    plot_title: str
        Title of the plot.

    title_colour: str
        Colour of the title text.
    """
    waves = data_dict["wave"]
    spec_sky = data_dict["spec_sky"]
    spec_star = data_dict["spec_star"]
    sigma_star = data_dict["sigma_star"]
    snr_star = data_dict["snr_star"]
    spec_lamp = data_dict["spec_lamp"]
    spec_flat = data_dict["spec_flat"]
    spec_star_norm = data_dict["spec_star_norm"]

    plt.close("all")
    fig, axes = plt.subplots(nrows=7, sharex=True, figsize=(16, 10))

    fig.subplots_adjust(
        left=0.075,
        bottom=0.075,
        right=0.98,
        top=0.975,
        hspace=0.000,
        wspace=0.001)

    axes[0].plot(waves.ravel(), spec_sky.ravel(), linewidth=0.2)
    axes[1].plot(waves.ravel(), spec_star.ravel(), linewidth=0.2)
    axes[2].plot(waves.ravel(), sigma_star.ravel(), linewidth=0.2)
    axes[3].plot(waves.ravel(), snr_star.ravel(), linewidth=0.2)
    axes[4].plot(waves.ravel(), spec_lamp.ravel(), linewidth=0.2)
    axes[5].plot(waves.ravel(), spec_flat.ravel(), linewidth=0.2)
    axes[6].plot(waves.ravel(), spec_star_norm.ravel(), linewidth=0.2)

    labels = ["Sky", "Star", "Sigma", "SNR", "Lamp", "Flat", "Star (Norm)"]

    for label_i, label in enumerate(labels):
        axes[label_i].set_ylabel(label)

    axes[-1].set_xlabel("Wavelength (Å)")
    axes[0].set_title(plot_title, c=title_colour)

    axes[-1].xaxis.set_major_locator(plticker.MultipleLocator(base=200))
    axes[-1].xaxis.set_minor_locator(plticker.MultipleLocator(base=100))

    # Check save folder and save
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plt.savefig(os.path.join(plot_folder, plot_fn))


def calc_flux_correction_resid(
    params,
    wave_obs_2D,
    spec_obs_2D,
    sigma_obs_2D,
    tau_H2O_2D,
    tau_O2_2D,
    spec_fluxed_2D,
    poly_order,
    optimise_order_overlap,
    do_tau_scaling_term_fit,
    scale_H2O=None,
    scale_O2=None,):
    """Calculates the residuals between a low-resolution flux calibrated
    spectrum and a high-resolution spectrum corrected for the atmospheric and
    instrumental transfer function.

    Parameters
    ----------
    params: float array
        Contains (scale_H2O, scale_O2, poly_coeff) where scale_H2O and scale_O2
        are scaling terms in the exponent for the strength of the telluric
        absorption, and poly_coeff is a *flattened* and appended 2D array of
        (original) shape [n_order, poly_order] containing the polynomial
        coefficients for each spectral order for use with
        numpy.polynomial.polynomial.Polynomial.

    wave_obs_2D, spec_obs_2D, sigma_obs_2D: 2D float array
        Observed spectra of shape [n_order, n_px].

    tau_H2O_2D, tau_O2_2D: 2D float array
        Telluric optical depths for H2O and O2 interpolated to be on the same
        wavelength scale as wave_obs_2D, of shape [m_order, n_px].

    pec_fluxed_2D: 1D float array
        Low-resolution fluxed spectra of our target star interpolated to be on
        the same wavelength scale as wave_obs_2D, of shape [m_order, n_px].

    poly_order: int
        Polynomial order to be fit as the transfer function for each order.

    optimise_order_overlap: boolean
        If true, we also compute residuals for the overlapping regions of
        orders to ensure the resulting fluxed spectrum is smooth across orders.

    do_tau_scaling_term_fit: boolean
        Whether to fit for the tau scaling term

    TODO

    Returns
    -------
    resid_vect: float
        Sum of the uncertainty weighted residuals vector.
    """
    # Grab dimensions for convenience
    (n_order, n_px) = wave_obs_2D.shape

    params = np.array(params)

    # Unpack params
    if do_tau_scaling_term_fit:
        scale_H2O = params[0]
        scale_O2 = params[1]
        poly_coef = params[2:].reshape((n_order, poly_order))

    # No scaling terms to fit, the entire parameter array is just poly coeff
    else:
        poly_coef = params.reshape((n_order, poly_order))

    # Generate 'fluxed' science spectrum by 'correcting' for the tellurics
    # and scaling by the transfer function polynomial
    spec_obs_2D_fluxed, sigma_obs_2D_fluxed = \
        correct_tellurics_and_apply_transfer_function(
            wave_obs_2D=wave_obs_2D,
            spec_obs_2D=spec_obs_2D,
            sigma_obs_2D=sigma_obs_2D,
            tau_H2O_2D=tau_H2O_2D,
            tau_O2_2D=tau_O2_2D,
            scale_H2O=scale_H2O,
            scale_O2=scale_O2,
            poly_coef=poly_coef,)

    # -------------------------------------------------------------------------
    # Compute residuals
    # -------------------------------------------------------------------------
    # Compute order-by-order residuals when comparing to the fluxed spectrum
    fluxed_resid = (spec_fluxed_2D - spec_obs_2D_fluxed)**2 
                  #/ sigma_obs_2D_cleaned**2)
    
    #fluxed_resid[do_interp_mask] = 0

    fluxed_resid = fluxed_resid.flatten()

    # [Optional ]Also compute residuals for overlapping sections of each order
    if optimise_order_overlap:
        for order_i in range(n_order-1):
            # Grab wavelength scale for this order and the one above
            wave_1 = wave_obs_2D[order_i]
            wave_2 = wave_obs_2D[order_i+1]

            spec_1 = spec_obs_2D_fluxed[order_i]
            spec_2 = spec_obs_2D_fluxed[order_i+1]

            sigma_1 = sigma_obs_2D_fluxed[order_i]
            sigma_2 = sigma_obs_2D_fluxed[order_i+1]

            is_nan_spec_2 = np.isnan(spec_2)

            # Compute which wavelengths are overlapping
            overlap_mask_1 = np.logical_and(
                wave_1 > np.min(wave_2), wave_1 < np.max(wave_2))

            overlap_mask_2 = np.logical_and(
                wave_2 > np.min(wave_1), wave_2 < np.max(wave_1))

            # Interpolate onto the same wavelength scale
            interp_mask = np.logical_and(overlap_mask_2, ~is_nan_spec_2)
            
            interp_spec = interp1d(
                x=wave_2[interp_mask],
                y=spec_2[interp_mask],
                bounds_error=False,
                fill_value=np.nan,
                kind="cubic",)
            
            interp_sigma = interp1d(
                x=wave_2[interp_mask],
                y=sigma_2[interp_mask],
                bounds_error=False,
                fill_value=np.nan,
                kind="cubic",)
            
            spec_2_interp = interp_spec(wave_1[overlap_mask_1])
            sigma_2_interp = interp_sigma(wave_1[overlap_mask_1])

            # We don't weight by the uncertainties, as that downweights the
            # order edges which makes the overlap between adjacent orders
            # worse.
            overlap_resid = (spec_1[overlap_mask_1] - spec_2_interp)**2 
                            #/ (sigma_1[overlap_mask_1]**2 + sigma_2_interp**2))

            is_nan = np.isnan(overlap_resid)
            overlap_resid[is_nan] = 0

    else:
        overlap_resid = []

    # Combine resid
    combined_resid = np.concatenate((fluxed_resid, overlap_resid))

    # -------
    if np.sum(~np.isfinite(combined_resid)) != 0:
        import pdb; pdb.set_trace()
        raise ValueError("Non-finite residuals!")
    
    r1 = np.sum(fluxed_resid)
    r2 = np.sum(overlap_resid)

    print("r1 = {}, r2 = {}\nH2O = {}\nO2 = {}\ncoeff = {}".format(
        r1, r2, scale_H2O, scale_O2, poly_coef))
    print("-"*80)

    return combined_resid


def correct_tellurics_and_apply_transfer_function(
    wave_obs_2D,
    spec_obs_2D,
    sigma_obs_2D,
    tau_H2O_2D,
    tau_O2_2D,
    scale_H2O,
    scale_O2,
    poly_coef,):
    """Calculates the residuals between a low-resolution flux calibrated
    spectrum and a high-resolution spectrum corrected for the atmospheric and
    instrumental transfer function.

    Parameters
    ----------
    wave_obs_2D, spec_obs_2D, sigma_obs_2D: 2D float array
        Observed spectra of shape [n_order, n_px].

    tau_H2O_2D, tau_O2_2D: 2D float array
        Telluric optical depths for H2O and O2 interpolated to be on the same
        wavelength scale as wave_obs_2D, of shape [m_order, n_px].

    scale_H2O, scale_O2: float
        Exponent scaling terms for the strength of the telluric absorption for
        H2O and O2 respectively. A value of 1.0 means the transmission function
        remains unchanged.
        
    poly_coeff: float array
        2D array of shape [n_order, poly_order] containing the polynomial 
        coefficients for each spectral order for use with
        numpy.polynomial.polynomial.Polynomial.

    Returns
    -------
    resid_vect: float
        Sum of the uncertainty weighted residuals vector.
    """
    # Grab dimensions for convenience
    (n_order, n_px) = wave_obs_2D.shape

    # Initialise output vectors
    spec_obs_2D_fluxed = spec_obs_2D.copy()
    sigma_obs_2D_fluxed = sigma_obs_2D.copy()

    # -------------------------------------------------------------------------
    # Calculate + "Correct" telluric transmission
    # -------------------------------------------------------------------------
    # Calculate teluric transmission
    trans_H2O_2D = np.exp(-scale_H2O * tau_H2O_2D)
    trans_O2_2D = np.exp(-scale_O2 * tau_O2_2D)

    # Set any NaNs in the transmission to 1.0
    missing_transmission = np.logical_or(
        np.isnan(trans_H2O_2D), np.isnan(trans_O2_2D))
    trans_H2O_2D[missing_transmission] = 1.0
    trans_O2_2D[missing_transmission] = 1.0

    spec_obs_2D_fluxed /= trans_H2O_2D * trans_O2_2D
    sigma_obs_2D_fluxed /= trans_H2O_2D * trans_O2_2D

    # -------------------------------------------------------------------------
    # Apply Polynomial
    # -------------------------------------------------------------------------
    for order_i in range(n_order):
        tf_poly = Polynomial(poly_coef[order_i])
        tf = tf_poly(wave_obs_2D[order_i])

        spec_obs_2D_fluxed[order_i] *= tf
        sigma_obs_2D_fluxed[order_i] *= tf

    # Done, return
    return spec_obs_2D_fluxed, sigma_obs_2D_fluxed


def fit_atmospheric_transmission(
    wave_obs_2D,
    spec_obs_2D,
    sigma_obs_2D,
    wave_telluric,
    trans_H2O_telluric,
    trans_O2_telluric,
    wave_fluxed,
    spec_fluxed,
    wave_synth,
    spec_synth,
    airmass,
    do_convolution=False,
    resolving_power_during_fit=1150,
    wavelength_subsample_fac=20,
    poly_order=4,
    optimise_order_overlap=False,
    extinction_curve_fn="data/paranal_extinction_patat2011.tsv",):
    """Function to fit polynomial transfer functions to each spectral order to
    flux calibrate an observed MIKE spectrum, with reference to a low-
    resolution fluxed spectrum of the target, a continuum normalised synthetic
    stellar spectrum, telluric transmission for H2O and O2, and atmospheric
    extinction at the observatory.

    Following the approach of pywifes.derive_wifes_calibration() from:
    https://github.com/PyWiFeS/pywifes/blob/main/src/pywifes/wifes_calib.py
    
    Parameters
    ----------
    wave_obs_2D, spec_obs_2D, sigma_obs_2D: 2D float array
        Observed spectra of shape [n_order, n_px].

    wave_telluric, trans_H2O_telluric, trans_O2_telluric: 1D float array
        Telluric wavelength scale and transmission for H2O and O2.

    wave_fluxed, spec_fluxed: 1D float array
        Low-resolution fluxed spectra of our target star.

    wave_synth, spec_synth: 1D float array
        Synthetic *continuum normalised* spectrum of our target star at the
        same resolution as our observations.

    airmass: float
        Airmass of the target.

    do_convolution: boolean, default: False
        Whether to convolve all spectra to a lower resolving power before
        fitting. This has the effect of better matching our MIKE spectra to the
        CALSPEC fluxed spectra, and smoothing out deep lines likely to either
        cause residuals or complications if just masking them.

    resolving_power_during_fit: int, default: 1150
        New resolving power if convolving to a lower resolution. The default
        value corresponds to 40x lower resolution from the MIKE red arm which
        very roughly (by eye) matches the CALSPEC spectra.

    wavelength_subsample_fac: int, default: 20
        Factor to subsample the wavelength scale by after broadening.

    poly_order: int, default: 4
        Polynomial order to be fit as the transfer function for each order.

    optimise_order_overlap: boolean, default: False
        If true, we also compute residuals for the overlapping regions of
        orders to ensure the resulting fluxed spectrum is smooth across orders.

    extinction_curve_fn: str, default: 'data/paranal_extinction_patat2011.tsv'
        Path to extinction at observatory.

    Returns
    -------
    fit_dict: dictionary
        Dictionary as output from from scipy.optimize.least_squares, with added
        keys: ['scale_H2O', 'scale_O2', 'poly_coef', 'wave_obs_2D',
        'spec_obs_2D', 'spec_synth_2D', 'spec_fluxed_2D', 'extinction_2D',
        'tau_H2O_2D', 'tau_O2_2D'].
    """
    # Grab dimensions for convenience
    (n_order, n_px) = wave_obs_2D.shape

    # Import the extinction curve
    ext_pd = pd.read_csv(extinction_curve_fn, delimiter="\t")

    # -------------------------------------------------------------------------
    # Masking
    # -------------------------------------------------------------------------
    # It's possible to receive spectra where entire orders are NaNs. If so, we
    # don't want to pass these on to the fitting function, but we do want to
    # create dummy coefficients so that everything is consistent in shape.
    useful_orders = np.logical_and(
        np.sum(np.isnan(spec_obs_2D), axis=1) < n_px,
        np.sum(np.isnan(sigma_obs_2D), axis=1) < n_px,)
    n_useful_orders = np.sum(useful_orders)

    # -------------------------------------------------------------------------
    # [Optional] Convolve to a lower resolution
    # -------------------------------------------------------------------------
    if do_convolution:
        # Telluric (H2O)
        trans_H2O_telluric = instrBroadGaussFast(
            wvl=wave_telluric,
            flux=trans_H2O_telluric,
            resolution=resolving_power_during_fit,
            edgeHandling="firstlast",
            maxsig=5,
            equid=True,)
        
        # Telluric (O2)
        trans_O2_telluric = instrBroadGaussFast(
            wvl=wave_telluric,
            flux=trans_O2_telluric,
            resolution=resolving_power_during_fit,
            edgeHandling="firstlast",
            maxsig=5,
            equid=True,)
        
        # Synthetic (MARCS)
        spec_synth = instrBroadGaussFast(
            wvl=wave_synth,
            flux=spec_synth,
            resolution=resolving_power_during_fit,
            edgeHandling="firstlast",
            maxsig=5,
            equid=True,)
        
        # Observed

        # Mask edges. We can't just set these to NaN when smoothing, as the NaN
        # pixels will end up 'corrupting' neighbouring px up to 5 sigma away.
        edge_px = np.logical_or(np.arange(n_px) < 5, np.arange(n_px) > n_px-5)

        wave_obs_2D = wave_obs_2D[:,~edge_px]
        spec_obs_2D = spec_obs_2D[:,~edge_px]
        sigma_obs_2D = sigma_obs_2D[:,~edge_px]

        # Compute new arrays with subsampled wavelength scale
        n_px_ss = wave_obs_2D[:,::wavelength_subsample_fac].shape[1]

        wave_obs_2D_new = np.full((n_order, n_px_ss), np.nan)
        spec_obs_2D_new = np.full((n_order, n_px_ss), np.nan)
        sigma_obs_2D_new = np.full((n_order, n_px_ss), np.nan)

        for order_i in range(n_order):
            if not useful_orders[order_i]:
                continue
            
            # Broaden spectra
            spec_broad = instrBroadGaussFast(
                wvl=wave_obs_2D[order_i],
                flux=spec_obs_2D[order_i],
                resolution=resolving_power_during_fit,
                edgeHandling="firstlast",
                maxsig=5,
                equid=True,)
            
            sigma_broad = instrBroadGaussFast(
                wvl=wave_obs_2D[order_i],
                flux=sigma_obs_2D[order_i],
                resolution=resolving_power_during_fit,
                edgeHandling="firstlast",
                maxsig=5,
                equid=True,)
            
            # Subsample wavelength scale
            interp_spec_broad = interp1d(
                x=wave_obs_2D[order_i],
                y=spec_broad,
                bounds_error=False,
                fill_value=1.0,
                kind="cubic",)
            
            interp_sigma_broad = interp1d(
                x=wave_obs_2D[order_i],
                y=sigma_broad,
                bounds_error=False,
                fill_value=1.0,
                kind="cubic",)

            wave_obs_2D_new[order_i] = \
                wave_obs_2D[order_i,::wavelength_subsample_fac].copy()
            spec_obs_2D_new[order_i] = \
                interp_spec_broad(wave_obs_2D_new[order_i])
            sigma_obs_2D_new[order_i] = \
                interp_sigma_broad(wave_obs_2D_new[order_i])

    else:
        wave_obs_2D_new = wave_obs_2D.copy()
        spec_obs_2D_new = spec_obs_2D.copy()
        sigma_obs_2D_new = sigma_obs_2D.copy()

    # -------------------------------------------------------------------------
    # Regrid all reference spectra to observed 2D wavelength scale
    # -------------------------------------------------------------------------
    # Construct interpolators
    interp_trans_H20 = interp1d(
        x=wave_telluric,
        y=trans_H2O_telluric,
        bounds_error=False,
        fill_value=1.0,
        kind="cubic",)
    
    interp_trans_O2 = interp1d(
        x=wave_telluric,
        y=trans_O2_telluric,
        bounds_error=False,
        fill_value=1.0,
        kind="cubic",)
    
    interp_spec_fluxed = interp1d(
        x=wave_fluxed,
        y=spec_fluxed,
        bounds_error=False,
        fill_value=np.nan,
        kind="cubic",)
    
    interp_spec_synth = interp1d(
        x=wave_synth,
        y=spec_synth,
        bounds_error=False,
        fill_value=np.nan,
        kind="cubic",)

    interp_extinction = interp1d(
        x=ext_pd["wave"].values,
        y=ext_pd["extinction"].values,
        bounds_error=False,
        fill_value=np.nan,
        kind="cubic",)

    tau_H2O_2D = np.full_like(wave_obs_2D_new, np.nan)
    tau_O2_2D = np.full_like(wave_obs_2D_new, np.nan)
    spec_fluxed_2D = np.full_like(wave_obs_2D_new, np.nan)
    spec_synth_2D = np.full_like(wave_obs_2D_new, np.nan)
    extinction_2D = np.full_like(wave_obs_2D_new, np.nan)

    for order_i in range(n_order):
        wave_ith = wave_obs_2D_new[order_i]
        tau_H2O_2D[order_i] = -np.log(interp_trans_H20(wave_ith))
        tau_O2_2D[order_i] = -np.log(interp_trans_O2(wave_ith))
        spec_fluxed_2D[order_i] = interp_spec_fluxed(wave_ith)
        spec_synth_2D[order_i] = interp_spec_synth(wave_ith)
        extinction_2D[order_i] = interp_extinction(wave_ith)

    # Apply airmass-dependent extinction curve to the flux standard to match
    # our observations
    spec_fluxed_2D *= 10 ** (-0.4 * airmass * extinction_2D)

    # -------------------------------------------------------------------------
    # 1) Initial fit (no order overlap or telluric scaling)
    # -------------------------------------------------------------------------
    print("\n\n\n", "-"*80, "\n", "Initial Fit\n", "-"*80, "\n")

    # Special terms for initial fit
    do_tau_scaling_term_fit = False
    optimise_order_overlap_on_initial_fit = False
    scale_H2O = 1.0
    scale_O2 = 1.0

    # Setup the list of parameters to pass to our fitting function
    poly_coeff_init = np.zeros((n_useful_orders, poly_order))
    poly_coeff_init[:,0] = 1

    params_init = poly_coeff_init.flatten()

    # Establish bounds to l
    bounds_low = np.full((n_useful_orders, poly_order), -np.inf).flatten()
    bounds_high = np.full((n_useful_orders, poly_order), np.inf).flatten()

    args = (
        wave_obs_2D_new[useful_orders],
        spec_obs_2D_new[useful_orders],
        sigma_obs_2D_new[useful_orders],
        tau_H2O_2D[useful_orders],
        tau_O2_2D[useful_orders],
        spec_fluxed_2D[useful_orders],
        poly_order,
        optimise_order_overlap_on_initial_fit,
        do_tau_scaling_term_fit,
        scale_H2O,
        scale_O2,)

    # Do fit
    fit_dict = least_squares(
        calc_flux_correction_resid, 
        params_init, 
        jac="3-point",
        args=args,
        bounds=(bounds_low, bounds_high),
        max_nfev=100,)
    
    # Pack variables/results into dict and return
    params = fit_dict["x"]

    poly_coeff_init_fit = params

    # -------------------------------------------------------------------------
    # 2) Final fit (including order overlap and telluric scaling)
    # -------------------------------------------------------------------------
    print("\n\n\n", "-"*80, "\n", "Final Fit\n", "-"*80, "\n")

    do_tau_scaling_term_fit = True
    params_init = [scale_H2O, scale_O2] + list(poly_coeff_init_fit.flatten())

    # Establish bounds to l
    bounds_low = [0, 0] + list(
        np.full((n_useful_orders, poly_order), -np.inf).flatten())
    bounds_high = [2.0, 2.0] + list(
        np.full((n_useful_orders, poly_order), np.inf).flatten())

    args = (
        wave_obs_2D_new[useful_orders],
        spec_obs_2D_new[useful_orders],
        sigma_obs_2D_new[useful_orders],
        tau_H2O_2D[useful_orders],
        tau_O2_2D[useful_orders],
        spec_fluxed_2D[useful_orders],
        poly_order,
        optimise_order_overlap,
        do_tau_scaling_term_fit)

    # Do fit
    fit_dict = least_squares(
        calc_flux_correction_resid, 
        params_init, 
        jac="3-point",
        args=args,
        bounds=(bounds_low, bounds_high),
        max_nfev=100,)
    
    # Pack variables/results into dict and return
    params = fit_dict["x"]

    scale_H2O = params[0]
    scale_O2 = params[1]

    # Add in dummy polynomial coeff if necessary
    if n_useful_orders != n_order:
        poly_coef = np.zeros((n_order, poly_order))
        poly_coef[useful_orders] = \
            params[2:].reshape((n_useful_orders, poly_order))
    else:
        poly_coef = params[2:].reshape((n_useful_orders, poly_order))

    # Compute final best-fit 'corrected' spectrum saving
    corr_spec, corr_sigma = correct_tellurics_and_apply_transfer_function(
        wave_obs_2D=wave_obs_2D_new[useful_orders],
        spec_obs_2D=spec_obs_2D_new[useful_orders],
        sigma_obs_2D=sigma_obs_2D_new[useful_orders],
        tau_H2O_2D=tau_H2O_2D[useful_orders],
        tau_O2_2D=tau_O2_2D[useful_orders],
        scale_H2O=scale_H2O,
        scale_O2=scale_O2,
        poly_coef=poly_coef[useful_orders])
    
    telluric_corr_spec_2D = np.full_like(spec_obs_2D_new, np.nan)
    telluric_corr_spec_2D[useful_orders] = corr_spec

    fit_dict["scale_H2O"] = scale_H2O
    fit_dict["scale_O2"] = scale_O2
    fit_dict["poly_coef"] = poly_coef
    fit_dict["wave_obs_2D"] = wave_obs_2D
    fit_dict["spec_obs_2D"] = spec_obs_2D
    fit_dict["spec_obs_2D"] = spec_obs_2D
    fit_dict["sigma_obs_2D"] = sigma_obs_2D
    fit_dict["wave_obs_2D_broad"] = wave_obs_2D_new
    fit_dict["spec_obs_2D_broad"] = spec_obs_2D_new
    fit_dict["sigma_obs_2D_broad"] = sigma_obs_2D_new
    fit_dict["spec_synth_2D"] = spec_synth_2D
    fit_dict["spec_fluxed_2D"] = spec_fluxed_2D
    fit_dict["extinction_2D"] = extinction_2D
    fit_dict["tau_H2O_2D"] = tau_H2O_2D
    fit_dict["tau_O2_2D"] = tau_O2_2D
    fit_dict["telluric_corr_spec_2D"] = telluric_corr_spec_2D

    return fit_dict


def flux_calibrate_mike_spectrum(
    wave_2D,
    spec_2D,
    sigma_2D,
    airmass,
    poly_order,
    arm,
    coeff_save_folder,
    label,
    extinction_curve_fn="data/paranal_extinction_patat2011.tsv",):
    """Flux calibrates a single MIKE spectrum using pre-computed polynomial
    coefficients from fit_atmospheric_transmission, the airmass of the target,
    and the observatory extinction.

    Parameters
    ----------
    wave_2D, spec_2D, sigma_2D: 2D float array
        Observed spectra of shape [n_order, n_px].

    wave_telluric, trans_H2O_telluric, trans_O2_telluric: 1D float array
        Telluric wavelength scale and transmission for H2O and O2.

    airmass: float
        Airmass of the target.

    poly_order: int
        Polynomial order to be fit as the transfer function for each order.

    arm: str
        String identifying the spectrograph arm.

    coeff_save_folder: str
        File path to save coeffients to.

    label: str
        Label for the file used to identify the flux standard, e.g.
        "<night>_<target_id>".

    extinction_curve_fn: str, default: 'data/paranal_extinction_patat2011.tsv'
        Path to extinction at observatory.
    """
    # Import polynomial coefficients
    orders, poly_coeff = pum.load_flux_calibration_poly_coeff(
        poly_order=poly_order,
        arm=arm,
        save_path=coeff_save_folder,
        label=label,)
    
    # Grab dimensions and sanity check
    (n_order, n_px) = wave_2D.shape

    if n_order != len(orders):
        raise Exception("Order numbers don't match!")

    # Import extinction curve + interpolate
    ext_pd = pd.read_csv(extinction_curve_fn, delimiter="\t")
    
    interp_extinction = interp1d(
        x=ext_pd["wave"].values,
        y=ext_pd["extinction"].values,
        bounds_error=False,
        fill_value=np.nan,
        kind="cubic",)

    # Initialise fluxed arrays
    spec_2D_fc = np.full_like(spec_2D, np.nan)
    sigma_2D_fc = np.full_like(sigma_2D, np.nan)

    # Flux calibrate
    for order_i in range(n_order):
        wave_1D = wave_2D[order_i]
        spec_1D = spec_2D[order_i]
        sigma_1D = sigma_2D[order_i]

        # Compute transfer function for this order
        tf_poly = Polynomial(poly_coeff[order_i])
        flux_cal_tf = 10.0**(-0.4*tf_poly(wave_1D))

        # Calculate extinction for object for this order
        obj_ext = 10.0**(-0.4*((airmass-1.0)*interp_extinction(wave_1D)))

        # Calculate combined correction factor
        flux_cal_comb = flux_cal_tf*obj_ext

        # Perform the flux calibration. TODO: exp, delta_lambda
        spec_2D_fc[order_i] = spec_1D / flux_cal_comb 
        sigma_2D_fc[order_i] = sigma_1D / flux_cal_comb

    return spec_2D_fc, sigma_2D_fc