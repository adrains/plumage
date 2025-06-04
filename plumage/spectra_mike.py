"""Functions to work with MIKE spectra.

TODO: there should be a dedicated spectra module, with e.g.:
    plumage.spectra.wifes and plumage.spectra.mike
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
from tqdm import tqdm
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.ticker as plticker

HEADERS_TO_EXTRACT = ["SITENAME", "SITEALT", "SITELAT", "SITELONG",
        "TELESCOP", "OBSERVER", "DATE-OBS", "UT-DATE", "UT-START", "LC-START",
        "INSTRUME", "RA-D", "DEC-D", "AIRMASS", "OBJECT", "EXPTIME", "NLOOPS",
        "BINNING", "SPEED", "SLITSIZE",]

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


def unify_wavelength_scales(exp_dict,):
    """Unifies the dimensions of the wavelength scales for exposures of the
    same targets taken in sequence in the event one or more orders did not
    extract properly. Missing stellar orders are populated with NaNs, whereas
    the wavelength scale and any calibration fluxes are copied from the
    reference exposure which has all orders extracted.

    exp_dict is updated in place.

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

    # All done, everything has been updated in place


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
        data_dict["wave"] = waves
        data_dict["spec_sky"] = data[0]
        data_dict["spec_star"] = data[1]
        data_dict["sigma_star"] = data[2]
        data_dict["snr_star"] = data[3]
        data_dict["spec_lamp"] = data[4]
        data_dict["spec_flat"] = data[5]
        data_dict["spec_star_norm"] = data[6]

    return data_dict


def load_and_combine_single_star(filenames, plot_folder,):
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
        Dictionary representation of MIKE fits files, with keys for relevant
        header keywords, the wavelength scale, and each of the seven (sky, obj,
        sigma, snr, lamp, flat, obj / flat) data-cubes. Each key corresponds to
        a list of length the number of exposures.

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
    unify_wavelength_scales(all_exp_dict)

    # -------------------------------------------------------------------------
    # Flat field normalisation
    # -------------------------------------------------------------------------
    # Normalise all stellar fluxes + sigmas by the flat field
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

    return all_exp_dict, obj_dict


def load_all_mike_fits(
    spectra_folder="spectra/",
    id_crossmatch_fn="data/mike_id_cross_match.tsv",
    plot_folder="",):
    """Load in all fits cubes containing 1D spectra to extract both the spectra
    and key details of the observations.

    Parameters
    ----------
    spectra_folder: string
        Root directory of nightly extracted 1D fits cubes.

    id_crossmatch_fn: default: 'data/mike_id_cross_match.tsv'

    plot_folder: str, default: ''
        Base folder to save diagnostic reduction plots to.

    Returns
    -------
    TODO
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
                    obj_blue_fns, plot_folder)
            else:
                exp_dict_b = None
                obj_dict_b = None

            # Read in red data
            is_obj_red_arm_date = np.all(np.stack(
                (obj_names == obj, is_red, ut_dates == ut_date)), axis=0)

            if np.sum(is_obj_red_arm_date) > 0:
                obj_red_fns = fits_files[is_obj_red_arm_date]
                print("\n\t:\n\t", "\n\t".join(obj_red_fns), sep="",)
                exp_dict_r, obj_dict_r = load_and_combine_single_star(
                    obj_red_fns, plot_folder)
            else:
                exp_dict_r = None
                obj_dict_r = None

            # Combine
            blue_dict[obj] = obj_dict_b
            red_dict[obj] = obj_dict_r

    return blue_dict, red_dict


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