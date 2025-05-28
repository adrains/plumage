"""Functions to work with MIKE spectra.

TODO: there should be a dedicated spectra module, with e.g.:
    plumage.spectra.wifes and plumage.spectra.mike
"""
import numpy as np
from astropy.io import fits
import warnings
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord, EarthLocation

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


def load_and_combine_single_star(filenames,):
    """Load in a list of fits files corresponding to reduced MIKE data of a
    *single* arm, and co-add the spectra.

    Parameters
    ----------
    filenames: str list
        List of filenames of reduced MIKE fits files to open.

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
    for fn in filenames:
        exp_dict = load_single_mike_fits(fn)

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