"""Code for dealing with observed spectra
"""
import os
import numpy as np 
import pandas as pd
import glob
import pickle
import warnings
from tqdm import tqdm
from scipy.optimize import leastsq
from astropy.table import Table
from astropy.io import fits
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial, polyval
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d
from collections import Counter


# -----------------------------------------------------------------------------
# Loading spectra
# -----------------------------------------------------------------------------
def load_input_catalogue(catalogue_file="data/all_star_2m3_crossmatch.fits"):
    """Load the input catalogue of all science, calibrator, and standard
    targets from across all science programs.
    """
    dat = Table.read(catalogue_file, format="fits") 
    catalogue = dat.to_pandas() 

    return catalogue


def load_all_spectra(spectra_folder="spectra/", ext_snr=1, ext_fluxed=2,
                     ext_telluric_corr=3, include_subfolders=False):
    """Load in all fits cubes containing 1D spectra to extract both the spectra
    and key details of the observations.

    Parameters
    ----------
    spectra_folder: string
        Root directory of nightly extracted 1D fits cubes.

    ext_snr: int
        The fits extension with non-fluxed 1D spectra to get a measure of SNR.
    
    ext_fluxed: int
        The fits extension with fluxed 1D spectra.
    
    ext_telluric_corr: int
        The fits extension with fluxed *and* telluric corrected spectra.

    Returns
    -------
    observations: pandas dataframe
        Dataframe containing information about each observation.

    spectra_b: float array
        3D numpy array containing blue arm spectra of form 
        [N_ob, wl/spec/sigma, flux].

    spectra_r: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, flux].
    """
    # Initialise
    ids = []
    spectra_b = []   # blue
    spectra_r = []   # red
    snrs_b = []
    snrs_r = []
    exp_time = []
    obs_mjd = []
    obs_date = []
    ra = []
    dec = []
    airmass = []
    bcor = []
    readmode = []
    grating_b = []
    grating_r = []
    beam_splitter = []
    xbin = []
    ybin = []
    fullframe = []
    filename_b = []
    filename_r = []
    fluxed = []
    telluric_corr = []

    if not include_subfolders:
        spectra_b_path = os.path.join(spectra_folder, "*_b.fits")
        spectra_r_path = os.path.join(spectra_folder, "*_r.fits")
    else:
        spectra_b_path = os.path.join(spectra_folder, "*", "*_b.fits")
        spectra_r_path = os.path.join(spectra_folder, "*", "*_r.fits")
    
    spectra_b_files = glob.glob(spectra_b_path)
    spectra_r_files = glob.glob(spectra_r_path)

    spectra_b_files.sort()
    spectra_r_files.sort()

    bad_files = []

    # Ensure we have a matching number of blue and red spectra
    if len(spectra_b_files) != len(spectra_r_files):
        spec_files_b = [bb.split("_b")[0] for bb in spectra_b_files]
        spec_files_r = [rr.split("_r")[0] for rr in spectra_r_files]
        n_files = Counter(spec_files_b + spec_files_r)
        unmatched_spec = [si for si in n_files if n_files[si]==1]

        raise ValueError("Unequal number of blue and red spectra for following"
                         " observations: %s" % unmatched_spec)

    for fi, (file_b, file_r) in enumerate(
        zip(tqdm(spectra_b_files), spectra_r_files)):
        # Load in and extract required information from fits files
        with fits.open(file_b) as fits_b, fits.open(file_r) as fits_r:
            # Determine how many extensions the fits file has.
            #  2 extensions --> non-fluxed, no telluric corr [PyWiFeS p08]
            #  3 extensions --> fluxed, no telluric corr [PyWiFeS p09]
            #  4 extensions --> fluxed and telluric corr [PyWiFeS p10]
            if len(fits_b) != len(fits_r):
                raise ValueError("Blue and red have different # fits ext")
            
            if len(fits_b) == ext_snr+1:
                is_fluxed = False
                is_telluric_corr = False
                ext_sci = 1

            elif len(fits_b) == ext_fluxed+1:
                is_fluxed = True
                is_telluric_corr = False
                ext_sci = 2

            elif len(fits_b) == ext_telluric_corr+1:
                is_fluxed = True
                is_telluric_corr = True
                ext_sci = 3

            else: 
                raise ValueError("Unexpected # of HDUs. Should have 2-4.")

            # Take the "most-complete" (in terms of data reduction) extension
            # for use as the "science" spectrum
            spec_b = np.stack(fits_b[ext_sci].data)
            spec_r = np.stack(fits_r[ext_sci].data)

            # Ensure that there is actually signal here. If not, flag the files
            # as bad and skip processing them
            if (len(spec_b[:,1][np.isfinite(spec_b[:,1])]) == 0
                or len(spec_r[:,1][np.isfinite(spec_r[:,1])]) == 0):
                bad_files.append(file_b)
                bad_files.append(file_r)
                continue

            # Now that we know we're not working with bad files, add ext info
            fluxed.append(is_fluxed)
            telluric_corr.append(is_telluric_corr)

            # Get SNR measurements for each arm
            sig_b = np.median(fits_b[ext_snr].data["spectrum"])
            snrs_b.append(sig_b / sig_b**0.5)

            sig_r = np.median(fits_r[ext_snr].data["spectrum"])
            snrs_r.append(sig_r / sig_r**0.5)

            # HACK. FIX THIS.
            # Uncertainties on flux calibratated spectra don't currently 
            # make sense, so get the uncertainties from the unfluxxed spectra
            # in terms of fractions, then apply to the fluxed spectra 
            sigma_b_pc = (fits_b[ext_snr].data["sigma"] 
                          / fits_b[ext_snr].data["spectrum"])
            sigma_r_pc = (fits_r[ext_snr].data["sigma"]
                          / fits_r[ext_snr].data["spectrum"])
            
            # Sort out the uncertainties
            spec_b[:,2] = spec_b[:,1] * sigma_b_pc
            spectra_b.append(spec_b.T)

            spec_r[:,2] = spec_r[:,1] * sigma_r_pc
            spectra_r.append(spec_r.T)

            # Get object name and details of observation
            header = fits_b[0].header
            ids.append(header["OBJNAME"])
            exp_time.append(float(header["EXPTIME"]))
            obs_mjd.append(float(header["MJD-OBS"]))
            obs_date.append(header["DATE-OBS"])
            ra.append(header["RA"])
            dec.append(header["DEC"])
            airmass.append(float(header["AIRMASS"]))
            bcor.append(float(header["RADVEL"]))
            readmode.append(header["READMODE"])
            grating_b.append(header["GRATINGB"])
            grating_r.append(header["GRATINGR"])
            beam_splitter.append(header["BEAMSPLT"])
            xbin.append(int(header["CCDSUM"].split(" ")[0]))
            ybin.append(int(header["CCDSUM"].split(" ")[1]))

            # Determine if full or half-frame
            y_min = int(header["CCDSEC"].split(",")[-1].split(":")[0])
            if y_min == 1:
                fullframe.append(True)
            else:
                fullframe.append(False)

            filename_b.append(os.path.split(file_b)[-1])
            filename_r.append(os.path.split(file_r)[-1])
        
    # Now combine the arrays into our output structures
    spectra_b = np.stack(spectra_b)
    spectra_r = np.stack(spectra_r)

    # Convert arrays where necessary
    snrs_b = np.array(snrs_b).astype(float).astype(int)
    snrs_r = np.array(snrs_r).astype(float).astype(int)

    data = [ids, snrs_b, snrs_r, exp_time, obs_mjd, obs_date, ra, dec, airmass,
            bcor, readmode, grating_b, grating_r, beam_splitter, xbin, ybin, 
            fullframe, filename_b, filename_r, fluxed, telluric_corr]
    cols = ["id", "snr_b", "snr_r", "exp_time", "mjd", "date", "ra", 
            "dec", "airmass", "bcor_pw", "readmode", "grating_b", "grating_r", 
            "beam_splitter", "xbin", "ybin", "fullframe", "filename_b", 
            "filename_r", "fluxed", "telluric_corr"]

    # Create our resulting dataframe from a dict comprehension
    data = {col: vals for col, vals in zip(cols, data)}  
    observations = pd.DataFrame(data=data)

    # Print bad filenames
    print("Excluded %i bad (i.e. all nan) files: %s" % 
          (len(bad_files), bad_files))

    return observations, spectra_b, spectra_r


def save_pkl_spectra(observations, spectra_b, spectra_r, label="", 
                     rv_corr=False):
    """Save the imported spectra and observation info into respective pickle
    files in spectra/.

    Parameters
    ----------
    observations: pandas dataframe
        Dataframe containing information about each observation.

    spectra_b: float array
        3D numpy array containing blue arm spectra of form 
        [N_ob, wl/spec/sigma, flux].

    spectra_r: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, flux].

    label: string
        Unique component of the pickle filename.

    rv_corr: boolean
        Boolean flag for whether spectra have been RV corrected
    """
    # Distinguish whether these spectra are post normalisation and RV corr
    if rv_corr:
        ext = "_rv_corr"
    else:
        ext = ""

    # Save observation log
    ob_out = os.path.join("spectra", "saved_observations_%s%s.pkl" 
                          % (label, ext))
    pkl_obs = open(ob_out, "wb")
    pickle.dump(observations, pkl_obs)
    pkl_obs.close()

    # Save blue arm spectra
    sb_out = os.path.join("spectra", "saved_spectra_b_%s%s.pkl" % (label, ext))
    pkl_sb = open(sb_out, "wb")
    pickle.dump(spectra_b, pkl_sb)
    pkl_sb.close()

    # Save red arm 
    sr_out = os.path.join("spectra", "saved_spectra_r_%s%s.pkl" % (label, ext))
    pkl_sr = open(sr_out, "wb")
    pickle.dump(spectra_r, pkl_sr)
    pkl_sr.close()


def load_pkl_spectra(label, rv_corr=False):
    """Load the save spectra and observations from pickle files stored in
    spectra/.

    Parameters
    ----------
    label: string
        Unique component of the pickle filename.

    rv_corr: boolean
        Boolean flag for whether spectra have been RV corrected.

    Returns
    -------
    observations: pandas dataframe
        Dataframe containing information about each observation.

    spectra_b: float array
        3D numpy array containing blue arm spectra of form 
        [N_ob, wl/spec/sigma, value].

    spectra_r: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, value].
    """
    # Distinguish whether these spectra are post normalisation and RV corr
    if rv_corr:
        ext = "_rv_corr"
    else:
        ext = ""

    # Load observation log
    ob_out = os.path.join("spectra", "saved_observations_%s%s.pkl" 
                          % (label, ext))
    pkl_obs = open(ob_out, "rb")
    observations = pickle.load(pkl_obs)
    pkl_obs.close()

    # Load blue arm spectra
    sb_out = os.path.join("spectra", "saved_spectra_b_%s%s.pkl" % (label, ext))
    pkl_sb = open(sb_out, "rb")
    spectra_b = pickle.load(pkl_sb)
    pkl_sb.close()

    # Load red arm spectra
    sr_out = os.path.join("spectra", "saved_spectra_r_%s%s.pkl" % (label, ext))
    pkl_sr = open(sr_out, "rb")
    spectra_r = pickle.load(pkl_sr)
    pkl_sr.close()

    return observations, spectra_b, spectra_r


# -----------------------------------------------------------------------------
# Processing spectra
# -----------------------------------------------------------------------------
def clean_spectra(spectra):
    """Clean non-realistic spectral pixels

    Parameters
    ----------
    spectra: float array
        3D numpy array containing blue arm spectra of form 
        [N_ob, wl/spec/sigma, value].

    """
    # Suppress warnings so that the comparison operators don't trigger invalid
    # value warnings due to nans being in the array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) 

        # Set any spectra with negative or zero flux values to np.nan, along
        # with the associated error
        spectra[:,2] = np.where(spectra[:,1] <= 0, np.nan, spectra[:,2])
        spectra[:,1] = np.where(spectra[:,1] <= 0, np.nan, spectra[:,1])


def normalise_spectrum(wl, spectrum, e_spectrum=None, plot_fit=False, 
                       plot_norm=False):
    """Normalise a single spectrum by a 2nd order polynomial in log space. 
    Automatically detects which WiFeS arm and grating is being used and masks 
    out regions accordingly. Currently only implemented for B3000 and R7000.

    Parameters
    ----------
    wl: float array
        Wavelength scale for the spectrum.

    spectrum: float array
        The spectrum corresponding to wl, units irrelevant.

    e_spectrum: float array or None
        Errors (if available) on fluxes. None if not available or relevant
        (e.g. for synthetic spectra)

    plot_fit, plot_norm: boolean
        Whether to plot the normalised spectrum, or show the polynomial fit.

    Returns
    -------
    spectrum_norm: float array
        The normalised spectrum.
    """
    # Fit to flux in log space
    spectrum_fit = np.log(spectrum)
    #e_spectrum_fit = np.log(spectrum)

    # Red arm, R7000 grating
    if np.round(wl.mean()/100)*100 == 6200:
        lambda_0 = 6200
        edges = np.logical_or(wl < 5420, wl > 6980)
        nan = ~np.isfinite(spectrum_fit)
        ignore = np.logical_or.reduce((edges, nan))
    
    # Blue arm, B3000 grating
    elif np.round(wl.mean()/100)*100 == 4600:
        lambda_0 = 4600
        edges = np.logical_or(wl < 3600, wl > 5400)
        nan = ~np.isfinite(spectrum_fit)
        ignore = np.logical_or.reduce((edges, nan))

    # B3000+R7000 for synthetic testing
    elif np.round(wl.mean()/100)*100 == 5500:
        lambda_0 = 5500
        edges = np.logical_or(wl < 3500, wl > 7000) # This is everything
        nan = ~np.isfinite(spectrum_fit)
        ignore = np.logical_or.reduce((edges, nan))

    else:
        raise Exception("Grating not recognised or implemented. Must be"
                        " either B3000 or R7000.")

    # Make the mask
    #mask = np.ones_like(spectrum).astype(bool)
    mask = make_wavelength_mask(wl, mask_emission=True)
    mask[ignore] = False

    # Normalise wavelength scale (pivot about 0)
    wl_norm = (1/wl - 1/lambda_0)*(wl[0]-lambda_0)

    # Fit 2nd order polynomial to get coefficients
    poly = Polynomial.fit(wl_norm[mask], spectrum_fit[mask], 2)

    # Calculate the normalising function and normalise
    norm = poly(wl_norm)

    spectrum_norm = spectrum / np.exp(norm)

    # Plot
    if plot_fit:
        plt.figure()
        plt.plot(wl[:-1], spectrum_fit[:-1], label="flux")
        plt.plot(wl, norm, label="fit")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Flux (Normalised)")

    elif plot_norm:
        plt.figure()
        plt.plot(wl, spectrum_norm, label="Normalised flux")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Flux (Normalised)")

    # Normalise the uncertainties too if we have been given them
    if e_spectrum is not None:
        e_spectrum_norm = e_spectrum / np.exp(norm)

        return spectrum_norm, e_spectrum_norm
    
    else:
        return spectrum_norm


def normalise_spectra(spectra, normalise_uncertainties=False):
    """Normalises all spectra

    Parameters
    ----------
    spectra: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, value].

    normalise_uncertainties: boolean
        Whether to normalise uncertainties (use False if no uncertainties).
    """
    spectra_norm = spectra.copy()

    for spec_i in tqdm(range(len(spectra_norm)), desc="Normalising spectra"):
        if normalise_uncertainties:
            spec_norm, e_spec_norm = normalise_spectrum(spectra[spec_i][0,:],
                                       spectra[spec_i][1,:],
                                       spectra[spec_i][2,:]) 

            spectra_norm[spec_i][1,:] = spec_norm
            spectra_norm[spec_i][2,:] = e_spec_norm

        else:
            spec_norm = normalise_spectrum(spectra[spec_i][0,:],
                                       spectra[spec_i][1,:])
            spectra_norm[spec_i][1,:] = spec_norm
    
    return spectra_norm


def norm_spec_by_wl_region(wave, spectrum, arm, e_spec=None):
    """Normalise spectra by a specific wavelength region for synthetic fitting.
    """
    # Normalise by the region just to the blue of H alpha
    if arm == "r":
        norm_mask = np.logical_and(wave > 6535, wave < 6545)

    # Normalise by bandhead near 5000 A
    elif arm == "b":
        norm_mask = np.logical_and(wave > 4925, wave < 4950)

    else:
        raise ValueError("Arm must be either r or b")

    # Normalise the specta (and errors if provided)
    spec_norm = spectrum / np.nanmean(spectrum[norm_mask])

    if e_spec is not None:
        e_spec_norm = e_spec / np.nanmean(spectrum[norm_mask])
        return spec_norm, e_spec_norm

    else:
        return spec_norm


def make_wavelength_mask(wave_array, mask_emission=True, 
    mask_blue_edges=False, mask_sky_emission=False, mask_edges=False,
    mask_bad_px=True):
    """Make a wavelength mask for the provided wave array, masking out the 
    regions specified.

    Parameters
    ----------
    wave_array: float array
        Wavelength scale to be masked.

    mask_emission: boolean
        Whether to mask stellar activity based emission.

    mask_blue_edges: boolean
        Whether to mask B3000 spectra to match the R7000 wavelength range.

    mask_sky_emission: boolean
        Whether to mask regions prone to sky emission.

    mask_edges: boolean
        Whether to mask edges of wavelength scale to avoid edge effects when
        fitting.

    Returns
    -------
    mask: boolean array
        Boolean mask for wavelength scale.
    """
    # O2  bands are saturated - don't depend on airmass
    # H2O bands DO depend on airmass!!
    
    O2_telluric_bands = [
        [6856.0, 6956.0],
        [7584.0, 7693.0]]
        #[7547.0, 7693.0]]

    H2O_telluric_bands = [
        [6270.0, 6290.0],
        [7154.0, 7332.0],
        [8114.0, 8344.0],
        [8937.0, 9194.0],
        [9270.0, 9776.0]]

    strong_H2O_telluric_bands = [
        [6270.0, 6290.0],
        [7154.0, 7332.0],
        [8114.0, 8344.0],
        [8937.0, 9194.0],
        [9270.0, 9400.0]]

    master_H2O_telluric_bands = [
        #[5870.0, 6000.0],          # This is the Na doublet
        [6270.0, 6290.0],
        [6459.0, 6598.0],
        [7154.0, 7332.0],
        [8114.0, 8344.0],
        [8937.0, 9194.0],
        [9270.0, 9776.0]]

    balmer_series = [
        [3825.0, 3845.0], # H Eta
        [3880.0, 3900.0], # H Zeta
        [3960.0, 3980.0], # H Epsilon
        [4090.0, 4110.0], # H Delta
        [4330.0, 4350.0], # H Gamma
        [4850.0, 4870.0], # H Beta
        [6550.0, 6575.0], # H Alpha
    ]

    calcium_hk = [
        [3929.0, 3939.0], # Ca II K
        [3964.0, 3974.0], # Ca II H
    ]

    sky_emission = [
        [5575.0, 5580.0],
        #[5885.0, 5900.0],
        [6298.0, 6303.0],
        #[6360.0, 6370.0],
        #[6580.0, 6585.0]
    ]

    bad_px = [
        [5575.0, 5581.0]    # bad column
    ]

    band_list = O2_telluric_bands + strong_H2O_telluric_bands

    # Mask out Balmer series
    if mask_emission:
        band_list += balmer_series + calcium_hk

    # In cases of poor sky subtraction, get rid of sky emission
    if mask_sky_emission:
        band_list += sky_emission

    if mask_bad_px:
        band_list += bad_px

    mask = np.ones(len(wave_array))

    for band in band_list:
        mask *= ((wave_array <= band[0])+
                 (wave_array >= band[1]))
    
    # Mask out blue edges and red overlap region
    if mask_blue_edges:
        mask *= ((wave_array >= 3600)*
                 (wave_array <= 5400))

    # Mask out the edges to avoid edge effects
    if mask_edges:
        mask[:10] = False
        mask[-10:] = False

    return mask.astype(bool)


def mask_wavelengths(spectra, mask_emission=True, mask_blue_edges=False, 
                     mask_sky_emission=False):
    """
    """
    wl_mask = make_wavelength_mask(
        spectra[0,0], 
        mask_emission=mask_emission,
        mask_blue_edges=mask_blue_edges,
        mask_sky_emission=mask_sky_emission,)

    dims = spectra.shape
    wl_mask = np.tile(wl_mask, dims[0]*dims[1]).reshape(dims)
    
    spectra = spectra[wl_mask]
    spectra = spectra.reshape(
        [dims[0], dims[1], int(len(spectra)/np.prod(dims[:2]))])

    return spectra


def reformat_spectra(wl, spec):
    """Take a single wavelength vector plus a 2D set of spectra and combine
    them into a single array.
    """
    wl_2d = np.tile(wl, len(spec)).reshape((len(spec),len(wl))) 
    spec_all = np.stack((wl_2d, spec),axis=2).swapaxes(1,2)

    return spec_all


def make_wl_scale(wl_min, wl_max, n_px):
    """Make a new wavelength scale given min and max wavelengths, plus number
    of pixels.

    Parameters
    ----------
    wl_min, wl_max: float
        Minimum and maximum wavelength values respectively.

    n_px: int
        Number of pixels to have in the wavelength scale.

    Returns
    -------
    wl_scale: float array
        New wavelength scale.
    """
    wl_per_px = (wl_max - wl_min) / n_px
    wl_scale = np.arange(wl_min, wl_max, wl_per_px)

    return wl_scale


def prepare_spectra(spec_b, spec_r, use_both_arms, remove_blue_overlap=False):
    """
    """
    # Mask wavelengths
    wl_mask = make_wavelength_mask(spec_b[0,0], mask_blue_edges=True,
                                   mask_emission=True, mask_sky_emission=True)

    if remove_blue_overlap:
        wl_mask = np.logical_and(wl_mask, spec_b[0,0] < 5400)
    
    spec_b_m = spec_b[:, :, wl_mask]

    wl_mask = make_wavelength_mask(spec_r[0,0], mask_emission=True, mask_sky_emission=True)
    spec_r_m = spec_r[:, :, wl_mask]
    
    # Separate out the spectra
    if use_both_arms:
        wls = np.concatenate((spec_b_m[0,0,:], spec_r_m[0,0,:]))

        training_set_flux = np.concatenate((spec_b_m[:,1,:], spec_r_m[:,1,:]), 
                                        axis=1)
        training_set_ivar = np.concatenate((1/spec_b_m[:,2,:]**2, 
                                        1/spec_r_m[:,2,:]**2), axis=1)
    else:
        wls = spec_r_m[0,0,:]
        training_set_flux = spec_r_m[:,1,:]
        training_set_ivar = 1/spec_r_m[:,2,:]**2
    #training_set_ivar = 1e5 * np.ones_like(std_spectra_r[:,2,:]) # TEMPORARY

    # If flux is nan, set to 1 and give high variance (inverse variance of 0)
    training_set_ivar[~np.isfinite(training_set_flux)] = 1e-8
    training_set_flux[~np.isfinite(training_set_flux)] = 1

    # If the inverse variance is nan, do the same
    training_set_flux[~np.isfinite(training_set_ivar)] = 1
    training_set_ivar[~np.isfinite(training_set_ivar)] = 1e-8

    return wls, training_set_flux, training_set_ivar

# -----------------------------------------------------------------------------
# Radial Velocities
# -----------------------------------------------------------------------------
def compute_barycentric_correction(ras, decs, times, exps, site="SSO", 
                                   disable_auto_max_age=False,
                                   overrid_iers=False):
    """Compute the barycentric corrections for a set of stars

    In late 2019 issues were encountered accessing online files related to the
    International Earth Rotation and Reference Systems Service. This is 
    required to calculate barycentric corrections for the data. This astopy 
    issue may prove a useful resource again if the issue reoccurs:
    - https://github.com/astropy/astropy/issues/8981

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
    loc = EarthLocation.of_site(site)

    # Initialise barycentric correction array
    bcors = []

    if disable_auto_max_age:
        #from astropy.utils.iers import IERS_A_URL
        IERS_A_URL = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals.all'
        #from astropy.utils.iers import conf
        #conf.auto_max_age = None
    
    # Override the IERS server if the mirror is down or not up to date
    if overrid_iers:
        from astropy.utils import iers
        from astropy.utils.iers import conf as iers_conf
        url = "https://datacenter.iers.org/data/9/finals2000A.all"
        iers_conf.iers_auto_url = url
        iers_conf.reload()

    # Calculate the barycentric correction for every star
    for ra, dec, time, exp in zip(tqdm(ras), decs, times, exps):
        sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

        # Get the *mid-point* of the observeration
        time = Time(float(time), format="mjd") + 0.5*float(exp)*u.second
        barycorr = sc.radial_velocity_correction(obstime=time, location=loc)  
        bcors.append(barycorr.to(u.km/u.s).value)

    return bcors


def calc_rv_shift_residual(rv, wave, spec, e_spec, ref_spec_interp, bcor):
    """Loss function for least squares fitting.

    Parameters
    ----------
    rv: float array
        rv[0] is the radial velocity in km/s
    
    wave: float array
        Wavelength scale of the science spectra

    spec: float array
        Fluxes for the science spectra

    e_spec: float array
        Uncertainties on the science fluxes

    ref_spec_interp: InterpolatedUnivariateSpline function
        Interpolation function to take wavelengths and compute template
        spectrum

    bcor: float
        Barycentric velocity in km/s

    Returns
    -------
    resid_vect: float array
        Error weighted loss.
    """
    # Shift the template spectrum
    ref_spec = ref_spec_interp(wave * (1-(rv[0]-bcor)/(const.c.si.value/1000)))

    # Return loss
    resid_vect = (spec - ref_spec) / e_spec

    #chi_sq = np.nansum(resid_vect**2)

    return resid_vect


def calc_rv_shift(ref_wl, ref_spec, sci_wl, sci_spec, e_sci_spec, bad_px_mask, 
                  bcor):
    """Calculate the RV shift for a single science and template spectrum.

    Parameters
    ----------
    ref_wl: float array
        Wavelength scale of the template spectra
    
    ref_spec: float array
        Fluxes for the template spectra

    sci_wl: float array
        Wavelength scale of the science spectra

    sci_spec: float array
        Fluxes for the science spectra

    e_sci_spec: float array
        Uncertainties on the science fluxes

    sci_mask: float array
        Boolean mask for which spectral pixels to *not* include in the fit.

    bcor: float
        Barycentric velocity in km/s

    Returns
    -------
    rv_fit, e_rv, rchi2: float
        Best fit RV (km/s), RV uncertainty, and corresponding reduced chi^2.
    
    infodict: dictionary
        Outputs of scipy.optimize.leastsq()
    """
    # Make interpolation function
    ref_spec_interp = ius(ref_wl, ref_spec)

    # Mask the science data by setting the masked regions to nans
    sci_wl_m = sci_wl.copy()
    sci_spec_m = sci_spec.copy()
    e_sci_spec_m = e_sci_spec.copy()

    #sci_wl_m = sci_wl_m[bad_px_mask]
    #sci_spec_m = sci_spec_m[bad_px_mask]
    sci_spec_m[bad_px_mask] = 1
    e_sci_spec_m[bad_px_mask] = 1E30

    e_sci_spec_m[np.isnan(sci_spec_m)] = 1E30
    sci_spec_m[np.isnan(sci_spec_m)] = 1

    # Do fit, have initial RV guess be 0 km/s
    rv = 0
    args = (sci_wl_m, sci_spec_m, e_sci_spec_m, ref_spec_interp, bcor)
    rv_fit, cov, infodict, mesg, ier = leastsq(calc_rv_shift_residual, rv, 
                                            args=args, full_output=True)

    # Scale the covariance matrix, calculate RV uncertainty
    # docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
    if cov is None:
        e_rv = np.nan
    else:
        cov *= np.nanvar(infodict["fvec"])
        e_rv = cov**0.5

    # Add extra parameters to infodict
    infodict["template_spec"] = ref_spec_interp(
        sci_wl_m * (1-(float(rv_fit)-bcor)/(const.c.si.value/1000)))
    infodict["cov"] = cov
    infodict["mesg"] = mesg
    infodict["ier"] = ier

    # Calculate reduced chi^2
    rchi2 = np.nansum(infodict["fvec"]**2) / (len(sci_spec)-len(rv_fit))
    
    return float(rv_fit), float(e_rv), rchi2, infodict


def do_template_match(sci_spectra, bcor, ref_params, ref_spectra, arm,
                      print_diagnostics=False, n_std=3, mask_blue_edges=True):
    """Find the best fitting template to determine the RV and temperature of
    the star.

    Parameters
    ----------
    sci_spectra: float array
        2D numpy array containing spectra of form [wl/spec/sigma, flux].
    
    bcor: float
        Barycentric velocity in km/s

    ref_params: float array
        Array of stellar parameters of form [teff, logg, feh]
    
    ref_spectra: float array
        Array of imported template spectra of form [star, wl/spec, flux]
    
    print_diagnostics: boolean
        Whether to print diagnostics about the fit

    Returns
    -------
    rv_fit, e_rv, rchi2: float
        Best fit RV (km/s), RV uncertainty, and corresponding reduced chi^2.

    nres: float array
        Fit residuals.

    params: float
        Fitted stellar parameters vector

    rchi2_grid: float array
        Grid of reduced chi^2 values corresponding to all template spectra.
    """
    # Ensure we have a correct arm
    valid_arms = ["b", "r"]

    if arm not in valid_arms:
        raise ValueError("Arm must be either r or b")

    # Fit each template to the science data to figure out the best match
    rvs = []
    e_rvs = []
    rchi2_grid = []
    norm_res = []
    info_dicts = []

    for params, ref_spec in zip(ref_params, ref_spectra):
        # Initialise bad pixel mask. Initially this mask will exclude telluric
        # regions, any pixels that are nan (e.g. due to having possessed 
        # negative flux values) and the very edges of the blue data. Will not
        # exclude emission regions (i.e. the Balmer series and Ca II H+K) as 
        # these will be captured by the second pass where we exclude pixels 
        # with large residuals

        # Only mask blue edges if we're dealing with blue data, otherwise leave
        # as is
        if arm != "b":
            mask_blue_edges = False

        bad_px_mask = ~make_wavelength_mask(
            sci_spectra[0],
            mask_emission=False,
            mask_blue_edges=mask_blue_edges)

        # Now that we have our initial mask, OR this with the array of any nan
        # fluxes or uncertainties
        nan_px_mask = np.logical_or(
            ~np.isfinite(sci_spectra[1]), 
            ~np.isfinite(sci_spectra[2]))

        bad_px_mask = np.logical_or(bad_px_mask, nan_px_mask)
        
        # Finally, mask out the final pixel as it is known to be bad 
        # TODO: generalise
        bad_px_mask[-1] = True 

        # Do first fit without any masking beyond tellurics
        rv, e_rv, rchi2, infodict = calc_rv_shift(
            ref_spec[0,:], 
            ref_spec[1,:], 
            sci_spectra[0,:], 
            sci_spectra[1,:], 
            sci_spectra[2,:],
            bad_px_mask,
            bcor)

        # Now mask out any pixels that have residuals greater than the 
        # threshold value, and fit again
        resid = infodict["fvec"]
        std = np.std(resid[~bad_px_mask])
        bad_px_mask[np.abs(resid) > (n_std*std)] = True

        # Do second fit, now with masking
        rv, e_rv, rchi2, infodict = calc_rv_shift(
            ref_spec[0,:], 
            ref_spec[1,:], 
            sci_spectra[0,:], 
            sci_spectra[1,:], 
            sci_spectra[2,:],
            bad_px_mask,
            bcor)

        infodict["bad_px_mask"] = bad_px_mask

        rvs.append(rv)
        e_rvs.append(e_rv)
        rchi2_grid.append(rchi2)
        norm_res.append(infodict["fvec"])
        info_dicts.append(infodict)
        
        if print_diagnostics:
            print("\tTeff = %i K, RV = %0.2f km/s, quality = %0.2f" 
                  % (params[0], rv, infodict["fvec"]))

    # Now figure out what best fit is
    fit_i = np.argmin(rchi2_grid)

    rv = np.array(rvs[fit_i])
    e_rv = np.array(e_rvs[fit_i])
    rchi2 = np.array(rchi2_grid[fit_i])
    nres = np.array(norm_res[fit_i])
    params = np.array(ref_params[fit_i])
    info_dict = info_dicts[fit_i]

    return rv, e_rv, rchi2, nres, params, rchi2_grid, info_dict


def do_all_template_matches(sci_spectra, observations, ref_params, ref_spectra,
                            arm, print_diagnostics=False, save_column_ext=""):
    """Do template fitting on all stars for radial velocity and temperature, 
    and save the results to observations.

    Parameters
    ----------
    sci_spectra: float array
        3D numpy array containing spectra of form [N_ob, wl/spec/sigma, flux].
    
    observations: pandas dataframe
        Dataframe containing information about each observation.

    ref_params: float array
        Array of stellar parameters of form [teff, logg, feh]
    
    ref_spectra: float array
        Array of imported template spectra of form [star, wl/spec, flux]
    
    arm: string
        Which WiFeS arm we're using (either b or r).

    print_diagnostics: boolean
        Whether to print diagnostics about the fit

    Returns
    -------
    nres: float array
        Fit residuals for all science spectra.

    rchi2_grid: float array
        Grid of reduced chi^2 values corresponding to all template spectra for
        all science spectra.
    """
    # Initialise
    all_rvs = []
    all_e_rvs = []
    all_rchi2 = []
    all_nres = []
    all_params = []
    all_rchi2_grid = []
    info_dicts = []
    bad_px_masks = []

    # For every star, do template fitting
    for star_i, sci_spec in enumerate(tqdm(sci_spectra, desc="Fitting RVs")):
        if print_diagnostics:
            print("\n(%4i/%i) Running fitting on %s:" 
                 % (star_i+1, len(sci_spectra), 
                    observations.iloc[star_i]["id"]))

        bcor = observations.iloc[star_i]["bcor"]
        rv, e_rv, rchi2, nres, params, rchi2_grid, idict = do_template_match(
            sci_spec, 
            bcor, 
            ref_params, 
            ref_spectra, 
            arm,
            print_diagnostics)
        
        all_rvs.append(rv)
        all_e_rvs.append(e_rv)
        all_rchi2.append(rchi2)
        all_nres.append(nres)
        all_params.append(params)
        all_rchi2_grid.append(rchi2_grid)
        info_dicts.append(idict)
        bad_px_masks.append(idict["bad_px_mask"])

    # Convert to numpy arrays
    all_params = np.array(all_params)
    all_nres = np.array(all_nres)
    all_rchi2_grid = np.array(all_rchi2_grid)
    bad_px_masks = np.array(bad_px_masks)

    # Add to observations
    if save_column_ext != "":
        save_column_ext = "_{}".format(save_column_ext)

    observations["teff_fit{}".format(save_column_ext)] = all_params[:,0]
    observations["logg_fit{}".format(save_column_ext)] = all_params[:,1]
    observations["feh_fit{}".format(save_column_ext)] = all_params[:,2]
    observations["vsini_fit{}".format(save_column_ext)] = all_params[:,3]
    observations["rv{}".format(save_column_ext)] = np.array(all_rvs)
    observations["e_rv{}".format(save_column_ext)] = np.array(all_e_rvs)
    observations["rchi2{}".format(save_column_ext)] = np.array(all_rchi2)

    return all_nres, all_rchi2_grid, bad_px_masks, info_dicts


def correct_rv(sci_spectra, bcor, rv, wl_new):
    """Interpolate science spectrum onto new wavelength scale in the rest
    frame. This uses the opposite sign convention of calc_rv_shift_residual.

    Parameters
    ----------
    sci_spectra: float array
        2D numpy array containing spectra of form [wl/spec/sigma, flux].
    
    bcor: float
        Barycentric velocity in km/s

    rv: float array
        Fitted radial velocity in km/s

    wl_new: float array
        New wavelength scale to regrid the spectra onto once in the rest frame

    Returns
    -------
    rest_frame_spec: float array
        2D numpy array containing spectra of form [wl/spec/sigma, flux] now
        in the rest frame
    """
    # Setup the interpolation
    calc_spec = interp1d(sci_spectra[0,:], sci_spectra[1,:], kind="linear",
                         bounds_error=False, assume_sorted=True)
    
    calc_sigma = interp1d(sci_spectra[0,:], sci_spectra[2,:], kind="linear",
                         bounds_error=False, assume_sorted=True)

    # We're *undoing* the shift imparted by barycentric motion and radial
    # velocity, so this relation will have an opposite sign to the one in
    # calc_rv_shift_residual.
    rest_frame_spec = calc_spec(wl_new * (1+(rv-bcor)/(const.c.si.value/1000)))
    rest_frame_sigma = calc_sigma(wl_new * (1+(rv-bcor)/(const.c.si.value/1000)))

    rest_frame_spec = np.stack([wl_new, rest_frame_spec, rest_frame_sigma])

    return rest_frame_spec


def correct_all_rvs(sci_spectra, observations, wl_new):
    """
    Parameters
    ----------
    sci_spectra: float array
        3D numpy array containing spectra of form [N_ob, wl/spec/sigma, flux].
    
    observations: pandas dataframe
        Dataframe containing information about each observation.

    wl_new: float array
        New wavelength scale to regrid the spectra onto once in the rest frame

    Returns
    -------
    rest_frame_spectra: float array
        3D numpy array containing spectra of form [star, wl/spec/sigma, flux] 
        now in the rest frame
    """
    #Initialise
    rest_frame_spectra = []

    for star_i, sci_spec in enumerate(sci_spectra):
        bcor = observations.iloc[star_i]["bcor"]
        rv = observations.iloc[star_i]["rv"]
        
        rest_frame_spectra.append(correct_rv(sci_spec, bcor, rv, wl_new))

    rest_frame_spectra = np.stack(rest_frame_spectra)

    return rest_frame_spectra