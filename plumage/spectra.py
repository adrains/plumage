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
                     ext_telluric_corr=3, include_subfolders=False, 
                     use_counts_ext_and_flux=False, 
                     correct_negative_fluxes=False,):
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
    corrected_neg_fluxes_b_all = []
    corrected_neg_fluxes_r_all = []
    n_neg_px_b_all = []
    n_neg_px_r_all = []
    neg_flux_weight_b_all = []
    neg_flux_weight_r_all = []

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
            # If we're relying on our own flux calibration, no need to bother
            # with other extensions
            if use_counts_ext_and_flux:
                is_fluxed = True
                is_telluric_corr = False
                ext_sci = 1

            # Otherwise work as usual with what PyWiFeS outputs
            else:
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

            # PyWiFeS/process_stellar extraction causes non-physical negative
            # fluxes for some stars, likely those that have a substantial sky
            # background from e.g. a nearby star. Short of re-reducing these 
            # stars, we can correct this here in a ~mostly~ rigorous way. 
            # Uncertainties from this approach will likely be overestimated.
            # TODO: implement this for when we're not doing our own fluxing.
            corrected_neg_fluxes_b = False
            corrected_neg_fluxes_r = False

            n_neg_px_b = np.nansum(spec_b[:,1] < 0)
            n_neg_px_r = np.nansum(spec_r[:,1] < 0)

            neg_flux_weight_b = np.nan
            neg_flux_weight_r = np.nan
            
            if correct_negative_fluxes and (ext_sci == ext_snr):
                # Only run if we have negative fluxes
                if n_neg_px_b > 0:
                    spec_b, neg_flux_weight_b, _ = correct_neg_counts(spec_b)
                    corrected_neg_fluxes_b = True

                if n_neg_px_r > 0:
                    spec_r, neg_flux_weight_r, _ = correct_neg_counts(spec_r)
                    corrected_neg_fluxes_r = True

            # Regardless, store info on negative flux processing
            corrected_neg_fluxes_b_all.append(corrected_neg_fluxes_b)
            corrected_neg_fluxes_r_all.append(corrected_neg_fluxes_r)
            n_neg_px_b_all.append(n_neg_px_b)
            n_neg_px_r_all.append(n_neg_px_r)
            neg_flux_weight_b_all.append(neg_flux_weight_b)
            neg_flux_weight_r_all.append(neg_flux_weight_r)

            # Get headers
            header_b = fits_b[0].header
            header_r = fits_r[0].header

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
            
            # Doing our own flux correction
            if use_counts_ext_and_flux:
                # Import blue
                spec_b_fluxed, e_spec_b_fluxed = flux_calibrate_spectra(
                    wave=spec_b[:,0],
                    spectra=spec_b[:,1],
                    e_spectra=spec_b[:,2],
                    airmass=float(header_b["AIRMASS"]),
                    arm="b",
                    exptime=float(header_b['EXPTIME']),
                )

                # Import red
                spec_r_fluxed, e_spec_r_fluxed = flux_calibrate_spectra(
                    wave=spec_r[:,0],
                    spectra=spec_r[:,1],
                    e_spectra=spec_r[:,2],
                    airmass=float(header_r["AIRMASS"]),
                    arm="r",
                    exptime=float(header_r['EXPTIME']),
                )

                # Save
                spec_b[:,1] = spec_b_fluxed
                spec_b[:,2] = e_spec_b_fluxed

                spec_r[:,1] = spec_r_fluxed
                spec_r[:,2] = e_spec_r_fluxed

            else:
                # HACK. FIX THIS.
                # Uncertainties on flux calibratated spectra don't currently 
                # make sense, get the uncertainties from the unfluxxed spectra
                # in terms of fractions, then apply to the fluxed spectra 
                sigma_b_pc = (fits_b[ext_snr].data["sigma"] 
                            / fits_b[ext_snr].data["spectrum"])
                sigma_r_pc = (fits_r[ext_snr].data["sigma"]
                            / fits_r[ext_snr].data["spectrum"])
                
                # Sort out the uncertainties
                spec_b[:,2] = spec_b[:,1] * sigma_b_pc

                spec_r[:,2] = spec_r[:,1] * sigma_r_pc

            # Append
            spectra_b.append(spec_b.T)
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
            fullframe, filename_b, filename_r, fluxed, telluric_corr,
            corrected_neg_fluxes_b_all, corrected_neg_fluxes_r_all, 
            n_neg_px_b_all, n_neg_px_r_all, neg_flux_weight_b_all, 
            neg_flux_weight_r_all]
    cols = ["id", "snr_b", "snr_r", "exp_time", "mjd", "date", "ra", 
            "dec", "airmass", "bcor_pw", "readmode", "grating_b", "grating_r", 
            "beam_splitter", "xbin", "ybin", "fullframe", "filename_b", 
            "filename_r", "fluxed", "telluric_corr", "corrected_neg_fluxes_b", 
            "corrected_neg_fluxes_r", "n_neg_px_b", "n_neg_px_r",
            "neg_flux_weight_b", "neg_flux_weight_r",]

    # Create our resulting dataframe from a dict comprehension
    data = {col: vals for col, vals in zip(cols, data)}  
    observations = pd.DataFrame(data=data)

    # Print bad filenames
    print("Excluded %i bad (i.e. all nan) files: %s" % 
          (len(bad_files), bad_files))

    return observations, spectra_b, spectra_r


def correct_neg_counts(spec, var_rn=11, n_max_weighting=10, 
    max_px_flux_fac=0.5, useful_frac_uncertainty=0.5, 
    negative_flux_correct_fac=1.01, do_plot=False,):
    """

    Note: this function should *only* be run if there are negative fluxes,
    implying an incorrect sky subtraction.
    """
    # Construct mask to only use positive values of flux, and pixels with 
    # reasonable fractional uncertainties
    is_pos = spec[:,1] > 0
    useful_uncertainties = (spec[:,2] / spec[:,1]) < useful_frac_uncertainty
    mask = np.logical_and(is_pos, useful_uncertainties)

    # Calculate Poisson uncertainty based on the assumed flux on the brightest
    # spaxel (by default 50% of total flux)
    sigma_poisson = max_px_flux_fac * spec[mask,1] / np.sqrt(spec[mask,1])

    # Now find the best fit weighting (a combination of the number of pixels
    # readout, and the seeing - i.e. how concentrated/distributed the flux is).
    # We do this by approximating the uncertainty as Poisson + readout noise,
    # and seeing what value of weighting matches best with the observed 
    # uncertainties.
    delta_sigma_all = np.zeros(n_max_weighting+1)

    for n_weighting in range(0, n_max_weighting+1):
        # Predict our uncertainty with the current value of n_weighting
        sigma_rn = n_weighting * var_rn**0.5
        sigma_total = np.sqrt(sigma_poisson**2 + sigma_rn**2)

        # Calculate the difference between this and our observed uncertainties
        delta_sigma = (spec[mask,2] / spec[mask,1]) - (sigma_total / spec[mask,1])

        delta_sigma_all[n_weighting] = np.nansum(np.abs(delta_sigma))

    # Find our optimal weighting - it will have the minimum difference
    fit_weighting = np.argmin(delta_sigma_all)

    # Now that we've computed the weighting from the data as is, correct
    # the negative fluxes
    min_count = np.abs(np.nanmin(spec[:,1]))

    # Increase all pixels by negative_flux_correct_fac x this minimum value 
    # (i.e. make the assumption that our minimum pixel got some flux, avoid 
    # division by 0)
    out_spec = spec.copy()
    out_spec[:,1] = spec[:,1] + negative_flux_correct_fac * min_count

    # Now recalculate the uncertainties by adding the Poisson uncertainty in 
    # quadrature with readout noise by assuming our calculated weighting
    sigma_poisson = out_spec[:,1] / np.sqrt(out_spec[:,1])
    sigma_rn = fit_weighting * var_rn**0.5
    
    # And update the uncertainties
    sigma_new = np.sqrt(sigma_poisson**2 + sigma_rn**2)
    out_spec[:,2] = sigma_new

    #sigma_fac = (out_spec[:,2]/out_spec[:,1]) / (spec[:,2]/spec[:,1])
    #print("Sigma fac = {:0.2f}".format(np.nanmedian(sigma_fac[mask])))

    if do_plot:
        plt.close("all")
        plt.errorbar(
            x=out_spec[:,0],
            y=out_spec[:,1],
            yerr=out_spec[:,2],
            linewidth=0.4,
            color="#1f77b4",
            ecolor="k",
            elinewidth=0.1,
            label="Corrected",)

        plt.errorbar(
            x=spec[:,0],
            y=spec[:,1],
            yerr=spec[:,2],
            linewidth=0.4,
            color="#ff7f0e",
            ecolor="k",
            elinewidth=0.1,
            label="Uncorrected",)

        plt.legend(loc="best")

    return out_spec, fit_weighting, delta_sigma_all

    

# -----------------------------------------------------------------------------
# Processing spectra
# -----------------------------------------------------------------------------
def flux_calibrate_spectra(
    wave,
    spectra,
    e_spectra,
    airmass,
    arm,
    exptime,
    flux_cal_B3000_file="data/flux_cal_B3000.csv",
    flux_cal_R7000_file="data/flux_cal_R7000.csv",
    sso_extinction_file="data/sso_extinction.dat"):
    """
    """
    # Import flux calibration curve
    if arm == "b":
        flux_cal_data = np.loadtxt(flux_cal_B3000_file)
        wl_step = 0.77

    elif arm == "r":
        flux_cal_data = np.loadtxt(flux_cal_R7000_file)
        wl_step = 0.44

    else:
        raise ValueError("Invalid arm, must be either 'b' or 'r'")

    # Assume all wavelength vectors are the same from PyWiFeS
    #wave = spectra[0,0]

    # Create flux curve interpolation function
    compute_flux_corr = interp1d(
        flux_cal_data[:,0],
        flux_cal_data[:,1],
        bounds_error=False,
        fill_value=-100.0,
        kind='linear')

    flux_corr = compute_flux_corr(wave)
    inst_fcal_array = 10.0**(-0.4*flux_corr) 

    # Import SSO extinction curve
    ext_data = np.loadtxt(sso_extinction_file)
    extinct_interp = interp1d(
        ext_data[:,0],
        ext_data[:,1],
        bounds_error=False,
        fill_value=np.nan)

    # Calculate extinction for each object
    obj_ext = 10.0**(-0.4*((airmass-1.0)*extinct_interp(wave)))

    # Calculate combined correction factor
    fcal_array = inst_fcal_array*obj_ext

    # Do flux correction
    spec_fluxed = spectra / (fcal_array*exptime*wl_step)
    e_spec_fluxed  = e_spectra / (fcal_array*exptime*wl_step)
    
    return spec_fluxed, e_spec_fluxed


def clean_spectra(spectra, do_set_nan_for_neg_px=False,):
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

        if do_set_nan_for_neg_px:
            spectra[:,1] = np.where(spectra[:,1] <= 0, np.nan, spectra[:,1])


def normalise_spectrum(wl, spectrum, e_spectrum=None, plot_fit=False, 
                       plot_norm=False, mask=None, poly_order=2, wl_min=0,
                       wl_max=10000,):
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
    if mask is None:
        mask = make_wavelength_mask(wl, mask_emission=True)

    mask[ignore] = False

    # And mask out everything below/above min/max wavelength. By default this
    # has no effect as it is broader than the WiFeS bands.
    ignored_wl_mask = np.logical_or(wl < wl_min, wl > wl_max)
    mask[ignored_wl_mask] = False

    # Normalise wavelength scale (pivot about 0)
    wl_norm = (1/wl - 1/lambda_0)*(wl[0]-lambda_0)

    # Fit polynomial to get coefficients
    poly = Polynomial.fit(wl_norm[mask], spectrum_fit[mask], poly_order)

    # Calculate the normalising function and normalise
    norm = poly(wl_norm)

    spectrum_norm = spectrum / np.exp(norm)
    
    # Plot
    if plot_fit:
        fig, ax = plt.subplots()
        ax.plot(wl[:-1], spectrum_fit[:-1], label="flux")
        ax.plot(wl, norm, label="fit")
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (Normalised)")

        import plumage.plotting as pplt
        pplt.shade_excluded_regions(wl, mask, ax, ax, "red", 0.25,)

    elif plot_norm:
        plt.figure()
        plt.plot(wl, spectrum_norm, label="Normalised flux")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Flux (Normalised)")

    # Normalise the uncertainties too if we have been given them
    if e_spectrum is not None:
        e_spectrum_norm = e_spectrum / np.exp(norm)

        assert (len(spectrum_norm) == len(e_spectrum_norm) 
            and len(spectrum_norm) == len(spectrum))

        return spectrum_norm.astype(float), e_spectrum_norm.astype(float)
    
    else:
        return spectrum_norm.astype(float)


def normalise_spectra(wl, spectra, e_spectra=None, poly_order=2, wl_min=0,
    wl_max=10000,):
    """Normalises all spectra

    Parameters
    ----------
    wl: TODO

    spectra: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, value].

    e_spectra: TODO

    normalise_uncertainties: boolean
        Whether to normalise uncertainties (use False if no uncertainties).
    """
    # Initialise
    spectra_norm = spectra.copy()

    if e_spectra is not None:
        e_spectra_norm = e_spectra.copy()

    for spec_i in tqdm(range(len(spectra_norm)), desc="Normalising spectra"):
        # If uncertainties are provided
        if e_spectra is not None:
            spec_norm, e_spec_norm = normalise_spectrum(
                wl,
                spectra[spec_i],
                e_spectra[spec_i],
                poly_order=poly_order,
                wl_min=wl_min,
                wl_max=wl_max,)

            spectra_norm[spec_i] = spec_norm
            e_spectra_norm[spec_i] = e_spec_norm

        # No uncertainties provided (e.g. for template spectra)
        else:
            spec_norm = normalise_spectrum(
                wl,
                spectra[spec_i],
                poly_order=poly_order,
                wl_min=wl_min,
                wl_max=wl_max,)

            spectra_norm[spec_i] = spec_norm

    if e_spectra is not None:
        return spectra_norm, e_spectra_norm
    else:
        return spectra_norm


def norm_spec_by_wl_region(wave, spectrum, arm, e_spec=None):
    """Normalise spectra by a specific wavelength region for synthetic fitting.
    """
    # Normalise by median of wl > 6000
    if arm == "r":
        norm_mask = wave > 6200

    # Normalise median of wl > 4500
    elif arm == "b":
        norm_mask = wave > 4500

    else:
        raise ValueError("Arm must be either r or b")

    # Normalise the specta (and errors if provided)
    norm_fac = np.nanmedian(spectrum[norm_mask])

    spec_norm = spectrum / norm_fac

    if e_spec is not None:
        e_spec_norm = e_spec / norm_fac
        return spec_norm, e_spec_norm

    else:
        return spec_norm


def gaussian_weight_matrix(wl, wl_broadening):
    """Generates a matrix of Gaussian weights given a wavelength vector and
    broadening width.

    Originally written by Dr Anna Ho.
    https://github.com/annayqho/TheCannon/blob/master/TheCannon/normalization.py

    Parameters
    ----------
    wl: numpy ndarray
        pixel wavelength values
    wl_broadening: float
        width of Gaussian in Angstroms.

    Return
    ------
    Weight matrix
    """
    weights = np.exp(-0.5*(wl[:,None]-wl[None,:])**2/wl_broadening**2)

    return weights


def do_gaussian_spectrum_normalisation(
    wl,
    flux,
    ivar,
    adopted_px_mask,
    wl_broadening,):
    """ Returns the weighted mean block of spectra

    Originally written by Dr Anna Ho.
    https://github.com/annayqho/TheCannon/blob/master/TheCannon/normalization.py

    Parameters
    ----------
    wl: numpy ndarray

        wavelength vector
    flux: numpy ndarray

        block of flux values 
    ivar: numpy ndarray

        block of ivar values
    L: float
        width of Gaussian used to assign weights

    Returns
    -------
    smoothed_fluxes: numpy ndarray
        block of smoothed flux values, mean spectra
    """
    # 1) generate Gaussian weights for just the pixels we're using
    useful_wl = wl[adopted_px_mask]
    useful_fluxes = flux[adopted_px_mask]
    useful_ivar = ivar[adopted_px_mask]

    weights = gaussian_weight_matrix(useful_wl, wl_broadening)

    # 2) generate Gaussian normalisation for just good pixels
    denominator = np.dot(useful_ivar, weights.T)
    numerator = np.dot(useful_fluxes*useful_ivar, weights.T)
    bad = denominator == 0
    #cont = np.zeros(numerator.shape)
    continuum_sparse = numerator[~bad] / denominator[~bad]

    # 3) interpolate this for every pixel in our wavelength vector
    continuum_func = interp1d(
        useful_wl,
        continuum_sparse,
        fill_value='extrapolate')

    continuum = continuum_func(wl)

    # 4) normalise
    flux_norm = flux / continuum
    ivar_norm = ivar * continuum**2

    return flux_norm, ivar_norm, continuum


def gaussian_normalise_spectra(
    wl,
    fluxes,
    ivars,
    adopted_wl_mask,
    bad_px_masks,
    wl_broadening,):
    """Gaussian normalises a set of spectra
    """ 
    # Initialise outputs
    fluxes_norm = np.ones_like(fluxes)
    ivars_norm = np.ones_like(fluxes)
    continua = np.ones_like(fluxes)

    tqdm_label = "Gaussian normalising spectra"

    for spec_i in tqdm(range(len(fluxes)), desc=tqdm_label):
        # Make mask
        norm_mask = adopted_wl_mask * ~bad_px_masks[spec_i]
        
        # Run normalisation for each spectrum
        flux_norm, ivar_norm, continuum = do_gaussian_spectrum_normalisation(
            wl=wl,
            flux=fluxes[spec_i],
            ivar=ivars[spec_i],
            adopted_px_mask=norm_mask,
            wl_broadening=wl_broadening,)

        fluxes_norm[spec_i,:] = flux_norm
        ivars_norm[spec_i,:] = ivar_norm
        continua[spec_i,:] = continuum
        
    return fluxes_norm, ivars_norm, continua


def merge_wifes_arms(wl_b, spec_b, wl_r, spec_r):
    """Merge WiFeS arms with proper normalisation, then remove overlap region.
    """
    # Normalise red
    med_mask = wl_r > 6200
    norm_spec_r = spec_r / np.nanmedian(spec_r[med_mask])

    # Get a normalisation factor from the overlap region
    norm_fac_overlap = np.nanmean(norm_spec_r[wl_r < 5445])

    # Normalise blue by overlap
    norm_mask_b = np.logical_and(wl_b > 5400, wl_b < 5445)
    norm_spec_b = spec_b / np.nanmean(spec_b[norm_mask_b]) * norm_fac_overlap

    # Now get rid of overlap region
    overlap_mask = wl_b < 5400
    wl_b = wl_b[overlap_mask]
    norm_spec_b = norm_spec_b[overlap_mask]

    # Combine
    wl_br = np.concatenate((wl_b, wl_r))
    spec_br = np.concatenate((norm_spec_b, norm_spec_r))

    return wl_br, spec_br


def merge_wifes_arms_all(
    wl_b,
    spec_b,
    e_spec_b,
    wl_r,
    spec_r,
    e_spec_r,):
    """Merge WiFeS arms with proper normalisation, then remove overlap region.
    """
    # Normalise red
    med_mask = wl_r > 6200
    med_1d_r = np.nanmedian(spec_r[:,med_mask], axis=1)
    med_2d_r = np.repeat(med_1d_r, len(wl_r)).reshape(spec_r.shape)
    norm_spec_r = spec_r / med_2d_r
    e_norm_spec_r = e_spec_r / med_2d_r

    # Get a normalisation factor from the overlap region
    norm_fac_overlap = np.nanmean(norm_spec_r[:, wl_r < 5445], axis=1)

    # Normalise blue by overlap
    norm_mask_b = np.logical_and(wl_b > 5400, wl_b < 5445)
    med_1d_b = np.nanmean(spec_b[:, norm_mask_b], axis=1) / norm_fac_overlap
    med_2d_b = np.repeat(med_1d_b, len(wl_b)).reshape(spec_b.shape)
    norm_spec_b = spec_b / med_2d_b
    e_norm_spec_b = e_spec_b / med_2d_b

    # Now get rid of overlap region
    overlap_mask = wl_b < 5400
    wl_b = wl_b[overlap_mask]
    norm_spec_b = norm_spec_b[:,overlap_mask]
    e_norm_spec_b = e_norm_spec_b[:,overlap_mask]

    # Combine
    wl_br = np.concatenate((wl_b, wl_r))
    spec_br = np.hstack((norm_spec_b, norm_spec_r))
    e_spec_br = np.hstack((e_norm_spec_b, e_norm_spec_r))

    return wl_br, spec_br, e_spec_br


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

    # Mask the science data by setting uncertainties on the masked regions to
    # infinity (thereby setting the inverse variance to zero), as well as 
    # putting the bad pixels themselves to one so there are no nans involved in
    # calculation of the residuals
    sci_spec_m = sci_spec.copy()
    sci_spec_m[bad_px_mask] = 1
    e_sci_spec_m = e_sci_spec.copy()
    e_sci_spec_m[bad_px_mask] = np.inf

    # Do fit, have initial RV guess be 0 km/s
    rv = 0
    args = (sci_wl, sci_spec_m, e_sci_spec_m, ref_spec_interp, bcor)
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
        sci_wl * (1-(float(rv_fit)-bcor)/(const.c.si.value/1000)))
    infodict["cov"] = cov
    infodict["mesg"] = mesg
    infodict["ier"] = ier

    # Calculate reduced chi^2
    rchi2 = np.nansum(infodict["fvec"]**2) / (len(sci_spec)-len(rv_fit))
    
    return float(rv_fit), float(e_rv), rchi2, infodict


def do_template_match(
    sci_wave,
    sci_spec,
    e_sci_spec,
    bcor,
    ref_params,
    ref_wave,
    ref_spectra,
    arm,
    print_diagnostics=False,
    n_std=3,
    mask_blue_edges=True):
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
            sci_wave,
            mask_emission=False,
            mask_blue_edges=mask_blue_edges)

        # Now that we have our initial mask, OR this with the array of any nan
        # fluxes or uncertainties
        nan_px_mask = np.logical_or(
            ~np.isfinite(sci_spec), 
            ~np.isfinite(e_sci_spec))

        bad_px_mask = np.logical_or(bad_px_mask, nan_px_mask)
        
        # Finally, mask out the final pixel as it is known to be bad 
        # TODO: generalise
        bad_px_mask[-1] = True 

        # Do first fit without any masking beyond tellurics
        rv, e_rv, rchi2, infodict = calc_rv_shift(
            ref_wave,
            ref_spec,
            sci_wave,
            sci_spec,
            e_sci_spec,
            bad_px_mask,
            bcor)

        # Now that we have an initial guess for temperature, the next pass 
        # should also take into account which synthetic pixels are bad. This
        # can only happen here once we have an estimate of the RV
        # TODO: rearrange functions so we don't have circular import
        import plumage.synthetic as synth
        bad_synth_px_mask = synth.make_synth_mask_for_bad_wl_regions(
            sci_wave, rv, bcor, params[0])

        temp_bad_px_mask = np.logical_or(bad_px_mask, bad_synth_px_mask)

        # Now mask out any pixels that have residuals greater than the 
        # threshold value, and fit again. Compute this threshold value *only*
        # from the good (i.e. non masked) pixels.
        resid = infodict["fvec"]
        std = np.std(resid[~temp_bad_px_mask])
        bad_px_mask[np.abs(resid) > (n_std*std)] = True

        # Do second fit, now with masking
        rv, e_rv, rchi2, infodict = calc_rv_shift(
            ref_wave,
            ref_spec,
            sci_wave,
            sci_spec,
            e_sci_spec,
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


def do_all_template_matches(
    sci_wave,
    sci_spectra,
    e_sci_spectra,
    observations,
    ref_params,
    ref_wave,
    ref_spectra,
    arm,
    print_diagnostics=False,
    save_column_ext=""):
    """Do template fitting on all stars for radial velocity and temperature, 
    and save the results to observations.

    Parameters
    ----------
    sci_spectra: float array
        3D numpy array containing spectra of form [N_ob, wl/spec/sigma, flux].
    
    TODO

    TODO

    observations: pandas dataframe
        Dataframe containing information about each observation.

    ref_params: float array
        Array of stellar parameters of form [teff, logg, feh]
    
    ref_spectra: float array
        Array of imported template spectra of form [star, wl/spec, flux]
    
    TODO

    TODO

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

    desc = "Fitting RVs to {} arm".format(arm)

    # For every star, do template fitting
    for star_i in tqdm(range(len(sci_spectra)), desc=desc):
        if print_diagnostics:
            print("\n(%4i/%i) Running fitting on %s:" 
                 % (star_i+1, len(sci_spectra), 
                    observations.iloc[star_i]["id"]))

        bcor = observations.iloc[star_i]["bcor"]
        rv, e_rv, rchi2, nres, params, rchi2_grid, idict = do_template_match(
            sci_wave,
            sci_spectra[star_i],
            e_sci_spectra[star_i],
            bcor,
            ref_params,
            ref_wave,
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

    observations["teff_fit_rv{}".format(save_column_ext)] = all_params[:,0]
    observations["logg_fit_rv{}".format(save_column_ext)] = all_params[:,1]
    observations["feh_fit_rv{}".format(save_column_ext)] = all_params[:,2]
    observations["vsini_fit_rv{}".format(save_column_ext)] = all_params[:,3]
    observations["rv{}".format(save_column_ext)] = np.array(all_rvs)
    observations["e_rv{}".format(save_column_ext)] = np.array(all_e_rvs)
    observations["rchi2_rv{}".format(save_column_ext)] = np.array(all_rchi2)

    return all_nres, all_rchi2_grid, bad_px_masks, info_dicts


def correct_rv(
    wl_old,
    spectrum,
    e_spectrum,
    bcor,
    rv,
    wl_new):
    """Interpolate science spectrum onto new wavelength scale in the rest
    frame. This uses the opposite sign convention of calc_rv_shift_residual.

    Parameters
    ----------
    sci_spectra: float array
        2D numpy array containing spectra of form [wl/spec/sigma, flux].
    
    TODO

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
    calc_spec = interp1d(wl_old, spectrum, kind="linear",
                         bounds_error=False, assume_sorted=True)
    
    calc_sigma = interp1d(wl_old, e_spectrum, kind="linear",
                         bounds_error=False, assume_sorted=True)

    # We're *undoing* the shift imparted by barycentric motion and radial
    # velocity, so this relation will have an opposite sign to the one in
    # calc_rv_shift_residual.
    spec_rf = calc_spec(wl_new * (1+(rv-bcor)/(const.c.si.value/1000)))
    e_spec_rf = calc_sigma(wl_new * (1+(rv-bcor)/(const.c.si.value/1000)))

    #rest_frame_spec = np.stack([wl_new, rest_frame_spec, rest_frame_sigma])

    return spec_rf, e_spec_rf


def correct_all_rvs(
    wl_old,
    spectra,
    e_spectra,
    observations,
    wl_new):
    """
    Parameters
    ----------
    sci_spectra: float array
        3D numpy array containing spectra of form [N_ob, wl/spec/sigma, flux].
    
    TODO

    TODO

    observations: pandas dataframe
        Dataframe containing information about each observation.

    wl_new: float array
        New wavelength scale to regrid the spectra onto once in the rest frame

    Returns
    -------
    rest_frame_spectra: float array
        3D numpy array containing spectra of form [star, wl/spec/sigma, flux] 
        now in the rest frame

    TODO
    """
    #Initialise
    spectra_rf = []
    e_spectra_rf = []

    for star_i, (spectrum, e_spectrum) in enumerate(zip(spectra, e_spectra)):
        bcor = observations.iloc[star_i]["bcor"]
        rv = observations.iloc[star_i]["rv"]
        
        # Perform RV correction
        spec_rf, e_spec_rf = correct_rv(
            wl_old, spectrum, e_spectrum, bcor, rv, wl_new)
        
        # Save results
        spectra_rf.append(spec_rf)
        e_spectra_rf.append(e_spec_rf)

    spectra_rf = np.stack(spectra_rf)
    e_spectra_rf = np.stack(e_spectra_rf)

    return spectra_rf, e_spectra_rf