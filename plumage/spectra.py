"""Code for dealing with observed spectra
"""
import os
import numpy as np 
import pandas as pd
import glob
import pickle
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


def load_all_spectra(spectra_folder="spectra/", ext_snr=1, ext_sci=3):
    """Load in all fits cubes containing 1D spectra to extract both the spectra
    and key details of the observations.

    Parameters
    ----------
    spectra_folder: string
        Root directory of nightly extracted 1D fits cubes.

    ext_snr: int
        The fits extension with non-fluxed 1D spectra to get a measure of SNR.
    
    ext_sci: int
        The fits extension containing the spectra to be used for science.

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

    spectra_b_files = glob.glob(os.path.join(spectra_folder, "*", "*_b.fits"))
    spectra_r_files = glob.glob(os.path.join(spectra_folder, "*", "*_r.fits"))

    spectra_b_files.sort()
    spectra_r_files.sort()

    bad_files = []

    assert len(spectra_b_files) == len(spectra_r_files)

    for fi, (file_b, file_r) in enumerate(zip(tqdm(spectra_b_files), spectra_r_files)):
        # Load in and extract required information from fits files
        with fits.open(file_b) as fits_b, fits.open(file_r) as fits_r:
            # Get the flux and telluric corrected spectra
            spec_b = np.stack(fits_b[ext_sci].data)
            spec_r = np.stack(fits_r[ext_sci].data)

            # Ensure that there is actually signal here. If not, flag the files
            # as bad and skip processing them
            if (len(spec_b[:,1][np.isfinite(spec_b[:,1])]) == 0
                or len(spec_r[:,1][np.isfinite(spec_r[:,1])]) == 0):
                bad_files.append(file_b)
                bad_files.append(file_r)
                continue

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
            exp_time.append(header["EXPTIME"])
            obs_mjd.append(header["MJD-OBS"])
            obs_date.append(header["DATE-OBS"])
            ra.append(header["RA"])
            dec.append(header["DEC"])
            airmass.append(header["AIRMASS"])
        
    # Now combine the arrays into our output structures
    spectra_b = np.stack(spectra_b)
    spectra_r = np.stack(spectra_r)

    # Convert arrays
    snrs_b = np.array(snrs_b).astype(float).astype(int)
    snrs_r = np.array(snrs_r).astype(float).astype(int)
    exp_time = np.array(exp_time).astype(float)
    obs_mjd = np.array(obs_mjd).astype(float)
    airmass = np.array(airmass).astype(float)

    data = [ids, snrs_b, snrs_r, exp_time, obs_mjd, obs_date, ra, dec, airmass]
    cols = ["id", "snr_b", "snr_r", "exp_time", "mjd", "date", "ra", 
            "dec", "airmass"]

    observations = pd.DataFrame(data=np.array(data).T, columns=cols)

    # Print bad filenames
    print("Excluded %i bad (i.e. all nan) files: %s" % 
          (len(bad_files), bad_files))

    return observations, spectra_b, spectra_r


def save_pkl_spectra(observations, spectra_b, spectra_r, rv_corr=False):
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
    """
    # Distinguish whether these spectra are post normalisation and RV corr
    if rv_corr:
        ext = "_rv_corr"
    else:
        ext = ""

    # Get number of obs
    n_obs = len(observations)

    # Save observation log
    ob_out = os.path.join("spectra", "saved_observations_%i%s.pkl" % (n_obs, ext))
    pkl_obs = open(ob_out, "wb")
    pickle.dump(observations, pkl_obs)
    pkl_obs.close()

    # Save blue arm spectra
    sb_out = os.path.join("spectra", "saved_spectra_b_%i%s.pkl" % (n_obs, ext))
    pkl_sb = open(sb_out, "wb")
    pickle.dump(spectra_b, pkl_sb)
    pkl_sb.close()

    # Save red arm 
    sr_out = os.path.join("spectra", "saved_spectra_r_%i%s.pkl" % (n_obs, ext))
    pkl_sr = open(sr_out, "wb")
    pickle.dump(spectra_r, pkl_sr)
    pkl_sr.close()


def load_pkl_spectra(n_obs, rv_corr=False):
    """Load the save spectra and observations from pickle files stored in
    spectra/.

    Parameters
    ----------
    n_obs: int
        The number of observations, tells you which pickles to select.

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
    # Distinguish whether these spectra are post normalisation and RV corr
    if rv_corr:
        ext = "_rv_corr"
    else:
        ext = ""

    # Load observation log
    ob_out = os.path.join("spectra", "saved_observations_%i%s.pkl" % (n_obs, ext))
    pkl_obs = open(ob_out, "rb")
    observations = pickle.load(pkl_obs)
    pkl_obs.close()

    # Load blue arm spectra
    sb_out = os.path.join("spectra", "saved_spectra_b_%i%s.pkl" % (n_obs, ext))
    pkl_sb = open(sb_out, "rb")
    spectra_b = pickle.load(pkl_sb)
    pkl_sb.close()

    # Load red arm spectra
    sr_out = os.path.join("spectra", "saved_spectra_r_%i%s.pkl" % (n_obs, ext))
    pkl_sr = open(sr_out, "rb")
    spectra_r = pickle.load(pkl_sr)
    pkl_sr.close()

    return observations, spectra_b, spectra_r


# -----------------------------------------------------------------------------
# Processing spectra
# -----------------------------------------------------------------------------
def normalise_spectrum(wl, spectrum, e_spectrum=None, show_fit=False):
    """Normalise a single spectrum by a 2nd order polynomial in log space. 
    Automatically detects which WiFeS arm and grating is being used and masks 
    out regions accordingly. Currently only implemented for B3000 and R7000.

    Parameters
    ----------
    wl: float array
        Wavelength scale for the spectrum.

    spectrum: float array
        The spectrum corresponding to wl, units irrelevant.

    show_fit: boolean
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

        #h_alpha = np.logical_and(wl > 6540, wl < 6580)
        edges = np.logical_or(wl < 5420, wl > 6980)
        nan = ~np.isfinite(spectrum_fit)

        #ignore = np.logical_or.reduce((h_alpha, edges, nan))
        ignore = np.logical_or.reduce((edges, nan))
    
    # Blue arm, B3000 grating
    elif np.round(wl.mean()/100)*100 == 4600:
        lambda_0 = 4600
        #ca_hk = np.logical_and(wl > 3920, wl < 3980)
        edges = np.logical_or(wl < 3600, wl > 5400)
        nan = ~np.isfinite(spectrum_fit)

        #ignore = np.logical_or.reduce((ca_hk, edges, nan))
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
    #plt.close("all")
    if show_fit:
        plt.plot(wl[:-1], spectrum_fit[:-1], label="flux")
        plt.plot(wl, norm, label="fit")
    else:
        plt.plot(wl, spectrum_norm, label="Normalised flux")

    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux (Normalised)")
    #plt.ylim([-1, 10])

    # Normalise the uncertainties too if we have been given them
    if e_spectrum is not None:
        e_spectrum_norm = e_spectrum / np.exp(norm)

        return spectrum_norm, e_spectrum_norm
    
    else:
        return spectrum_norm


def normalise_spectra(spectra, normalise_uncertainties=False):
    """Normalises all spectra
    """
    spectra_norm = spectra.copy()

    for spec_i in tqdm(range(len(spectra_norm))):
        #print("(%4i/%i) normalised" % (spec_i+1, len(spectra)))
        if normalise_uncertainties:
            spec_norm, e_spec_norm = normalise_spectrum(spectra[spec_i][0,:], # wl
                                       spectra[spec_i][1,:], # flux
                                       spectra[spec_i][2,:]) # uncertainty

            spectra_norm[spec_i][1,:] = spec_norm
            spectra_norm[spec_i][2,:] = e_spec_norm

        else:
            spec_norm = normalise_spectrum(spectra[spec_i][0,:], # wl
                                       spectra[spec_i][1,:]) # flux
            spectra_norm[spec_i][1,:] = spec_norm
    
    return spectra_norm


def make_wavelength_mask(wave_array, mask_emission=True, 
    mask_blue_edges=False, mask_sky_emission=False):
    """
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
        [5870.0, 6000.0],
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
        [3925.0, 3945.0], # H Eta
        [3960.0, 3980.0], # H Zeta
    ]

    sky_emission = [
        [5575.0, 5585.0],
        [5885.0, 5900.0],
        [6295.0, 6305.0],
        [6360.0, 6370.0],
        [6580.0, 6585.0]
    ]

    band_list = O2_telluric_bands + strong_H2O_telluric_bands

    # Mask out Balmer series
    if mask_emission:
        band_list += balmer_series + calcium_hk

    # In cases of poor sky subtraction, get rid of sky emission
    if mask_sky_emission:
        band_list += sky_emission

    mask = np.ones(len(wave_array))

    for band in band_list:
        mask *= ((wave_array <= band[0])+
                 (wave_array >= band[1]))
    
    # Mask out blue edges and red overlap region
    if mask_blue_edges:
        mask *= ((wave_array >= 3600)+
                 (wave_array <= 5400))

    return mask.astype(bool)


# -----------------------------------------------------------------------------
# Radial Velocities
# -----------------------------------------------------------------------------
def compute_barycentric_correction(ras, decs, times, site="SSO", 
                                   disable_auto_max_age=False):
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

    Returns
    -------
    bcors: astropy.units.quantity.Quantity array
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

    # Calculate the barycentric correction for every star
    for ra, dec, time in zip(tqdm(ras), decs, times):
        sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
        time = Time(float(time), format="mjd")
        barycorr = sc.radial_velocity_correction(obstime=time, location=loc)  
        bcors.append(barycorr.to(u.km/u.s))

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
    loss: float
        The goodness of fit.
    """
    # Shift the template spectrum
    ref_spec = ref_spec_interp(wave * (1-(rv[0]-bcor)/(const.c.si.value/1000)))

    # Setup a mask to use. Ignore edges, and sigma clip *high* pixels to 
    # remove emission features and cosmics
    mask = make_wavelength_mask(wave, mask_emission=True)

    #mask = np.ones_like(ref_spec).astype(bool)
    mask[:10] = False
    mask[-10:] = False
    #nsig = 4
    #mask[spec > (np.nanmean(spec) + nsig*np.nanstd(spec))] = False

    # Return loss
    chi_sq = np.nansum(((spec[mask] - ref_spec[mask]) / e_spec[mask])**2)

    return chi_sq


def calc_rv_shift(ref_wl, ref_spec, sci_wl, sci_spec, e_sci_spec, bcor):
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

    bcor: float
        Barycentric velocity in km/s

    Returns
    -------
    fit, cov, infodict, mesg, ief: various
        Outputs of scipy.optimize.leastsq()
    """
    # Make interpolation function
    ref_spec_interp = ius(ref_wl, ref_spec)

    # Do fit, have initial RV guess be 0 km/s
    rv = 0
    args = (sci_wl, sci_spec, e_sci_spec, ref_spec_interp, bcor.value)
    fit, cov, infodict, mesg, ier = leastsq(calc_rv_shift_residual, rv, 
                                            args=args, full_output=True)

    # Work out the uncertainties
    res = sci_spec - ref_spec_interp(sci_wl * (1-(fit[0]-bcor.value)
                                       /(const.c.si.value/1000)))
    if cov is None:
        e_rv = np.nan
    else:
        e_rv = (cov * np.nanvar(res))**0.5

    return fit[0], float(e_rv), infodict # cov, , mesg, ier


def do_template_match(sci_spectra, bcor, ref_params, ref_spectra,
                      print_diagnostics=False):
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
    rv: float
        Fitted radial velocity in km/s

    teff: float
        Fitted effective temperature in K

    quality: float
        Quality of the fit
    """
    # Fit each template to the science data to figure out the best match
    rvs = []
    e_rvs = []
    grid_chi2 = []

    for params, ref_spec in zip(ref_params, ref_spectra):
        rv, e_rv, infodict = calc_rv_shift(ref_spec[0,:], ref_spec[1,:], 
                                        sci_spectra[0,:], sci_spectra[1,:], 
                                        sci_spectra[2,:], bcor)
        rvs.append(rv)
        e_rvs.append(e_rv)
        grid_chi2.append(infodict["fvec"])

        if print_diagnostics:
            print("\tTeff = %i K, RV = %0.2f km/s, quality = %0.2f" 
                  % (params[0], rv, infodict["fvec"]))

    # Now figure out what best fit is
    fit_i = np.argmin(grid_chi2)

    rv = np.array(rvs[fit_i])
    e_rv = np.array(e_rvs[fit_i])
    params = np.array(ref_params[fit_i])
    chi2 = np.array(grid_chi2[fit_i])

    return rv, e_rv, params, chi2, grid_chi2


def do_all_template_matches(sci_spectra, observations, ref_params, ref_spectra,
                            print_diagnostics=False):
    """Do template fitting on all stars for radial velocity and temperature.

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
    
    print_diagnostics: boolean
        Whether to print diagnostics about the fit

    Returns
    -------
    rvs: float array
        Fitted radial velocity in km/s

    teffs: float array
        Fitted effective temperature in K

    fit_quality: float array
        Quality of the fit
    """
    # Initialise
    all_rvs = []
    all_e_rvs = []
    all_params = []
    all_chi2 = []
    all_chi2_grid = []

    # For every star, do template fitting
    for star_i, sci_spec in enumerate(tqdm(sci_spectra)):
        if print_diagnostics:
            print("\n(%4i/%i) Running fitting on %s:" 
                 % (star_i+1, len(sci_spectra), observations.iloc[star_i]["id"]))

        bcor = observations.iloc[star_i]["bcor"]
        rv, e_rv, params, chi2, grid_chi2 = do_template_match(
            sci_spec, 
            bcor, 
            ref_params, 
            ref_spectra, 
            print_diagnostics)
        
        all_rvs.append(rv)
        all_e_rvs.append(e_rv)
        all_params.append(params)
        all_chi2.append(chi2)
        all_chi2_grid.append(grid_chi2)

    return all_rvs, all_e_rvs, all_params, all_chi2, all_chi2_grid


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
        bcor = observations.iloc[star_i]["bcor"].value
        rv = observations.iloc[star_i]["rv"]
        
        rest_frame_spectra.append(correct_rv(sci_spec, bcor, rv, wl_new))

    rest_frame_spectra = np.stack(rest_frame_spectra)

    return rest_frame_spectra



