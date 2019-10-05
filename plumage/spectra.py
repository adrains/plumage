"""
"""
import os
import numpy as np 
import pandas as pd
import glob
import pickle
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

    assert len(spectra_b_files) == len(spectra_r_files)

    for fi, (file_b, file_r) in enumerate(zip(spectra_b_files, spectra_r_files)):
        # Load in and extract required information from fits files
        with fits.open(file_b) as fits_b, fits.open(file_r) as fits_r:
            # Get object name and details of observation
            header = fits_b[0].header
            ids.append(header["OBJNAME"])
            exp_time.append(header["EXPTIME"])
            obs_mjd.append(header["MJD-OBS"])
            obs_date.append(header["DATE-OBS"])
            ra.append(header["RA"])
            dec.append(header["DEC"])
            airmass.append(header["AIRMASS"])
            
            print("(%4i/%i) Importing %s on %s" 
                  % (fi+1, len(spectra_b_files), ids[-1], obs_date[-1]))

            # Get SNR measurements for each arm
            sig_b = np.median(fits_b[ext_snr].data["spectrum"])
            snrs_b.append(sig_b / sig_b**0.5)

            sig_r = np.median(fits_r[ext_snr].data["spectrum"])
            snrs_r.append(sig_r / sig_r**0.5)

            # Get the flux and telluric corrected spectra
            spec_b = np.stack(fits_b[ext_sci].data)
            spectra_b.append(spec_b.T)
            spec_r = np.stack(fits_r[ext_sci].data)
            spectra_r.append(spec_r.T)
        
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

    return observations, spectra_b, spectra_r


def save_pkl_spectra(observations, spectra_b, spectra_r):
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
    # Get number of obs
    n_obs = len(observations)

    # Save observation log
    ob_out = os.path.join("spectra", "saved_observations_%i.pkl" % n_obs)
    pkl_obs = open(ob_out, "wb")
    pickle.dump(observations, pkl_obs)
    pkl_obs.close()

    # Save blue arm spectra
    sb_out = os.path.join("spectra", "saved_spectra_b_%i.pkl" % n_obs)
    pkl_sb = open(sb_out, "wb")
    pickle.dump(spectra_b, pkl_sb)
    pkl_sb.close()

    # Save red arm 
    sr_out = os.path.join("spectra", "saved_spectra_r_%i.pkl" % n_obs)
    pkl_sr = open(sr_out, "wb")
    pickle.dump(spectra_r, pkl_sr)
    pkl_sr.close()


def load_pkl_spectra(n_obs):
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
    # Load observation log
    ob_out = os.path.join("spectra", "saved_observations_%i.pkl" % n_obs)
    pkl_obs = open(ob_out, "rb")
    observations = pickle.load(pkl_obs)
    pkl_obs.close()

    # Load blue arm spectra
    sb_out = os.path.join("spectra", "saved_spectra_b_%i.pkl" % n_obs)
    pkl_sb = open(sb_out, "rb")
    spectra_b = pickle.load(pkl_sb)
    pkl_sb.close()

    # Load red arm spectra
    sr_out = os.path.join("spectra", "saved_spectra_r_%i.pkl" % n_obs)
    pkl_sr = open(sr_out, "rb")
    spectra_r = pickle.load(pkl_sr)
    pkl_sr.close()

    return observations, spectra_b, spectra_r


# -----------------------------------------------------------------------------
# Processing spectra
# -----------------------------------------------------------------------------
def normalise_spectra(wl, spectrum, show_fit=False):
    """Normalise spectra by a 2nd order polynomial in log space. Automatically
    detects which WiFeS arm and grating is being used and masks out regions
    accordingkly. Currently only implemented for B3000 and R7000.

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

    # Red arm, R7000 grating
    if np.round(wl.mean()/100)*100 == 6200:
        lambda_0 = 6200

        h_alpha = np.logical_and(wl > 6540, wl < 6580)
        edges = np.logical_or(wl < 5450, wl > 6950)
        nan = ~np.isfinite(spectrum_fit)

        ignore = np.logical_or.reduce((h_alpha, edges, nan))
    
    # Blue arm, B3000 grating
    elif np.round(wl.mean()/100)*100 == 4600:
        lambda_0 = 4600
        ca_hk = np.logical_and(wl > 3920, wl < 3980)
        edges = np.logical_or(wl < 4000, wl > 4650)
        nan = ~np.isfinite(spectrum_fit)

        ignore = np.logical_or.reduce((ca_hk, edges, nan))

    else:
        raise Exception("Grating not recognised or implemented. Must be"
                        " either B3000 or R7000.")

    # Make the mask
    mask = np.ones_like(spectrum).astype(bool)
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

    return spectrum_norm


# -----------------------------------------------------------------------------
# Radial Velocities
# -----------------------------------------------------------------------------
def compute_barycentric_correction(ras, decs, times, site="SSO"):
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

    # Calculate the barycentric correction for every star
    for ra, dec, time in zip(ras, decs, times):
        sc = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
        time = Time(float(time), format="mjd")
        barycorr = sc.radial_velocity_correction(obstime=time, location=loc)  
        bcors.append(barycorr.to(u.km/u.s))

    return bcors


def calc_rv_shif_residual(rv, wave, spec, e_spec, ref_spec_interp, bcor):
    """
    """
    # Shift the template spectrum
    ref_spec = ref_spec_interp(wave * (1 - (rv+bcor)/const.c.si.value))

    # Return loss
    return np.sum((ref_spec - spec)**2 / 2)


def calc_rv_shift(ref_wl, ref_spec):
    """
    """
    # Make interpolation function
    ref_spec_interp = ius(ref_wl, ref_spec)

    # Do fit
    rv = 0
    args = (wave, spec, e_spec, ref_spec_interp, bcor)
    fit = leastsq(calc_rv_shif_residual, rv, args=args)



def do_template_match(wl, flux_norm, obs_time, obs_loc, obs_coor):
    """
    """



    return teff, rv





