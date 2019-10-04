"""
"""
import os
import numpy as np 
import pandas as pd
import glob
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial, polyval


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


def load_all_spectra(spectra_folder="spectra/", ext_snr="08", ext_sci="10"):
    """Load in spectra
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
            sig_b = np.median(fits_b[1].data["spectrum"])
            snrs_b.append(sig_b / sig_b**0.5)

            sig_r = np.median(fits_r[1].data["spectrum"])
            snrs_r.append(sig_r / sig_r**0.5)

            # Get the flux and telluric corrected spectra
            spectra_b.append(np.stack(fits_b[3].data))
            spectra_r.append(np.stack(fits_r[3].data))
        
    # Now combine the arrays into our output structures
    spectra_b = np.stack(spectra_b)
    spectra_r = np.stack(spectra_r)

    data = [ids, snrs_b, snrs_r, exp_time, obs_mjd, obs_date, ra, dec, airmass]
    cols = ["id", "snr_b", "snr_r", "exp_time", "ob_mjd", "ob_date", "ra", 
            "dec", "airmass"]
    observations = pd.DataFrame(data=np.array(data).T, columns=cols)

    return observations, spectra_b, spectra_r


# -----------------------------------------------------------------------------
# Processing spectra
# -----------------------------------------------------------------------------
def normalise_spectra(wl, flux, lambda_0=6200, do_mask=True):
    """
    """
    # Pivot wavelengths about zero
    wl_norm = (1/wl - 1/lambda_0)*(wl[0]-lambda_0)

    # Mask
    mask = np.ones_like(flux)
    h_alpha = np.logical_and(wl > 6540, wl < 6580)
    edges = np.logical_or(wl < 5450, wl > 6950)
    flux_fit = np.log(flux)
    flux_fit[h_alpha] = np.nan
    flux_fit[edges] = np.nan

    #med = np.nanmedian(flux_fit)

    idx = np.isfinite(wl_norm) & np.isfinite(flux_fit)

    # Fit 2nd order polynomial
    poly = Polynomial.fit(wl_norm[idx], flux_fit[idx], 2)
    #poly = Polynomial.fit(wl_norm, flux_fit, 2)

    norm = poly(wl_norm)

    flux_norm = flux / np.exp(norm)

    # Plot
    #plt.close("all")
    #plt.plot(wl[:-1], flux_fit[:-1], label="Normalised flux")
    #plt.plot(wl, norm, label="poly")
    plt.plot(wl, flux_norm, label="Normalised flux")
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux (Normalised)")
    #plt.ylim([0.95,1.05])

    return wl_norm, flux_norm, norm


def compute_barycentric_correction(ra, dec, obs_time, site="SSO"):
    """
    """
    loc = EarthLocation.of_site(site)
    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    barycorr = sc.radial_velocity_correction(obstime=Time(obs_time), 
                                             location=loc)  
    barycorr.to(u.km/u.s)  

def do_template_match(wl, flux_norm, obs_time, obs_loc, obs_coor):
    """
    """



    return teff, rv





