"""
Create a mask for an M dwarf star: regions to include in the fit

marusa@mash:/priv/mulga1/marusa>source env/bin/activate
"""
from __future__ import division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import pidly
import pandas as pd


class ParameterOutOfBoundsException(Exception):
    pass

#------------------------------------------------------------------------------
# Properties
#------------------------------------------------------------------------------
using_echelle = False
norm = 1

# Echelle dispersers:
# - 300/300nm --> 300-500nm
# - 316/750nm --> 500-1,000nm
if using_echelle:
    resolution = 24000
    wl_min = 3800
    wl_max = 6800
    
    n_pixels = 40000
    wl_per_pixel = (wl_max - wl_min) / 40000
    
# Using WiFeS
else:
    # Red
    resolution = 24000
    wl_min = 5000
    wl_max = 10000
    
    # Blue
    resolution = 24000
    wl_min = 5000
    wl_max = 10000
#------------------------------------------------------------------------------    
# Setup IDL
def idl_init():
    """Initialise IDL by setting paths and compiling relevant files.
    """
    idl = pidly.IDL()
    idl("!path = '/home/thomasn/idl_libraries/coyote:' + !path")
    idl(".compile /home/thomasn/grids/gaussbroad.pro")
    idl(".compile /home/thomasn/grids/get_spec.pro")
    idl("grid='/home/thomasn/grids/grid_synthspec.sav'")
    
    return idl
    
def get_idl_spectrum(idl, teff, logg, feh, wl_min, wl_max, resolution, norm=1,
                     do_resample=False, wl_per_pixel=None):
    """
    Parameters
    ----------
    idl: pidly.IDL
        IDL wrapper.
    teff: int
        Temperature of the star in K.
    logg: float
        Log base 10 surface gravity of the star in cgs units.
    feh: float
        Metallicity of the star relative to Solar, [Fe/H].  
    wl_min: int
        Minimum wavelength in Angstroms.
    wl_max: int
        Maximum wavelenth in Angstroms.
    resolution: int
        Spectral resolution.
    norm: int 
        0: absolute flux, i.e. the normalised flux is multiplied by the 
           absolute continuum flux
        1: normalised flux only
        2: continuum flux only
        -1: absolute flux, normalised to the central-wavelength absolute flux
        large values: absolute flux, normalised to the wavelength "norm"
        
    Returns
    -------
    
    """
    if teff > 8000 or teff < 2500:
        raise ParameterOutOfBoundsException("Temperature must be 2500 <="
                                            " Teff (K) <= 8000")
    elif logg > 5.5 or logg < -1:
        raise ParameterOutOfBoundsException("Surface gravity must be -1 <="
                                            " logg (cgs) <= 5.5")
    elif feh > 1.0 or feh < -5:
        raise ParameterOutOfBoundsException("Metallicity must be -5 <="
                                            " [Fe/H] (dex) <= 1")
    elif wl_min > wl_max or wl_max > 200000 or wl_min < 2000:
        raise ParameterOutOfBoundsException("Wavelengths must be 2,000 <="
                                            " lambda (A) <= 60,000")                
    
    idl("CFe = 0. ;")
    cmd = ("spectrum = get_spec(%d, %f, %f, !null, CFe, %i, %i, ipres=%i, "
           "norm=%i, grid=grid, wave=wave)" % (teff, logg, feh, wl_min, wl_max, 
                                               resolution, norm))
    
    idl(cmd)
    
    if do_resample:
        idl("waveout = [%i+%f:%i-2*%f:%f]" % (wl_min, wl_per_pixel, wl_max, wl_per_pixel, wl_per_pixel))
        idl("spectrum = resamp(double(wave), double(spectrum), double(waveout))")
        idl("wave = waveout")
    
    return idl.wave, idl.spectrum

# Retrieve list of standards    
def retrieve_standards(idl):
    """Get spectra for standards
    """

    standards =  pd.read_csv("standards.tsv", sep="\t", header=0, 
                             dtype={"source_id":str})
                         
    mask = (standards["teff"] < 5500) * (standards["logg"] > 4.0)
    training_set = standards[mask][["teff","logg","feh"]]

    spectra = []

    idl = idl_init()

    standards = standards[mask].copy()

    for star_i, row in standards.iterrows():
        print(star_i)
        wave, spec = get_idl_spectrum(idl, row["teff"], row["logg"], row["feh"], 
                                      wl_min, wl_max, resolution, 1, True, 
                                      wl_per_pixel)
                                  
        spectra.append(spec)
    
    spectra = np.array(spectra) 
    normalized_ivar = np.ones_like(spectra) * 0.01   
    np.savetxt("spectra_standards.csv", spectra)
    np.savetxt("spectra_wavelengths.csv", wave)

    import thecannon as tc

    vectorizer = tc.vectorizer.PolynomialVectorizer(("teff", "logg", "feh"), 2)
    model = tc.CannonModel(training_set, spectra, normalized_ivar,
                           vectorizer=vectorizer)


def get_idl_spectra(idl, teffs, loggs, fehs, wl_min, wl_max, resolution, norm,
                    do_resample, wl_per_pixel):
    """Call get_idl_spectrum multiple times
    """
    spectra = []
    
    for star_i, (teff, logg, feh) in enumerate(zip(teffs, loggs, fehs)):
        print("Star %i, [%i, %0.2f, %0.2f]" % (star_i, teff, logg, feh))
        wave, spec = get_idl_spectrum(idl, teff, logg, feh, wl_min, wl_max, 
                                      resolution, norm, True, wl_per_pixel)
        spectra.append(spec)
    
    spectra = np.array(spectra)
    return wave, spectra


def save_spectra(wave, spectra):
    """
    """
    np.savetxt("sample_spectra.csv", spectra)
    np.savetxt("sample_wavelengths.csv", wave)
    
    
def plot_all_spectra(wave, spectra, teffs, loggs, fehs):
    """Plot a grid of spectra
    """
    plt.close("all")
    
    for star_i, (teff, logg, feh) in enumerate(zip(teffs, loggs, fehs)):
        lbl = "[%i, %0.2f, %0.2f]" % (teff, logg, feh)
        plt.plot(wave, spectra[star_i], label=lbl, alpha=0.9, linewidth=0.5)
        
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Normalised Flux")
    leg = plt.legend(loc="best")
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

# -----------------------------------------------------------------------------
# Template spectra
# -----------------------------------------------------------------------------\
def get_template_spectra(teffs, loggs, fehs, setting="R7000"):
    """Creates synthetic spectra in a given grating format to serve as RV
    templates.

    Parameters
    ----------
    teffs: float array
        Stellar temperatures in Kelvin.
    
    loggs: float array
        Stellar surface gravities in cgs units.

    fehs: float array
        Stellar metallicities relative to Solar.

    setting: string
        The grating setting to generate the spectra for. Currently only R7000
        and B3000 are supported.

    Returns
    -------
    spectra: 3D float array
        The synthetic spectra in the form [N_star, wl/flux, pixel]. The stars
        are ordered by teff, then logg, then [Fe/H].
    """
    # Get the spectrograph settings
    if setting == "R7000":
        wl_min = 5400
        wl_max = 7000
        resolution = 7000
        n_px = 3637

    elif setting == "B3000":
        wl_min = 3500
        wl_max = 5700
        resolution = 3000
        n_px = 2858

    else:
        raise Exception("Unknown grating setting.")

    # Calculate the wavelength spacing
    wl_per_pixel = (wl_max - wl_min) / n_px 

    # Load in the IDL object
    idl = idl_init()

    spectra = []

    for teff in teffs:
        for logg in loggs:
            for feh in fehs:
                wave, spec = get_idl_spectrum(idl, teff, logg, feh, wl_min, 
                                              wl_max, resolution, norm=0,
                                              do_resample=True, 
                                              wl_per_pixel=wl_per_pixel)
                
                spectra.append((wave,spec))
    
    return np.stack(spectra)


def save_synthetic_templates(spectra, teffs, loggs, fehs, setting="R7000"):
    """Save the generated synthetic templates in templates/.

    Parameters
    ----------
    spectra: 3D float array
        The synthetic spectra in the form [N_star, wl/flux, pixel]. The stars
        are ordered by teff, then logg, then [Fe/H].

    teffs: float array
        Stellar temperatures in Kelvin.
    
    loggs: float array
        Stellar surface gravities in cgs units.

    fehs: float array
        Stellar metallicities relative to Solar.

    setting: string
        The grating format of the spectra.
    """
    spec_i = 0
    params = []

    # For all params, save a CSV file
    for teff in teffs:
        for logg in loggs:
            for feh in fehs:
                fname = "template_%s_%i_%0.2f_%0.2f.csv" % (setting, teff, 
                                                            logg, feh)
                path = os.path.join("templates", fname)

                np.savetxt(path, spectra[spec_i,:,:].T)

                params.append([teff, logg, feh])

                spec_i += 1

    # Now save a file keeping track of the params of each star
    params_file = "template_%s_params.csv" % setting
    params_path = os.path.join("templates", params_file)

    np.savetxt(params_path, params, fmt=["%i", "%0.1f", "%0.1f"])
