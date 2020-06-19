"""Code for working with synthetic spectra.
"""
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import pidly
import pandas as pd
from tqdm import tqdm
import astropy.constants as const
from astropy import units as u
from scipy.optimize import leastsq, least_squares
import plumage.spectra as spec
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import LinearNDInterpolator

#------------------------------------------------------------------------------    
# Setup IDL
#------------------------------------------------------------------------------
def idl_init(grid_choice="full"):
    """Initialise IDL by setting paths and compiling relevant files.
    """
    if grid_choice == "full":
        grid = "grid_synthspec_main.sav"
    elif grid_choice == "R~3000":
        grid = "grid_synthspec_main-R3k.sav"
    elif grid_choice == "R~7000":
        grid = "grid_synthspec_main-R7k.sav"
    else:
        raise ValueError("Invalid grid choice - must be 'full', 'R~3000', or"
                        "R~7000")

    idl = pidly.IDL()
    idl("!path = '/home/thomasn/idl_libraries/coyote:' + !path")
    idl(".compile /home/thomasn/grids/gaussbroad.pro")
    idl(".compile /home/thomasn/grids/get_spec.pro")
    idl("grid='/home/thomasn/grids/%s'" % grid)
    
    return idl
    
def get_idl_spectrum(idl, teff, logg, feh, wl_min, wl_max, ipres, 
                     resolution=None, norm="abs", do_resample=True, 
                     wl_per_px=None, rv_bcor=0, wave_pad=50, abund_alpha=None,
                     abund_CFe=None,):
    """Calls Thomas Nordlander's IDL routines to generate a synthetic MARCS
    model spectrum at the requested parameters.

    Parameters
    ----------
    idl: pidly.IDL
        IDL wrapper

    teff: int
        Temperature of the star in K

    logg: float
        Log base 10 surface gravity of the star in cgs units

    feh: float
        Metallicity of the star relative to Solar, [Fe/H]
    
    wl_min: float
        Minimum wavelength in Angstroms

    wl_max: float
        Maximum wavelenth in Angstroms

    ipres: float
        Spectral resolution for constant-velocity broadening. Note that this is
        a mutually exclusive option to wavelength-constant broadening. If a 
        value is provided for resolution, will use wavelength-constant 
        broadening.

    resolution: float, default: None
        Width of broadening kernel in Angstroms for wavelength-constant 
        broadening. Note that this is a mutually exclusive option to constant-
        velocity broadening set by ipres, and will run if resolution is not
        None.

    norm: str, default: 'abs'
        'abs': absolute flux, i.e. the normalised flux is multiplied by the 
           absolute continuum flux
        'norm': normalised flux only
        'cont': continuum flux only
        'abs_norm': absolute flux, normalised to the central-wavelength 
            absolute flux large values: absolute flux, normalised to the  
            wavelength "norm"
    
    do_resample: bool, default: True
        Whether to resample onto custom wavelength scale.

    wl_per_px: bool, default: None
        Wavelength per pixel of new wavelength grid.

    rv_bcor: float, default: 0
        The velocity offset (km/s) for the synthetic spectrum, 0 by default. If 
        matching observations, this should be the total velocity offset, that
        is both the RV and barcentric offset.

    wave_pad: float, default: 50
        Padding in Angstroms added to either end of the requested wavelength 
        scale to account for the initial wavelength limits being applied before
        RV shifting and without interpolation. 50 A by default, as a star would
        need to have an RV of 2,000+ km/s to shift outside this.

    abund_alpha, abund_CFe: float or None, default: None
        Alpha element and [C/Fe] abundance respectively, uses Nordlander IDL
        defaults when set to None.

    Returns
    -------
    wave: 1D float array
        The wavelength scale for the synthetic spectra

    spectra: 1D float array
        Fluxes for the synthetic spectra.

    """
    NORM_VALS = {"abs":0, "norm":1, "cont":2, "abs_norm":-1}

    # Checks
    if teff > 8000 or teff < 2500:
        raise ValueError("Temperature must be 2500 <= Teff (K) <= 8000")

    if logg > 5.5 or logg < -1:
        raise ValueError("Surface gravity must be -1 <= logg (cgs) <= 5.5")

    if feh > 1.0 or feh < -5:
        raise ValueError("Metallicity must be -5 <= [Fe/H] (dex) <= 1")

    if wl_min > wl_max or wl_max > 200000 or wl_min < 2000:
        raise ValueError("Wavelengths must be 2,000 <= lambda (A) <= 60,000")

    if norm not in NORM_VALS:
        raise ValueError("Invalid normalisation value, see NORM_VALS.")
    
    norm_val = NORM_VALS[norm]

    # Initialise abundance parameters
    if abund_alpha is None:
        idl("alpha = !null;")
    else:
        idl("alpha = {};".format(abund_alpha))

    if abund_CFe is None:
        idl("CFe = 0.0;")
    else:
        idl("CFe = {};".format(abund_CFe))

    # If resampling, initialise the output wavelength scale
    if do_resample:
        idl("wout = [%f:%f:%f]" % (wl_min, wl_max, wl_per_px))
        wout = ", wout=wout"

        # Incorporate padding *after* we've set our output scale
        wl_min -= wave_pad
        wl_max += wave_pad
    else:
        wout = ""

    # Default behaviour is constant-velocity broadening, unless a value has 
    # been provided for resolution
    if resolution is None:
        cmd = ("spectrum = get_spec(%f, %f, %f, alpha, CFe, %f, %f, ipres=%f, "
               "norm=%i, grid=grid, wave=wave, vrad=%f%s)"
               % (teff, logg, feh, wl_min, wl_max, ipres, norm_val, rv_bcor, 
                  wout))
    # Otherwise do constant-wavelength broadening
    else:
        cmd = ("spectrum = get_spec(%f, %f, %f, alpha, CFe, %f, %f, norm=%i, "
               "resolution=%f, grid=grid, wave=wave, vrad=%f%s)" 
               % (teff, logg, feh, wl_min, wl_max, norm_val, resolution, 
                  rv_bcor, wout))
    # Run
    idl(cmd)
    
    if do_resample:
        return idl.wout, idl.spectrum
    else:
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
                                      wl_per_px)
                                  
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
                    do_resample, wl_per_px):
    """Call get_idl_spectrum multiple times
    """
    spectra = []
    
    for star_i, (teff, logg, feh) in enumerate(zip(teffs, loggs, fehs)):
        print("Star %i, [%i, %0.2f, %0.2f]" % (star_i, teff, logg, feh))
        wave, spec = get_idl_spectrum(idl, teff, logg, feh, wl_min, wl_max, 
                                      resolution, norm, True, wl_per_px)
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
# -----------------------------------------------------------------------------
def make_synth_mask_for_bad_wl_regions(wave, rv, bcor, teff, cutoff_temp=4000):
    """Makes a bad pixel mask corresponding to where Thomas Nordlander's 
    synthetic MARCS models are inconsistent with observational data at cooler
    temperatures. The wavelength is corrected to the rest frame before 
    excluding.

    Parameters
    ----------
    wave: float array
        The wavelength scale in Angstroms, uncorrected for RV
    
    rv: float
        The measured RV in km/s.

    bcor: float
        The measured barycentric correction in km/s

    teff: float
        The estimated temperature of the star in Kelvin

    cutoff_temp: float
        The temperature in Kelvin below which we exlclude ill-fitting synthetic
        spectral regions

    Returns
    -------
    bad_reg_mask: bool array
        Bad pixel mask that is True where Thomas Nordlander's synthetic MARCS
        models are inconsistent with observed spectra. Array of False if teff
        is above cutoff temp
    """
    # Initialise mask for bad regions
    bad_reg_mask = np.zeros_like(wave)

    # Don't mask anything if we're above the cutoff temperature
    if teff > cutoff_temp:
        return bad_reg_mask.astype(bool)

    # List of badly fitting regions for Thomas Nordlander's MARCS model library
    # These wavelengths are in the *rest frame*
    bad_regions = [
        [0, 4700],      # Blue
        [5498, 5585],   # ???
        [5615, 5759],   # ~ TiO
        [5809, 5840],   # ~ TiO
        [5847, 5886],   # Na doublet blue wing
        [5898, 5987],   # Na doublet red wing
        [6029, 6159],]  # ~ Ca/Fe molecular line or photodissociation opacity
    
    # Shift wavelength vector to rest frame
    wave_rf = wave * (1-(rv-bcor)/(const.c.si.value/1000))

    for region in bad_regions:
        bad_reg = np.logical_and(wave_rf >= region[0], wave_rf <= region[1])
        bad_reg_mask = np.logical_or(bad_reg_mask, bad_reg)

    return bad_reg_mask.astype(bool)
    

def get_template_spectra(teffs, loggs, fehs, vsinis=[1], setting="R7000",
                         norm="abs"):
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

    vsinis: float array
        Stellar vsini in km/s. Defaults to [1], which means the rotational
        velocity is unresolved.

    setting: string
        The grating setting to generate the spectra for. Currently only R7000
        and B3000 are supported.

    norm: str 
        'abs': absolute flux, i.e. the normalised flux is multiplied by the 
           absolute continuum flux
        'norm': normalised flux only
        'cont': continuum flux only
        'abs_norm': absolute flux, normalised to the central-wavelength 
            absolute flux large values: absolute flux, normalised to the  
            wavelength "norm"

    Returns
    -------
    wave: 1D float array
        The wavelength scale for the synthetic spectra

    spectra: 3D float array
        The synthetic spectra in the form [N_star, wl/flux, pixel]. The stars
        are ordered by teff, then logg, then [Fe/H].

    params: 3D float array
        Corresponding stellar parameters for the synthetic spectra.
    """
    # Get the spectrograph settings
    if setting == "R7000":
        wl_min = 5400
        wl_max = 7000
        resolution = 7000
        n_px = 3637
        wl_per_px = 0.44

    elif setting == "B3000":
        wl_min = 3500
        wl_max = 5700
        resolution = 3000
        n_px = 2858
        wl_per_px = 0.77

    else:
        raise ValueError("Unknown grating - choose either B3000 or R7000")
    
    # Determine the effective resolution based on vsini
    eff_rs = []

    for vsini in vsinis:
        eff_r = const.c.to("km/s").value / vsini

        # Take whichever the lower resolution is for the band in question
        if eff_r < resolution:
            eff_rs.append(eff_r)
        else:
            eff_rs.append(resolution)

    # Load in the IDL object
    idl = idl_init()

    spectra = []
    params = []

    for teff in teffs:
        for logg in loggs:
            for feh in fehs:
                for eff_r in eff_rs:
                    wave, spec = get_idl_spectrum(
                        idl, 
                        teff, 
                        logg, 
                        feh, 
                        wl_min, 
                        wl_max, 
                        eff_r, 
                        norm=norm,
                        do_resample=True, 
                        wl_per_px=wl_per_px)
                
                    spectra.append((spec))

                    params.append([teff, logg, feh, vsini])
    
    spectra = np.stack(spectra)
    params = np.stack(params)

    # Only return spectra from valid regions of the parameter space
    valid_i = np.sum(spectra, axis=1) != 0

    return wave, spectra[valid_i], params[valid_i]


def make_BR_WiFeS_synthetic_spectra(teffs, loggs, fehs, vsinis=[1], 
                                    norm="abs"):
    """Wrapper function around get_template_spectra() to simulate the full
    WiFeS B3000 and R7000 wavelength range. Independently gets both bands,
    then stitches together at 5400 A.

    Parameters
    ----------
        teffs: float array
        Stellar temperatures in Kelvin.
    
    loggs: float array
        Stellar surface gravities in cgs units.

    fehs: float array
        Stellar metallicities relative to Solar.

    vsinis: float array
        Stellar vsini in km/s. Defaults to [1], which means the rotational
        velocity is unresolved.

    norm: str 
        'abs': absolute flux, i.e. the normalised flux is multiplied by the 
           absolute continuum flux
        'norm': normalised flux only
        'cont': continuum flux only
        'abs_norm': absolute flux, normalised to the central-wavelength 
            absolute flux large values: absolute flux, normalised to the  
            wavelength "norm"
    """
    n_spec = len(teffs) * len(loggs) * len(fehs) * len(vsinis)
    print("Getting %i B3000 spectra..." % n_spec)
    wl_b, spec_b, params = get_template_spectra(teffs, loggs, fehs, 
                                                vsinis, "B3000", norm)
    print("\nGetting %i R7000 spectra..." % n_spec)
    wl_r, spec_r, params = get_template_spectra(teffs, loggs, fehs, 
                                                vsinis, "R7000", norm)
    blue_mask = wl_b < 5400
    wl_br = np.concatenate((wl_b[blue_mask], wl_r))
    spec_br = np.concatenate((spec_b[:,blue_mask], spec_r), axis=1)

    return wl_br, spec_br, params


def save_synthetic_templates(wave, spectra, params, label):
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
    n_spec = len(params)

    # Save a csv for wavelength scale
    wave_file = "template_wave_%i_%s.csv" % (n_spec, label)
    wave_path = os.path.join("templates", wave_file)
    np.savetxt(wave_path, wave)
    
    # Save a csv for the spectra
    spec_file = "template_spectra_%i_%s.csv" % (n_spec, label)
    spec_path = os.path.join("templates", spec_file)
    np.savetxt(spec_path, spectra)

    # Now save a file keeping track of the params of each star
    params_file = "template_params_%i_%s.csv" % (n_spec, label)
    params_path = os.path.join("templates", params_file)
    np.savetxt(params_path, params, fmt=["%i", "%0.2f", "%0.2f", "%i"])


def load_synthetic_templates(label):
    """Load in the saved synthetic templates

    Parameters
    ----------
    label: string
        The grating setting determining which templatesd to import.

    Returns
    -------
    params: float array
        Array of stellar parameters of form [teff, logg, feh]
    
    templates: float array
        Array of imported template spectra of form [star, wl/spec, flux]
    """
    # Load wavelength scale
    wave_file = "template_wave_%s.csv" % label
    wave_path = os.path.join("templates", wave_file)
    wave = params = np.loadtxt(wave_path)
    
    # Load spectra
    spec_file = "template_spectra_%s.csv" % label
    spec_path = os.path.join("templates", spec_file)
    spectra = params = np.loadtxt(spec_path)

    # Load params of each star
    params_file = "template_params_%s.csv" % label
    params_path = os.path.join("templates", params_file)
    params = np.loadtxt(params_path)

    return wave, spectra, params

def load_synthetic_templates_legacy(path, setting="R7000"):
    """Load in the saved synthetic templates, but in the old case where the
    synthetic spectra are in individual csvs. 

    Parameters
    ----------
    setting: string
        The grating setting determining which templatesd to import.
    Returns
    -------
    params: float array
        Array of stellar parameters of form [teff, logg, feh]
    
    templates: float array
        Array of imported template spectra of form [star, wl/spec, flux]
    """
    # Load in the synthetic star params
    params_file = os.path.join(path, "template_%s_params.csv" % setting)

    params = np.loadtxt(params_file)

    templates = []

    for param in tqdm(params):
        tfile = os.path.join(path, "template_%s_%i_%0.2f_%0.2f_%i.csv"
                             % (setting, param[0], param[1], param[2], 
                                param[3]))
        spec = np.loadtxt(tfile)
        templates.append(spec.T)

    return params, np.stack(templates)

# -----------------------------------------------------------------------------
# Synthetic Fitting
# -----------------------------------------------------------------------------
def calc_synth_fit_resid_one_arm(
    teff,
    logg,
    feh, 
    wave_sci, 
    spec_sci, 
    e_spec_sci, 
    bad_px_mask,
    rv, 
    bcor, 
    idl,
    band_settings,):
    """Calculates residuals between observed science spectrum, and MARCS model
    synthetic spectrum for a single arm of the spectrograph.

    Parameters
    ----------
    teff, logg, feh: float
        Stellar parameters for star.

    wave_sci: float array
        Wavelength scale corresponding to spec_sci.

    spec_sci: float array
        The science spectrum corresponding to wave_sci.

    e_spec_sci: float array
        Uncertainties on science spectrum.

    bad_px_mask: boolean array
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    rv: float
        Radial velocity in km/s

    bcor: float:
        Barycentric velocity in km/s

    idl: pidly.IDL
        pidly.IDL object to interface with Thomas Nordlander's synthetic 
        MARCS spectra through IDL.

    band_settings: dict
        Dictionary with settings for WiFeS band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    Returns
    -------
    resid_vect: float array
        Uncertainty weighted residual vector between science and synthetic
        spectra.

    spec_synth: float array
        Best fitting synthetic spectrum.
    """
    # Get the template spectrum
    wave_synth, spec_synth = get_idl_spectrum(
        idl, 
        teff, 
        logg, 
        feh, 
        band_settings["wl_min"], 
        band_settings["wl_max"], 
        ipres=band_settings["inst_res_pow"],
        resolution=band_settings["wl_per_px"],
        norm="abs",
        do_resample=True, 
        wl_per_px=band_settings["wl_broadening"],
        rv_bcor=(rv-bcor),
        )

    # Normalise spectra by wavelength region before fit
    spec_sci, e_spec_sci = spec.norm_spec_by_wl_region(
        wave_sci, 
        spec_sci, 
        band_settings["arm"], 
        e_spec_sci)

    # Normalise the synthetic spectrum only if it is non-zero. An array of 
    # zeros means that we generated a synthetic spectrum at non-physical values
    # and returned with zero flux, but our parameters were still inside the 
    # grid bounds. If it is composed of zeros, the residuals should be bad 
    # anyway.
    if np.sum(spec_synth) > 0:
        spec_synth = spec.norm_spec_by_wl_region(
            wave_sci, 
            spec_synth, 
            band_settings["arm"])

    # Mask the spectra by setting uncertainties on bad pixels to infinity 
    # (thereby setting the inverse variances to zero), as well as putting the
    # bad pixels themselves to one so there are no nans involved in calculation
    # of the residuals (as this breaks the least squares fitting)
    spec_sci_m = spec_sci.copy()
    spec_sci_m[bad_px_mask] = 1
    e_spec_sci_m = e_spec_sci.copy()
    e_spec_sci_m[bad_px_mask] = np.inf

    # Calculate the residual
    resid_vect = (spec_sci_m - spec_synth) / e_spec_sci_m

    return resid_vect, spec_synth


def calc_colour_resid(
    teff,
    logg,
    feh,
    stellar_colours,
    e_stellar_colours,
    colour_bands,
    sc_interp,):
    """Calculates the residuals between observed and synthetic colours.

    Parameters
    ----------
    teff, logg, feh: float
        Stellar parameters for star.
    
    stellar_colours: float array, default: None
        Array of observed stellar colour corresponding to colour_bands. If None
        photometry is not used in the fit.

    e_stellar_colours: float array, default: None
        Array of observed stellar colour uncertainties. If None photometry is 
        not used in the fit.

    colour_bands: string array, default: ['Rp-J', 'J-H', 'H-K']
        Colour bands to use in the fit.

    sc_interp: SyntheticColourInterpolator
        SyntheticColourInterpolator object able to interpolate synth colours.

    Returns
    -------
    resid: float array
        Uncertainty weighted residual vector between observed and synthetic
        colours.
    """
    # Input checking
    if (len(stellar_colours) != len(e_stellar_colours) 
        or len(e_stellar_colours) != len(colour_bands)):
        raise ValueError("stellar_colours, e_stellar_colours, and colour_bands"
                         "  should all have the same length")

    # Calculate a set of synthetic colours
    synth_colours = np.array([sc_interp.compute_colour((teff,logg,feh),cb) 
                              for cb in colour_bands])

    # Calculate the residuals
    resid = (stellar_colours - synth_colours) / e_stellar_colours

    return resid


def calc_synth_fit_resid_both_arms(
    params, 
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r,
    rv, 
    bcor, 
    idl,
    band_settings_r, 
    logg, 
    band_settings_b,
    wave_b, 
    spec_b, 
    e_spec_b, 
    bad_px_mask_b,
    stellar_colours,
    e_stellar_colours,
    colour_bands,
    sc_interp,
    feh_offset,):
    """Calculates the uncertainty weighted residuals between a science spectrum
    and a generated template spectrum with parameters per 'params'. If blue
    band spectrum related arrays are None, will just fit for red spectra. This
    is to account for potentially poor SNR in the blue.

    Parameters
    ----------
    params: float array
        Initial parameter guess of form (teff, logg, [Fe/H])

    wave_r: float array
        Wavelength scale for the red spectrum.

    spec_r: float array
        The red spectrum corresponding to wave_r.

    e_spec_r: float array
        Uncertainties on red spectra.

    bad_px_mask_r: boolean array
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    rv: float
        Radial velocity in km/s

    bcor: float:
        Barycentric velocity in km/s

    idl: pidly.IDL
        pidly.IDL object to interface with Thomas Nordlander's synthetic 
        MARCS spectra through IDL.

    band_settings_r: dict
        Dictionary with settings for WiFeS red band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    logg: float
        logg of the star, if fixing this dimension during fitting. If not None,
        will only use least squares to optimise Teff and [Fe/H]

    band_settings_r: dict
        Dictionary with settings for WiFeS blue band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    wave_b: float array, optional
        Wavelength scale for the blue spectrum.

    spec_b: float array, optional
        The red spectrum corresponding to wave_r.

    e_spec_b: float array, optional
        Uncertainties on red spectra.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    stellar_colours: float array, default: None
        Array of observed stellar colour corresponding to colour_bands. If None
        photometry is not used in the fit.

    e_stellar_colours: float array, default: None
        Array of observed stellar colour uncertainties. If None photometry is 
        not used in the fit.

    colour_bands: string array, default: ['Rp-J', 'J-H', 'H-K']
        Colour bands to use in the fit.

    sc_interp: SyntheticColourInterpolator
        SyntheticColourInterpolator object able to interpolate synth colours.

    feh_offset: float, default: 10
        Arbitrary offset to add to [Fe/H] so that it never goes below zero to
        improve compatability with diff_step.

    Returns
    -------
    resid_vect: float array
        Uncertainty weighted residual vector between science and synthetic
        spectra.
    """
    # Unpack params and unscale [Fe/H]
    if logg is None:
        teff, logg, feh = params - np.array([0,0,feh_offset])
    else:
        teff, feh = params - np.array([0,feh_offset])

    # Initialise boolean flags to indicate which residuals are included in the
    # fit - blue spectra, red spectra, and photometry
    brc = [False, True, False]

    # Fit red
    resid_vect_r, spec_synth_r = calc_synth_fit_resid_one_arm(
        teff,
        logg,
        feh, 
        wave_r, 
        spec_r, 
        e_spec_r, 
        bad_px_mask_r,
        rv, 
        bcor, 
        idl,
        band_settings_r,)
    
    # Do blue if we've been given it
    if ((wave_b is not None) or (spec_b is not None) or (e_spec_b is not None) 
        or (bad_px_mask_b is not None)):
        # Determine blue residuals
        resid_vect_b, spec_synth_b = calc_synth_fit_resid_one_arm(
            teff,
            logg,
            feh, 
            wave_b, 
            spec_b, 
            e_spec_b, 
            bad_px_mask_b,
            rv, 
            bcor, 
            idl,
            band_settings_b,)
        
        # Combine residual vectors
        resid_vect = np.concatenate((resid_vect_b, resid_vect_r))

        brc[0] = True
    
    else:
        # Default blue synth spec
        spec_synth_b = None

        # Just use red residual vector
        resid_vect = resid_vect_r
    
    # Do photometric fit if we've been given data
    if sc_interp is not None:
        resid_vect_colour = calc_colour_resid(
            teff,
            logg,
            feh,
            stellar_colours,
            e_stellar_colours,
            colour_bands,
            sc_interp)

        if not np.sum(np.isfinite(resid_vect_colour)):
            resid_vect_colour = np.ones_like(resid_vect_colour) * 1E30
        else:
            brc[2] = True

        # Append this residual vector to the one from our spectra
        resid_vect = np.concatenate((resid_vect, resid_vect_colour))

    # Print updates
    print("Teff = {:0.5f} K, logg = {:0.05f}, [Fe/H] = {:+0.05f}".format(
        teff, logg, feh), "\t{}".format(brc), end="")
    
    rchi2 = np.sum(resid_vect**2) / (len(resid_vect)-len(params))
    print("\t--> rchi^2 = {:0.1f}".format(rchi2))

    return resid_vect


def do_synthetic_fit(
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r, 
    params, 
    rv, 
    bcor,
    idl,
    band_settings_r,
    logg=None,
    band_settings_b=None,
    wave_b=None, 
    spec_b=None, 
    e_spec_b=None, 
    bad_px_mask_b=None,
    stellar_colours=None,
    e_stellar_colours=None,
    colour_bands=['Rp-J', 'J-H', 'H-K'],
    feh_offset=10,):
    """Performs least squares fitting (using scipy.optimize.least_squares) on
    science spectra given an initial stellar parameter guess. By default only
    fits red arm spectra (to account for poor SNR blue spectra), but will fit 
    blue spectra if provided.

    Parameters
    ----------
    wave_r: float array
        Wavelength scale for the red spectrum.

    spec_r: float array
        The red spectrum corresponding to wave_r.

    e_spec_r: float array
        Uncertainties on red spectra.

    bad_px_mask_r: boolean array
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    params: float array
        Initial parameter guess, either of form:
         1) (teff, logg, [Fe/H])
         2) (teff, [Fe/H]) if logg is provided

    rv: float
        Radial velocity in km/s

    bcor: float:
        Barycentric velocity in km/s
    
    idl: pidly.IDL
        pidly.IDL object to interface with Thomas Nordlander's synthetic 
        MARCS spectra through IDL.

    band_settings_r: dict
        Dictionary with settings for WiFeS red band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    logg: float, default: None
        logg of the star, if fixing this dimension during fitting. If provided,
        will only use least squares to optimise Teff and [Fe/H]

    band_settings_r: dict, default: None
        Dictionary with settings for WiFeS blue band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    wave_b: float array, optional
        Wavelength scale for the blue spectrum.

    spec_b: float array, optional
        The red spectrum corresponding to wave_r.

    e_spec_b: float array, optional
        Uncertainties on red spectra.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    stellar_colours: float array, default: None
        Array of observed stellar colour corresponding to colour_bands. If None
        photometry is not used in the fit.

    e_stellar_colours: float array, default: None
        Array of observed stellar colour uncertainties. If None photometry is 
        not used in the fit.

    colour_bands: string array, default: ['Rp-J', 'J-H', 'H-K']
        Colour bands to use in the fit.

    feh_offset: float, default: 10
        Arbitrary offset to add to [Fe/H] so that it never goes below zero to
        improve compatability with diff_step.

    Returns
    -------
    optimize_result: dict
        Dictionary of best fit results returned from 
        scipy.optimize.least_squares.
    """
    # If no photometry, no need to initialise synthetic colour interpolator
    if stellar_colours is None or e_stellar_colours is None:
        sc_interp = None

    # If we've been given photometry, initialise synthetic colour interpolator
    else:
        sc_interp = SyntheticColourInterpolator()

    # Setup fit settings
    args = (wave_r, spec_r, e_spec_r, bad_px_mask_r, rv, bcor , idl, 
            band_settings_r, logg, band_settings_b, wave_b, spec_b, e_spec_b, 
            bad_px_mask_b, stellar_colours, e_stellar_colours, colour_bands, 
            sc_interp, feh_offset)
    
    if logg is None:
        bounds = ((2800, 4, -2+feh_offset), (6000, 5.5, 0.5+feh_offset))
        scale = (1, 1, 1)
        step = (0.1, 0.1, 0.1)

        params = np.array(params) + np.array([0,0,feh_offset])
    else:
        bounds = ((2800, -2+feh_offset), (6000, 0.5+feh_offset))
        scale = (1, 1)
        step = (0.1, 0.1)

        params = np.array(params) + np.array([0,feh_offset])
    
    # Make sure initial teff guess isn't out of bounds, assign default if so
    if params[0] < bounds[0][0] or params[0] > bounds[1][0]:
        params[0] = 5250

    # Do fit
    opt_res = least_squares(
        calc_synth_fit_resid_both_arms, 
        params, 
        jac="3-point",
        bounds=bounds,
        x_scale=scale,
        diff_step=step,
        args=args, 
    )

    # Unscale [Fe/H]
    opt_res["x"][-1] -= feh_offset

    if logg is None:
        teff, logg, feh = opt_res["x"]
    else:
        teff, feh = opt_res["x"]

    # Generate and save synthetic spectra at optimal params
    if band_settings_b is not None:
        _, spec_synth_b = get_idl_spectrum(
            idl, 
            teff, 
            logg, 
            feh, 
            wl_min=band_settings_b["wl_min"], 
            wl_max=band_settings_b["wl_max"], 
            ipres=band_settings_b["inst_res_pow"],
            resolution=band_settings_b["wl_broadening"],
            norm="abs",
            do_resample=True, 
            wl_per_px=band_settings_b["wl_per_px"],
            rv_bcor=(rv-bcor),
            )
    else:
        spec_synth_b = None

    _, spec_synth_r = get_idl_spectrum(
        idl, 
        teff, 
        logg, 
        feh, 
        wl_min=band_settings_r["wl_min"], 
        wl_max=band_settings_r["wl_max"], 
        ipres=band_settings_r["inst_res_pow"],
        resolution=band_settings_r["wl_broadening"],
        norm="abs",
        do_resample=True, 
        wl_per_px=band_settings_r["wl_per_px"],
        rv_bcor=(rv-bcor),
        )

    # Calculate uncertainties
    jac = opt_res["jac"]
    res = opt_res["fun"]
    cov = np.linalg.inv(jac.T.dot(jac))
    std = np.sqrt(np.diagonal(cov)) * np.nanvar(res)

    # Add uncertainties and synthtic spectra to return dict
    opt_res["std"] = std
    opt_res["spec_synth_b"] = spec_synth_b
    opt_res["spec_synth_r"] = spec_synth_r

    return opt_res

# -----------------------------------------------------------------------------
# Synthetic Photometry
# -----------------------------------------------------------------------------
def load_filter_profile(
    filt, 
    min_wl=1200, 
    max_wl=30000, 
    gaia_filt_path="data/GaiaDR2_Passbands.dat",
    tmass_filt_path="data/2mass_{}_profile.txt"):
    """Load in the specified filter profile and zero pad both ends in 
    preparation for feeding into an interpolation function.
    """
    filt = filt.upper()

    filters_gaia = np.array(["G", "BP", "RP"])
    filters_2mass = np.array(["J", "H", "K"])

    if filt not in filters_gaia and filt not in filters_2mass:
        raise ValueError("Filter must be either {} or {}".format(
            filters_gaia, filters_2mass))

    # 2MASS filter profiles
    if filt in filters_2mass:
        # Load the filter profile, and convert to Angstroms
        tmpb = pd.read_csv(tmass_filt_path.format(filt), delim_whitespace=True)

        wl = tmpb["wl"].values * 10**4
        filt_profile = tmpb["pb"].values
    
    # Gaia filter profiles
    elif filt in filters_gaia:
        # Import Gaia passbands
        gpb = pd.read_csv(gaia_filt_path, header=0, delim_whitespace=True)

        # Set the undefined passband values to 0
        gpb[gpb[["G_pb", "BP_pb", "RP_pb"]]==99.99] = 0   
        
        wl = gpb["wl"].values * 10
        filt_profile = gpb["{}_pb".format(filt)].values
    
    # Pre- and post- append zeros from min_wl to max_wl
    prepad_wl = np.arange(min_wl, wl[0], 1000)
    prepad = np.zeros_like(prepad_wl)
    
    postpad_wl = np.arange(wl[-1], max_wl, 1000)[1:]
    postpad = np.zeros_like(postpad_wl)
    
    wl_filt = np.concatenate((prepad_wl, wl, postpad_wl))
    filt_profile = np.concatenate((prepad, filt_profile, postpad))
    
    return wl_filt, filt_profile


def calc_synth_mag(wave, fluxes, filt):
    """Function to put a given filter profile on a different wavelength scale, 
    and compute the fluxes given a spectrum. 

    NOTE: Wrong formalism, colour are wrong!!!!

    Parameters
    ----------
    wave: float array
        Wavelengths in A.
    fluxes: float array
        Spectral fluxes in cgs wavelength-units (erg/s/A)
    """
    # Input checking
    filt = filt.upper()

    filters_gaia = np.array(["G", "BP", "RP"])
    filters_2mass = np.array(["J", "H", "K"])

    if filt not in filters_gaia and filt not in filters_2mass:
        raise ValueError("Filter must be either {} or {}".format(
            filters_gaia, filters_2mass))

    # Load in filter profile
    wl_filt, filt_profile = load_filter_profile(filt)

    # Set up the interpolator for the filter profile
    calc_filt_profile = ius(wl_filt, filt_profile)#, kind="linear")
                                 
    # Interpolate the filter profile onto the science wavelength scale
    filt_profile_sci = calc_filt_profile(wave)
    
    # Assume solar radii for synthetic fluxes/star, and scale to 10 pc distance
    r_sun = 6.95700*10**8        # metres
    #d_10pc = 10 * 3.0857*10**16  # metres
    
    #flux = (r_sun / d_10pc)**2 * fluxes

    # Compute the flux density of the star
    #ean_flux_density = (simps(fluxes*filt_profile_sci*wl_sci, wl_sci)
    #                     / simps(filt_profile_sci*wl_sci, wl_sci))

    #mean_flux_density = simps(fluxes*filt_profile_sci*wl_sci, wl_sci)
    
    if filt in filters_gaia:
        # Gaia filter zeropoints from: 
        # https://www.cosmos.esa.int/web/gaia/iow_20180316
        gaia_pa = 0.7278 # m^2
        zps_gaia = {"G":(25.6883657251, 0.0017850023),
                    "BP":(25.3513881707, 0.0013918258),
                    "RP":(24.7619199882, 0.0019145719)}

        # mag_star = -2.5 log10(I) + ZP
        flux_int = simps(wave*fluxes*filt_profile_sci)

        i_zeta = gaia_pa/(const.c.value*const.h.value*10**9) * flux_int
        mag = -2.5 * np.log10(i_zeta) + zps_gaia[filt][0]

    else:
        # Define Vega fluxes (erg s^-1 cm^-2 A^-1) & zeropoints for each filter
        # Vega fluxes from Casagrande & VendenBerg 2014, Table 1
        # Band zeropoints from Bessell, Castelli, & Plez 1998, Table A2
        zps_2mass = {"J":[3.129* 10**-10, 0.899], 
                    "H":[1.113* 10**-10, 1.379], 
                    "K":[4.283* 10**-11, 1.886]}  # erg cm-2 s-1 A-1

        #fluxes /= 4*np.pi#*(r_sun*10)**2

        # 2MASS filter profiles (T_zeta) are from Cohen et al. 2003 
        # (2003AJ....126.1090C), and have already been normalised by lambda
        wmf = simps(fluxes*filt_profile_sci, wave)
              # / simps(filt_profile_sci, wave))

        # Calculate the magnitude of each star w.r.t. Vega
        # mag_star = -2.5 log10(F_star / F_vega) + zero-point
        mag = -2.5 * np.log10(wmf/zps_2mass[filt][0]) + zps_2mass[filt][1]
    
    return mag


def calc_synth_colour(wave, fluxes, filt_1, filt_2):
    """Calculates a synthetic stellar colour of form (mag_1 - mag_2) where 
    mag_1 and mag_2 are from filt_1 and filt_2 respectively.
    """
    # Input checking
    filt_1 = filt_1.upper()
    filt_2 = filt_2.upper()

    filters = np.array(["G", "BP", "RP", "J", "H", "K"])

    if filt_1 not in filters or filt_2 not in filters:
        raise ValueError("Filters must both be in {}".format(filters))
    
    mag_1 = calc_synth_mag(wave, fluxes, filt_1)
    mag_2 = calc_synth_mag(wave, fluxes, filt_2)

    colour = mag_1 - mag_2

    return colour


def calc_synth_colour_array(wave, fluxes, colour_bands):
    """
    """
    colours = []

    # For every colour band specified, calculate the colour
    for colour_band in colour_bands:
        filt_1 = colour_band.split("-")[0]
        filt_2 = colour_band.split("-")[1]

        colours.append(calc_synth_colour(wave, fluxes, filt_1, filt_2))

    return np.array(colours)


# -----------------------------------------------------------------------------
# Working with Casagrande BC code
# ----------------------------------------------------------------------------- 
def generate_casagrande_bc_grid(
    teff_lims=(2800, 6000),
    logg_lims=(4.0, 5.5),
    feh_lims=(-2.0, 0.5),
    ebv_lims=None,
    delta_teff=100,
    delta_logg=0.1,
    delta_feh=0.1,
    delta_ebv=None,
    filters=["Bp", "Rp", "J", "H", "K"],
    bc_path="/Users/adamrains/code/bolometric-corrections/"):
    """Setup grid of stellar parameters for use with the bolometric correction
    code from Casagrande & VandenBerg (2014, 2018a, 2018b):
    
    https://github.com/casaluca/bolometric-corrections
    
    Check that selectbc.data looks like this to compute Bp, Rp, J, H, K:
        1  = ialf (= [alpha/Fe] variation: select from choices listed below)
        5  = nfil (= number of filter bandpasses to be considered; maximum = 5)
        27 86  =  photometric system and filter (select from menu below)
        27 88  =  photometric system and filter (select from menu below)
        1 1  =  photometric system and filter (select from menu below)
        1 2  =  photometric system and filter (select from menu below)
        1 3  =  photometric system and filter (select from menu below)

    For simplicity (but mostly to avoid errors), this function does not run 
    bcgo, but the file is in an acceptable format to do so from a terminal.

    Parameters
    ----------
    teff_lims: float array, default: (2800, 6000)
        Limits on the temperature grid of form (teff_min, teff_max) in K.

    logg_lims: float array, default: (4.0, 5.5)
        Limits on the surface gravity grid of form (logg_min, logg_max).

    feh_lims: float array, default: (-2.0, 0.5)
        Limits on the metallicity grid of form (feh_min, feh_max).

    ebv_lims: float array, default: None
        Limits on the E(B-V) grid of form (ebv_min, ebv_max). If None, no
        reddening is assumed.

    delta_teff: float, default: 100
        Grid step in temperature dimension.

    delta_logg: float, default: 0.1
        Grid step in gravity dimension.

    delta_feh: float, default: 0.1
        Grid step in metallicity dimension.

    delta_ebv: float, default: None
        Grid step in E(B-V) dimension.

    filters: string array, default: ["Bp", "Rp", "J", "H", "K"]
        Filters to compute colours for.

    bc_path: string, default: "/Users/adamrains/code/bolometric-corrections/"
        Path to save grid value file to.
    """
    # Initialise arrays
    teffs = np.arange(teff_lims[0], teff_lims[1]+delta_teff, delta_teff)
    loggs = np.arange(logg_lims[0], logg_lims[1]+delta_logg, delta_logg)
    fehs = np.arange(feh_lims[0], feh_lims[1]+delta_feh, delta_feh)

    if ebv_lims is None or delta_ebv is None:
        ebvs = [0]
    else:
        ebvs = np.arange(ebv_lims[0], ebv_lims[1], delta_ebv)

    # Make grid
    grid = []
    point_i = 0

    for logg in loggs:
        for feh in fehs:
            for teff in teffs:
                for ebv in ebvs:
                    if not (teff > 4000 and logg > 5):
                        grid.append(
                            ["i{:05d}".format(point_i), logg, feh, teff, ebv])
                        #grid.append([logg, feh, teff])

                    point_i += 1

    grid = np.stack(grid)

    # Save
    in_bc_path = os.path.join(bc_path, "input.sample.all")
    np.savetxt(in_bc_path, grid, delimiter=" ", fmt="%s")

    # Run
    #os.system("cd {}; ./bcall".format(bc_path))
    #os.system("cd {}; ./bcgo".format(bc_path))


def initialise_casagrande_bc_grid(
    filters=["Bp", "Rp", "J", "H", "K"],
    bc_path="/Users/adamrains/code/bolometric-corrections/",
    save_path="data/synth_colour_grid.csv"):
    """Initialise a grid of stellar parameters, bolometric corrections, and 
    synthetic colours based on the output of the bolometric-corrections code.

    Parameters
    ----------
    filters: string array, default: ["Bp", "Rp", "J", "H", "K"]
        Filters to compute colours for.

    bc_path: string, default: "/Users/adamrains/code/bolometric-corrections/"
        Path to load bolometric correction results from
    """
    # Read in
    out_bc_path = os.path.join(bc_path, "output.file.all")
    bc_grid = pd.read_csv(out_bc_path, delim_whitespace=True)

    bc_cols = ["BC_{}".format(filt) for filt in filters]
    colour_cols = ["{}-{}".format(f1, f2) 
                   for f1, f2 in zip(filters[:-1], filters[1:])]

    bc_grid.columns = ["id", "logg", "feh", "teff", "ebv"] + bc_cols

    bc_values = bc_grid[bc_cols].values

    # Calculate the colours, noting that we have to do the subtraction 
    # 'backwards' to calculate a colour from a BC
    colours = pd.DataFrame(
        data=np.stack([bc_values[:,i+1]-bc_values[:,i] 
                       for i in range(bc_values.shape[1]-1)]).T,
        columns=colour_cols)

    bc_grid = pd.concat((bc_grid,colours), axis=1)

    bc_grid.to_csv(save_path)

    return bc_grid


def load_casagrade_colour_grid(load_path="data/synth_colour_grid.csv"):
    """Import the constructed grid, with columns [id, logg, feh, teff, ebv,
     BC_Bp, BC_Rp, BC_J, BC_H, BC_K, Bp-Rp, Rp-J, J-H, H-K].

    Parameters
    ----------
    load_path: string, default: 'data/synth_colour_grid.csv'
        Where to load the grid from.
    """
    bc_grid = pd.read_csv(load_path, index_col=0)

    return bc_grid


class SyntheticColourInterpolator():
    """Class to interpolate synthetic colours, meaning that you don't need to
    reload and set up the interpolator every time as you would with a function.
    """
    def __init__(self):
        """Constructor
        """
        param_cols = ["teff", "logg", "feh"]

        self.bc_grid = load_casagrade_colour_grid()
        self.VALID_COLOURS = ["Bp", "Rp", "J", "H", "K"]

        # Setup interpolators for each possible colour
        self.calc_bprp =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["Bp-Rp"].values)

        self.calc_rpj =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["Rp-J"].values)

        self.calc_jh =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["J-H"].values)

        self.calc_hk =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["H-K"].values)


    def compute_colour(self, params, colour_bands):
        """Computes a synthetic colour based on params and provided colour band

        Parameters
        ------
        params: float array
            Stellar parameters of form (teff, logg, [Fe/H]).

        colour_bands: string
            The stellar colour, either 'Bp-Rp', 'Rp-J', 'J-H', or 'H-K'.

        Returns
        -------
        colour: float array
            Resulting synthetic stellar colour.
        """
        if colour_bands == "Bp-Rp":
            colour = self.calc_bprp(params[0], params[1], params[2])

        elif colour_bands == "Rp-J":
            colour = self.calc_rpj(params[0], params[1], params[2])

        elif colour_bands == "J-H":
            colour = self.calc_jh(params[0], params[1], params[2])

        elif colour_bands == "H-K":
            colour = self.calc_hk(params[0], params[1], params[2])

        else:
            raise ValueError("Invalid colour, must be in {}".format(
                self.VALID_COLOURS))

        return colour