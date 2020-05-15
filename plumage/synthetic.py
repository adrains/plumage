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

from scipy.interpolate import InterpolatedUnivariateSpline as ius

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
                     resolution=None, norm="abs", do_resample=False, 
                     wl_per_pixel=None, rv_bcor=0):
    """
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
    
    do_resample: bool
        Whether to resample onto custom wavelength scale.

    wl_per_pixel: bool
        Wavelength per pixel of new wavelength grid.

    rv_bcor: float, default: 0
        The velocity offset (km/s) for the synthetic spectrum, 0 by default. If 
        matching observations, this should be the total velocity offset, that
        is both the RV and barcentric offset.

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
        raise ValueError("Temperature must be 2500 <="
                                            " Teff (K) <= 8000")
    if logg > 5.5 or logg < -1:
        raise ValueError("Surface gravity must be -1 <="
                                            " logg (cgs) <= 5.5")
    if feh > 1.0 or feh < -5:
        raise ValueError("Metallicity must be -5 <="
                                            " [Fe/H] (dex) <= 1")
    if wl_min > wl_max or wl_max > 200000 or wl_min < 2000:
        raise ValueError("Wavelengths must be 2,000 <="
                                            " lambda (A) <= 60,000")                
    if norm not in NORM_VALS:
        raise ValueError("Invalid normalisation value, see NORM_VALS.")
    
    norm_val = NORM_VALS[norm]

    idl("CFe = 0. ;")

    # Default behaviour is constant-velocity broadening, unless a value has 
    # been provided for resolution
    if resolution is None:
        cmd = ("spectrum = get_spec(%d, %f, %f, !null, CFe, %f, %f, ipres=%f, "
               "norm=%i, grid=grid, wave=wave, vrad=%f)" 
               % (teff, logg, feh, wl_min, wl_max, ipres, norm_val, rv_bcor))
    # Otherwise do constant-wavelength broadening
    else:
        cmd = ("spectrum = get_spec(%d, %f, %f, !null, CFe, %f, %f, norm=%i, "
               "resolution=%f, grid=grid, wave=wave, vrad=%f)" 
               % (teff, logg, feh, wl_min, wl_max, norm_val, resolution, 
                  rv_bcor))
    
    idl(cmd)
    
    if do_resample:
        idl("waveout = [%i+%f:%i-2*%f:%f]" 
            % (wl_min, wl_per_pixel, wl_max, wl_per_pixel, wl_per_pixel))
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

    elif setting == "B3000":
        wl_min = 3500
        wl_max = 5700
        resolution = 3000
        n_px = 2858

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

    # Calculate the wavelength spacing
    wl_per_pixel = (wl_max - wl_min) / n_px 

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
                        wl_per_pixel=wl_per_pixel)
                
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
    params, 
    wave_sci, 
    spec_sci, 
    e_spec_sci, 
    bad_px_mask,
    rv, 
    bcor, 
    idl,
    arm,
    ):
    """
    """
    # Initialise details of synthetic wavelength scale
    # TODO - Generalise this
    if arm == "r":
        res = 7000
        wl_min = 5400
        wl_max = 7000
        n_px = 3637
    elif arm == "b":
        res = 3000
        wl_min = 3500
        wl_max = 5700
        n_px = 2858
    else:
        raise ValueError("Must be either r or b")

    wl_per_px = (wl_max - wl_min) / n_px

    # Get the template spectrum
    wave_synth, spec_synth = get_idl_spectrum(
        idl, 
        params[0], 
        params[1], 
        params[2], 
        wl_min, 
        wl_max, 
        res, 
        norm="abs",
        do_resample=True, 
        wl_per_pixel=wl_per_px,
        )

    # The grid we put our new synthetic spectrum on should be put in the same
    # RV frame as the science spectrum
    wave_rv_scale = 1 - (rv - bcor)/(const.c.si.value/1000)
    ref_spec_interp = ius(wave_synth, spec_synth)

    wave_synth = wave_sci * wave_rv_scale
    spec_synth = ref_spec_interp(wave_synth)

    # Normalise spectra by wavelength region before fit
    spec_sci, e_spec_sci = spec.norm_spec_by_wl_region(wave_sci, spec_sci, arm, e_spec_sci)
    spec_synth = spec.norm_spec_by_wl_region(wave_sci, spec_synth, arm)

    # Calculate the residual
    resid_vect = (spec_sci[~bad_px_mask] 
                  - spec_synth[~bad_px_mask]) / e_spec_sci[~bad_px_mask]

    if not np.isfinite(np.sum(resid_vect)):
        resid_vect = np.ones_like(resid_vect) * 1E30

    return resid_vect, spec_synth


def calc_synth_fit_resid_both_arms(
    params, 
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r,
    rv, 
    bcor, 
    idl,
    wave_b, 
    spec_b, 
    e_spec_b, 
    bad_px_mask_b,
    best_fit_spec_dict,
    #norm_range
    ):
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

    wave_b: float array, optional
        Wavelength scale for the blue spectrum.

    spec_b: float array, optional
        The red spectrum corresponding to wave_r.

    e_spec_b: float array, optional
        Uncertainties on red spectra.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    best_fit_spec_dict: dict
        Initially empty dictionary used to return normalised science spectrum 
        and best fit synthetic spectrum.

    Returns
    -------
    resid_vect: float array
        Uncertainty weighted residual vector between science and synthetic
        spectra.
    """
    print(params)

    # Fit red
    resid_vect_r, spec_synth_r = calc_synth_fit_resid_one_arm(
        params, 
        wave_r, 
        spec_r, 
        e_spec_r, 
        bad_px_mask_r,
        rv, 
        bcor, 
        idl,
        "r")
    
    # Do blue if we've been given it
    if ((not wave_b is None) or (not spec_b is None) or (not e_spec_b is None) 
        or (not bad_px_mask_b is None)):
        # Determine blue residuals
        resid_vect_b, spec_synth_b = calc_synth_fit_resid_one_arm(
            params, 
            wave_b, 
            spec_b, 
            e_spec_b, 
            bad_px_mask_b,
            rv, 
            bcor, 
            idl,
            "b")
        
        # Combine residual vectors
        resid_vect = np.concatenate((resid_vect_b, resid_vect_r))
    
    else:
        # Default blue synth spec
        spec_synth_b = None

        # Just use red residual vector
        resid_vect = resid_vect_r
    
    # Save the best fit normalised spectra
    #best_fit_spec_dict["wave"] = wave_r
    #best_fit_spec_dict["wl_mask"] = ~bad_px_mask_r
    #best_fit_spec_dict["spec_sci"] = spec_r
    #best_fit_spec_dict["e_spec_sci"] = e_spec_r
    best_fit_spec_dict["spec_synth_b"] = spec_synth_b
    best_fit_spec_dict["spec_synth_r"] = spec_synth_r

    return resid_vect


def do_synthetic_fit(
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r, 
    params, 
    rv, 
    bcor,
    wave_b=None, 
    spec_b=None, 
    e_spec_b=None, 
    bad_px_mask_b=None,):
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
        Initial parameter guess of form (teff, logg, [Fe/H])

    rv: float
        Radial velocity in km/s

    bcor: float:
        Barycentric velocity in km/s

    wave_b: float array, optional
        Wavelength scale for the blue spectrum.

    spec_b: float array, optional
        The red spectrum corresponding to wave_r.

    e_spec_b: float array, optional
        Uncertainties on red spectra.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    Returns
    -------
    optimize_result: dict
        Dictionary of best fit results returned from 
        scipy.optimize.least_squares.

    best_fit_spec_dict: dict
        Dictionary containing normalised science spectrum and best fit 
        synthetic spectrum
    """
    # initialise IDL
    idl = idl_init()

    # Parameter to store best fit synthetic spectra
    best_fit_spec_dict = {}

    # Setup fit settings
    args = (wave_r, spec_r, e_spec_r, bad_px_mask_r, rv, bcor , idl, 
            wave_b, spec_b, e_spec_b, bad_px_mask_b, best_fit_spec_dict)
    bounds = ((2500, -1, -5), (7900, 5.5, 1))
    scale = (1, 1, 1)
    step = (10, 0.1, 0.1)
    
    # Make sure initial teff guess isn't out of bounds, assign default if so
    params = np.array(params)

    if params[0] < bounds[0][0] or params[0] > bounds[1][0]:
        params[0] = 5250

    # Do fit
    optimize_result = least_squares(
        calc_synth_fit_resid_both_arms, 
        params, 
        jac="3-point",
        bounds=bounds,
        x_scale=scale,
        diff_step=step,
        args=args, 
    )

    return optimize_result, best_fit_spec_dict