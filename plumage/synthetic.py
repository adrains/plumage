"""Code for working with synthetic spectra.
"""
from __future__ import division, print_function
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
import spectra as spec
import plotting as pplt
from scipy.interpolate import InterpolatedUnivariateSpline as ius

#------------------------------------------------------------------------------    
# Setup IDL
#------------------------------------------------------------------------------
def idl_init():
    """Initialise IDL by setting paths and compiling relevant files.
    """
    idl = pidly.IDL()
    idl("!path = '/home/thomasn/idl_libraries/coyote:' + !path")
    idl(".compile /home/thomasn/grids/gaussbroad.pro")
    idl(".compile /home/thomasn/grids/get_spec.pro")
    idl("grid='/home/thomasn/grids/grid_synthspec.sav'")
    
    return idl
    
def get_idl_spectrum(idl, teff, logg, feh, wl_min, wl_max, resolution, 
                     norm="abs", do_resample=False, wl_per_pixel=None):
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
    cmd = ("spectrum = get_spec(%d, %f, %f, !null, CFe, %i, %i, ipres=%i, "
           "norm=%i, grid=grid, wave=wave)" % (teff, logg, feh, wl_min, wl_max, 
                                               resolution, norm_val))
    
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
# rv, wave, spec, e_spec, ref_spec_interp, bcor
def calc_synth_fit_resid(
    params, 
    wave, 
    spec_sci, 
    e_spec, 
    rv, 
    bcor, 
    idl,
    wl_min,
    wl_max,
    res,
    wl_per_px,
    band,
    #norm_range
    ):
    """
    """
    print(params)

    # Get the template spectrum
    try:
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
    except:
        import pdb
        pdb.set_trace()

    # The grid we put our new synthetic spectrum on should be put in the same
    # RV frame as the science spectrum
    try:
        wave_rv_scale = 1 - (rv - bcor)/(const.c.si.value/1000)
        ref_spec_interp = ius(wave_synth, spec_synth)

        wave_synth = wave * wave_rv_scale
        spec_synth = ref_spec_interp(wave_synth)

        # Normalise by the region just to the blue of H alpha
        if band == "red":
            norm_mask = np.logical_and(wave > 6535, wave < 6545)
            xx = 6000
            mask_blue_edges = False

            #emission_mask = np.logical_and(wave > 6562.3*wave_rv_scale, 
            #                            wave < 6563.3*wave_rv_scale)
            #emission_mean = np.nanmean(spec_sci[emission_mask])

        # Normalise by bandhead near 5000 A
        elif band == "blue":
            norm_mask = np.logical_and(wave > 4925, wave < 4950)
            xx = 4500
            mask_blue_edges = True

            #emission_mean = 0

        else:
            raise ValueError("Must be either red or blue")
    except:
        import pdb
        pdb.set_trace()

    try:
        spec_synth /= np.nanmean(spec_synth[norm_mask])
        spec_sci /= np.nanmean(spec_sci[norm_mask])
        e_spec /= np.nanmean(spec_sci[norm_mask])

        # Check to see whether to mask emission
        #if emission_mean > 1:
        #    mask_emission = True
        #else:
        #    mask_emission = False
        mask_emission = True
        
    except:
        import pdb
        pdb.set_trace()

    # Make a wavelength mask
    try:
        wl_mask = spec.make_wavelength_mask(
            wave, 
            mask_emission=mask_emission, 
            mask_edges=True,
            mask_sky_emission=True,
            mask_blue_edges=mask_blue_edges)
        
        # Calculate the residual
        resid_vect = (spec_sci[wl_mask] - spec_synth[wl_mask]) / e_spec[wl_mask]

        pplt.plot_synthetic_fit(wave[wl_mask], spec_sci[wl_mask], spec_synth[wl_mask], params)

        if not np.isfinite(np.sum(resid_vect)):
            resid_vect = np.ones_like(resid_vect) * 1E30
    except:
        import pdb
        pdb.set_trace()
    return resid_vect


def do_synthetic_fit(wave, spec, e_spec, params, rv, bcor, band="red"):
    """
    """
    # initialise IDL
    idl = idl_init()

    # TODO - Generalise this
    if band == "red":
        res = 7000
        wl_min = 5400
        wl_max = 7000
        n_px = 3637
    elif band == "blue":
        res = 3000
        wl_min = 3500
        wl_max = 5700
        n_px = 2858
    else:
        raise ValueError("Must be either red or blue")

    wl_per_px = (wl_max - wl_min) / n_px  

    # Do fit, have initial RV guess be 0 km/s
    args = (wave, spec, e_spec, rv, bcor, idl, wl_min, wl_max, res, wl_per_px, band)
    bounds = ((2500, -1, -5), (7900, 5.5, 1))
    scale = (1, 1, 1)
    step = (10, 0.1, 0.1)
    # param_fit, cov, infodict, mesg, ier
    try:
        optimize_result = least_squares(
            calc_synth_fit_resid, 
            params, 
            jac="3-point",
            #loss="arctan",
            bounds=bounds,
            x_scale=scale,
            diff_step=step,
            args=args, 
        )
    except:
        import pdb
        pdb.set_trace()
        #full_output=True)
    return optimize_result
    return param_fit, cov, infodict, mesg, ier

    # Scale the covariance matrix, calculate RV uncertainty
    # docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
    if cov is None:
        e_rv = np.nan
    else:
        cov *= np.nanvar(infodict["fvec"])
        e_rv = cov**0.5

    # Add extra parameters to infodict
    infodict["cov"] = cov
    infodict["mesg"] = mesg
    infodict["ier"] = ier

    # Calculate reduced chi^2
    rchi2 = np.nansum(infodict["fvec"]**2) / (len(spec_sci)-len(param_fit))
    
    return float(param_fit), float(e_rv), rchi2, infodict