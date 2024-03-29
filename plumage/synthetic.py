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
from collections import OrderedDict
import astropy.constants as const
from astropy import units as u
from scipy.optimize import leastsq, least_squares
import plumage.spectra as spec
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import LinearNDInterpolator
import astropy.convolution as conv

#------------------------------------------------------------------------------    
# Setup IDL
#------------------------------------------------------------------------------
def idl_init(drive="home"):
    """Initialise IDL by setting paths and compiling relevant files. Also 
    initialises paths to various IDL grids.

    Parameters
    ----------
    drive: string, default: 'home'
        Keyword for how to reference grids, either 'home' for alias, or 'priv'
        for longer path on avatar.

    Returns
    -------
    idl: pidly.IDL()
        pidly IDL object to pass IDL commands to.
    """
    # Do initial compilation
    idl = pidly.IDL()
    idl("!path = '/home/thomasn/idl_libraries/coyote:' + !path")
    idl(".compile /home/thomasn/grids/gaussbroad.pro")
    idl(".compile /home/thomasn/grids/get_spec.pro")

    # Intialise full, B3000, and R7000 grids
    if drive == "home":
        root = "/home/thomasn/grids/"
    
    elif drive == "priv":
        root = "/priv/avatar/thomasn/Turbospectrum-15.1/COM-v15.1/grids/"

    else:
        raise ValueError(
            "Unknown value for drive, must be either 'home' or 'priv'")

    idl("grid='{}'".format(os.path.join(root, "grid_synthspec_main.sav")))
    idl("grid_b='{}'".format(os.path.join(root, "grid_3000-10000_res1.0.sav")))
    idl("grid_r='{}'".format(os.path.join(root, "grid_3000-10000_res0.45.sav")))
    idl("grid_mdwarf='{}'".format(os.path.join(root, "grid_synthY_Mdwarftest.sav")))

    return idl
    
def get_idl_spectrum(idl, teff, logg, feh, wl_min, wl_max, ipres, grid="full",
                     resolution=None, norm="abs", do_resample=True, 
                     wl_per_px=None, rv_bcor=0, wave_pad=50, abund_alpha=None,
                     abund=None, ebv=0, do_log_resample=False, n_px=None):
    """Calls Thomas Nordlander's IDL routines (specifically get_spec) to 
    generate a synthetic MARCS model spectrum at the requested parameters. 

    Parameters
    ----------
    idl: pidly.IDL
        pidly IDL wrapper

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

    grid: string, default: 'full'
        Which grid to get the synthetic spectrum from, must be one of:
         1) 'full' - the full unbroadened grid
         2) 'B3000' - grid pre-broadened to suit WiFeS B3000 spectra
         3) 'R7000' - grid pre-broadened to suit WiFeS R7000 spectra
        If either 'B3000' or 'R7000' are provided, no further broadening is 
        done, and both ipres and resolution are not required.

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

    abund_alpha, abund: float or None, default: None
        Alpha element and [C/Fe] abundance respectively, uses Nordlander IDL
        defaults when set to None.

    ebv: float, default: 0
        E(B-V) - stellar reddening.

    do_log_resample: bool, default: False
        Whether to resample onto a logarithmic wavelength scale.

    n_px: float, default: None
        Number of pixels for use when doing logarithmic scaling.

    Returns
    -------
    wave: 1D float array
        The wavelength scale for the synthetic spectra

    spectra: 1D float array
        Fluxes for the synthetic spectra.

    """
    NORM_VALS = {"abs":0, "norm":1, "cont":2, "abs_norm":-1}

    # Checks on input parameters
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

    if abund is None:
        idl("abund = 0.0;")
    else:
        idl("abund = {};".format(abund))

    # If resampling, initialise the output wavelength scale
    if do_resample:
        # Check if we're doing logarithmic resampling
        if do_log_resample:
            idl("wmin_log = ALog10(%f)" % (wl_min))
            idl("wmax_log = ALog10(%f)" % (wl_max))
            idl("exps = findgen(%f)/(%f-1)*(wmax_log - wmin_log) + wmin_log"
                % (n_px, n_px))
            #idl("exps = cgScaleVector(Findgen(%f), wmin_log, wmax_log)" % (n_px))
            idl("wout = 10.^exps")

        # Otherwise standard sampling
        else:
            idl("wout = [%f:%f:%f]" % (wl_min, wl_max, wl_per_px))
        
        wout = ", wout=wout"

        # Incorporate padding *after* we've set our output scale
        wl_min -= wave_pad
        wl_max += wave_pad
    else:
        wout = ""

    # Default behaviour: full, unbroadened grid
    if grid == "full":
        # Default behaviour is constant-velocity broadening, unless a value has 
        # been provided for resolution
        if resolution is None:
            cmd = ("spectrum = get_spec(%f, %f, %f, alpha, abund, %f, %f, "
                "ipres=%f, norm=%i, grid=grid, wave=wave, vrad=%f%s, ebmv=%f)"
                % (teff, logg, feh, wl_min, wl_max, ipres, norm_val, rv_bcor, 
                    wout, ebv))
        # Otherwise do constant-wavelength broadening
        else:
            cmd = ("spectrum = get_spec(%f, %f, %f, alpha, abund, %f, %f, "
                "norm=%i, resolution=%f, grid=grid, wave=wave, vrad=%f%s, ebmv=%f)" 
                % (teff, logg, feh, wl_min, wl_max, norm_val, resolution, 
                    rv_bcor, wout, ebv))

    # Generating WiFeS B3000 spectra. Grid has already been broadened.
    elif grid == "B3000":
        cmd = ("spectrum = get_spec(%f, %f, %f, alpha, abund, %f, %f, norm=%i, "
                "grid=grid_b, wave=wave, vrad=%f%s, ebmv=%f)" 
                % (teff, logg, feh, wl_min, wl_max, norm_val, rv_bcor, wout,
                   ebv))
    
    # Generating WiFeS R7000 spectra. Grid has already been broadened.
    elif grid == "R7000":
        cmd = ("spectrum = get_spec(%f, %f, %f, alpha, abund, %f, %f, norm=%i, "
                "grid=grid_r, wave=wave, vrad=%f%s, ebmv=%f)" 
                % (teff, logg, feh, wl_min, wl_max, norm_val, rv_bcor, wout,
                   ebv))
    
    # Mdwarf grid
    elif grid == "m_dwarf":
        cmd = ("spectrum = get_spec(%f, %f, %f, alpha, abund, %f, %f, "
                "ipres=%f, norm=%i, grid=grid_mdwarf, wave=wave, vrad=%f%s, ebmv=%f)"
                % (teff, logg, feh, wl_min, wl_max, ipres, norm_val, rv_bcor, 
                    wout, ebv))
    else:
        raise ValueError("Invalid grid")

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
def make_synth_mask_for_bad_wl_regions(
    wave, 
    rv, 
    bcor, 
    teff, 
    cutoff_temp=3800,
    mask_blue=True,
    mask_missing_opacities=True,
    mask_tio=True,
    mask_sodium_wings=True,
    low_cutoff=None,
    high_cutoff=None,):
    """Makes a bad pixel mask corresponding to where Thomas Nordlander's 
    synthetic MARCS models are inconsistent with observational data for cool
    (<~4,500 K) dwarf stars. The wavelength is corrected to the rest frame 
    before excluding.

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
        spectral regions. Setting this warmer than any stars is the same as
        assuming all synthetic pixels are reliable for the purpose of 
        generating a bad pixel mask.

    mask_blue: boolean, default: True
        Whether to bulk mask everything below 4,500 A.

    mask_missing_opacities: boolean, default: True
        Whether to mask out 5411-5424 A (from Mann+2013), 5498-5585 A (unknown
        feature), and 6029-6159 A (suspected Ca/Fe molecular line or 
        photodissociation opacity)

    mask_tio: boolean, default: True
        Whether to mask out suspected bad TiO regions 5615-5759 and 5809-5840 A

    mask_sodium_wings: boolean, default: True
        Whether to mask out the wings of the soudium doublet, 5847-5886 and
        5898-5987 A.

    low_cutoff, high_cutoff: float or None, default: None
        Low and high wavelength cutoff in A if required.

    Returns
    -------
    bad_reg_mask: bool array
        Bad pixel mask that is True where Thomas Nordlander's synthetic MARCS
        models are inconsistent with observed spectra. Array of False if teff
        is above cutoff temp
    """
    # Initialise mask for bad regions
    bad_reg_mask = np.zeros_like(wave)

    # Don't mask anything if we're above the cutoff temperature *and* we aren't
    # bulk masking outside of our two cutoffs
    if teff > cutoff_temp and low_cutoff is None and high_cutoff is None:
        return bad_reg_mask.astype(bool)

    # List of badly fitting regions for Thomas Nordlander's MARCS model library
    # These wavelengths are in the *rest frame*
    bad_regions = []

    if mask_blue:
        # Everything bluer than 4500 A (formerly 4,700 A)
        bad_regions.append([0, 4500])
    
    if mask_missing_opacities:
        # From Mann+2013 10 A, 10% dev test
        bad_regions.append([5411, 5424])

        # Unknown feature
        bad_regions.append([5498, 5585])
        
        # ~ Ca/Fe molecular line or photodissociation opacity
        bad_regions.append([6029, 6159])

    if mask_tio:
        bad_regions.append([5615, 5759])
        bad_regions.append([5809, 5840])

    if mask_sodium_wings:
        bad_regions.append([5847, 5886])
        bad_regions.append([5898, 5987])

    # Mask out anything outside of low and high cutoff if provided
    if (low_cutoff is not None and high_cutoff is not None 
        and low_cutoff < high_cutoff):
        bad_regions.append([0, low_cutoff])
        bad_regions.append([high_cutoff, 1E10])
    
    # Shift wavelength vector to rest frame
    wave_rf = wave * (1-(rv-bcor)/(const.c.si.value/1000))

    for region in bad_regions:
        bad_reg = np.logical_and(wave_rf >= region[0], wave_rf <= region[1])
        bad_reg_mask = np.logical_or(bad_reg_mask, bad_reg)

    return bad_reg_mask.astype(bool)
    

def get_template_spectra(
    teffs,
    loggs,
    fehs,
    vsinis=[1],
    setting="R7000",
    norm="abs",
    wl_min=None,
    wl_max=None,
    ipres=None,
    resolution=None,
    n_px=None,
    wl_per_px=None,
    do_log_resample=False,):
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

    TODO

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
        ipres = 7000
        n_px = 3637
        wl_per_px = 0.44

    elif setting == "B3000":
        wl_min = 3500
        wl_max = 5700
        ipres = 3000
        n_px = 2858
        wl_per_px = 0.77

    elif setting =="custom":
        assert (type(wl_min) is int) or (type(wl_min) is float)
        assert (type(wl_max) is int) or (type(wl_max) is float)
        assert wl_min < wl_max
        assert (ipres is not None) or (resolution is not None)

    else:
        raise ValueError("Unknown grating - choose either B3000 or R7000")
    
    # Determine the effective resolution based on vsini
    eff_rs = []

    for vsini in vsinis:
        eff_r = const.c.to("km/s").value / vsini

        # Take whichever the lower ipres is for the band in question
        if eff_r < ipres:
            eff_rs.append(eff_r)
        else:
            eff_rs.append(ipres)

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
                        ipres=eff_r,
                        resolution=resolution,
                        norm=norm,
                        do_resample=True, 
                        wl_per_px=wl_per_px,
                        n_px=n_px,
                        do_log_resample=do_log_resample)
                
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

def save_synthetic_templates_fits(
    wave,
    spectra,
    params,
    label,
    path="templates"):
    """Save the generated synthetic templates in templates/.

    TODO move

    Parameters
    ----------
    spectra: 3D float array
        The synthetic spectra in the form [N_star, n_pixel]. The stars
        are ordered by teff, then logg, then [Fe/H].

    params: TODO

    label: TODO
    """
    from astropy.io import fits
    from astropy.table import Table

    # Intialise HDU List
    hdu = fits.HDUList()

    # Assert that all wavelength scales are the same
    assert len(wave) == spectra.shape[1]
    
    # HDU 1: Wavelengths
    wave_img =  fits.PrimaryHDU(wave)
    wave_img.header["EXTNAME"] = ("WAVE", "Wavelength scale")
    hdu.append(wave_img)

    # HDU 2: Flux
    spec_img =  fits.PrimaryHDU(spectra)
    spec_img.header["EXTNAME"] = ("SPEC", "Flux")
    hdu.append(spec_img)

    # HDU 7: table of observational information
    params = pd.DataFrame(data=params, columns=["teff", "logg", "feh", "vsini"])
    obs_tab = fits.BinTableHDU(Table.from_pandas(params))
    obs_tab.header["EXTNAME"] = ("PARAMS", "Stellar parameters")
    hdu.append(obs_tab)

    # Done, save
    save_path = os.path.join(path,  "template_{}.fits".format(label))
    hdu.writeto(save_path, overwrite=True)


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
    ebv,
    rv, 
    bcor, 
    idl,
    band_settings,
    do_polynomial_spectra_norm,):
    """Calculates residuals between observed science spectrum, and MARCS model
    synthetic spectrum for a single arm of the spectrograph.

    Parameters
    ----------
    teff, logg, feh: float
        Stellar parameters for star.

    wave_sci, spec_sci, e_spec_sci: float array
        The science wavelength scale, spectrum, and uncertainties vectors.

    bad_px_mask: boolean array
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    ebv: float
        Reddening of star

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

    do_polynomial_spectra_norm: boolean
        Whether to use a polynomial to normalise out spectra prior to fitting.
        Main use case is for unfluxed spectra or testing.

    Returns
    -------
    resid_vect: float array
        Uncertainty weighted residual vector between science and synthetic
        spectra.

    spec_synth: float array
        Unnormalised synthetic spectrum at provided stellar parameters and 
        instrument settings.
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
        grid=band_settings["grid"],
        resolution=band_settings["wl_per_px"],
        norm="abs",
        do_resample=True, 
        wl_per_px=band_settings["wl_broadening"],
        rv_bcor=(rv-bcor),
        ebv=bcor,
        )

    # If we don't trust our flux calibration, normalise spectra by low order
    # polynomial, taking into account bad px mask
    if do_polynomial_spectra_norm:
        # Normalise the science spectra
        spec_sci_norm, e_spec_sci_norm = spec.normalise_spectrum(
            wave_sci,
            spec_sci,
            e_spec_sci,
            mask=~bad_px_mask,) # Note mask inversion

        # Normalise the synthetic spectrum only if it is non-zero. An array of 
        # zeros means that we generated a synthetic spectrum at non-physical 
        # values and returned with zero flux, but our parameters were still 
        # inside the  grid bounds. If it is composed of zeros, the residuals 
        # should be bad anyway.
        if np.sum(spec_synth) > 0:
            spec_synth_norm = spec.normalise_spectrum(
                wave_sci, 
                spec_synth, 
                mask=~bad_px_mask,)  # Note mask inversion

    # Otherwise assume spectra have been adequately fluxed, and normalise
    # spectra by wavelength region before fit
    else:
        spec_sci_norm, e_spec_sci_norm = spec.norm_spec_by_wl_region(
            wave_sci, 
            spec_sci, 
            band_settings["arm"], 
            e_spec_sci)

        # Normalise the synthetic spectrum only if it is non-zero (as above)
        if np.sum(spec_synth) > 0:
            spec_synth_norm = spec.norm_spec_by_wl_region(
                wave_sci, 
                spec_synth, 
                band_settings["arm"])

    # Mask the spectra by setting uncertainties on bad pixels to infinity 
    # (thereby setting the inverse variances to zero), as well as putting the
    # bad pixels themselves to one so there are no nans involved in calculation
    # of the residuals (as this breaks the least squares fitting)
    spec_sci_m = spec_sci_norm.copy()
    spec_sci_m[bad_px_mask] = 1
    e_spec_sci_m = e_spec_sci_norm.copy()
    e_spec_sci_m[bad_px_mask] = np.inf

    # Calculate the residual
    resid_vect = (spec_sci_m - spec_synth_norm) / e_spec_sci_m

    return resid_vect, spec_synth


def calc_bc_resid(
    teff,
    logg,
    feh,
    Mbol,
    stellar_phot,
    e_stellar_phot,
    phot_bands,
    bc_interp,):
    """Calculates the residuals between observed and synthetic photometry as
    determined from a bolometric correction.

    Parameters
    ----------
    teff, logg, feh, Mbol: float
        Stellar parameters for star.
    
    stellar_phot, e_stellar_phot: float array
        Arrays of observed stellar photometry and photometric uncertainties
        corresponding to phot_bands. 

    phot_bands: string array
        Photometric filters to be used in the fit, valid filters are:
        ["Bp", "Rp", "J", "H", "K", "v", "g", "r", "i", "z"]

    bc_interp: SyntheticBCInterpolator
        SyntheticBCInterpolator object able to interpolate bolometric 
        corrections for the given filters using the grid from:
        https://github.com/casaluca/bolometric-corrections

    Returns
    -------
    resid: float array
        Uncertainty weighted residual vector between observed and synthetic
        colours.
    
    synth_phot: float array
        Array of synthetic photometry corresponding to phot_bands.

    synth_bc: float array
        Array of synthetic bolometric corrections corresponding to phot_bands.
    """
    # Input checking
    if (len(stellar_phot) != len(e_stellar_phot) 
        or len(e_stellar_phot) != len(phot_bands)):
        raise ValueError("stellar_phot, e_stellar_phot, and phot_bands"
                         "  should all have the same length")

    # Calculate a set of synthetic bolometric corrections
    synth_bc = np.array([bc_interp.compute_bc((teff,logg,feh),pb) 
                              for pb in phot_bands])

    # Now calculate the magnitudes
    synth_phot = Mbol - synth_bc # + extinction

    # Calculate the residuals
    resid = (stellar_phot - synth_phot) / (e_stellar_phot)

    return resid, synth_phot, synth_bc


def calc_synth_fit_resid(
    params, 
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r,
    bcor, 
    idl,
    band_settings_r,
    params_fit_keys,
    params_fixed,
    band_settings_b,
    wave_b, 
    spec_b, 
    e_spec_b, 
    bad_px_mask_b,
    stellar_phot,
    e_stellar_phot,
    phot_bands,
    bc_interp,
    feh_offset,
    resid_norm_fac, 
    phot_scale_fac,
    suppress_fit_diagnostics=False,
    return_synth_models=False,
    do_polynomial_spectra_norm=False,):
    """Calculates the uncertainty weighted residuals between the combined
    and rescaled residual vectors of:
        a) Observed vs synthetic spectrum (red arm)
        b) Observed vs synthetic spectrum (blue arm)
        c) Observed vs synthetic photometry
    
    Note that in order to be combined, the residuals must be rescaled so that
    no one vector or kind of information dominates the fit (e.g. given that 
    a given spectral pixel does not contain an equivalent amount of information
    to a given photometric filter). This rescaling is done by doing a separate
    fit for each of a), b) and c) to find the global minimum chi^2. These 
    minima are passed in as the dictionary resid_norm_fac, but the separate 
    fits require this function to be run multiple times.

    If any of the associated vectors are None, they will not be included in 
    the fitting process.

    The two spectral fits (red and blue) require Teff, logg, and [Fe/H] for 
    their fits (any of which can be fixed). The photometric fit on the other
    hand requires mbol too, which is used as a (physically meaningful) scaling
    parameter to allow comparison between the observed and synthetic
    photometry, the latter of which would otherwise remain in the 'model 
    frame'. This mbol parameter can then later be used to determine fbol, the
    apparent bolometric flux, which in turn can be used to determine the 
    stellar radius.

    Note that currently photometry must be corrected for reddening ahead of
    time, while spectra is corrected here.

    Parameters
    ----------
    params: float array
        Initial parameter guess of form (teff, logg, [Fe/H])

    wave_r, spec_r, e_spec_r: float array or None
        The red wavelength scale, spectrum, and uncertainties vectors.

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

    ebv: float
        Reddening of form E(B-V).

    params_fit_keys: string list
        List of parameters that are to be fitted for, either:
         - 'teff'
         - 'logg'
         - 'feh'
         - 'mbol'

    params_fixed: dict
        Dictionary pairing of parameter ('teff', 'logg', 'feh', 'mbol') for
        those parameters that are fixed during fitting. Will contain those
        parameters not in params_fit_keys.

    band_settings_b: dict
        Dictionary with settings for WiFeS blue band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    wave_r, spec_b, e_spec_b: float array or None
        The blue wavelength scale, spectrum, and uncertainties vectors.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for blue arm
        corresponding to wave_b.

    stellar_phot, e_stellar_phot: float array or None
        Array of observed stellar photometry and photometric uncertainties
        corresponding to colour_bands.

    phot_bands: string array or None
        Photometric filters to use in fit.

    bc_interp: SyntheticBCInterpolator or None
        SyntheticBCInterpolator object able to interpolate synthetic bolometric
        corrections. If None, will not do photometric fit.

    feh_offset: float
        Arbitrary offset added to [Fe/H] so that it never goes below zero to
        improve compatability with diff_step in least_squares. As we're inside
        the least squares fitting here, we need to unscale [Fe/H].

    resid_norm_fac: dict
        Dictionary with keys ['red', 'blue', 'phot'] corresponding to the 
        minimum chi^2 for that specific residual vector to rescale it with.
        Set these all to 1 for the first pass of any fit.

    phot_scale_fac: float
        Scaling factor to better put photometry and spectroscopy on the same
        scale, taken to mean that every phot_scale_fac spectroscopic pixels 
        are correlated. Ballpark value to set this to, at least for cool 
        dwarfs, is ~20.

    suppress_fit_diagnostics: bool, default: False
        Whether to suppress printed diagnostics. 

    return_synth_models: bool, default: False
        Whether to return just the residual vector, or that in addition to the
        synthetic spectra and photometry at the given stellar parameters:
            spec_synth_b, spec_synth_r, synth_phot, synth_bc

    do_polynomial_spectra_norm: bool, default: False
        Whether to normalise spectra by a polynomial, mostly for testing
        purposes.

    Returns
    -------
    resid_vect: float array
        Uncertainty weighted residual vector between science and synthetic
        spectra.

    spec_synth_b, spec_synth_r, synth_phot, synth_bc: float arrays
        Synthetic spectra and photometry at the given stellar parameters. Only
        returned if return_synth_models is true.
    """
    # Unpack params, first teff
    ti = np.argwhere(params_fit_keys=="teff")
    teff = params[int(ti)] if len(ti) > 0 else params_fixed["teff"]

    # logg
    gi = np.argwhere(params_fit_keys=="logg")
    logg = params[int(gi)] if len(gi) > 0 else params_fixed["logg"]

    # [Fe/H] (and unscale)
    fi = np.argwhere(params_fit_keys=="feh")
    feh = params[int(fi)] - feh_offset if len(fi) > 0 else params_fixed["feh"]

    # Mbol
    mi = np.argwhere(params_fit_keys=="Mbol")
    Mbol = params[int(mi)] if len(mi) > 0 else params_fixed["Mbol"]

    # RV
    ri = np.argwhere(params_fit_keys=="rv")
    rv = params[int(ri)] if len(ri) > 0 else params_fixed["rv"]

    # E(B-V)
    ei = np.argwhere(params_fit_keys=="ebv")
    ebv = params[int(ei)] if len(ei) > 0 else params_fixed["ebv"]

    # Initialise boolean flags to indicate which residuals are included in the
    # fit - blue spectra, red spectra, and photometry
    used_blue = False
    used_red = False
    used_phot = False

    # Intitialise empty models
    spec_synth_b = None
    spec_synth_r = None
    synth_phot = None
    synth_bc = None

    # Initialise empty residual arrays
    resid_vect_r = []
    resid_vect_b = []
    resid_vect_bc = []

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Fit red
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ((wave_r is not None) and (spec_r is not None) 
        and (e_spec_r is not None) and (bad_px_mask_r is not None)):
        # Determine red residuals
        resid_vect_r, spec_synth_r = calc_synth_fit_resid_one_arm(
            teff, logg, feh, wave_r, spec_r, e_spec_r, bad_px_mask_r, ebv, rv,
            bcor, idl, band_settings_r, do_polynomial_spectra_norm)

        # Normalise residuals by minimum rchi^2
        resid_vect_r /= resid_norm_fac["red"]
        
        # Update flag
        used_red = True

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Fit blue
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ((wave_b is not None) and (spec_b is not None) 
        and (e_spec_b is not None) and (bad_px_mask_b is not None)):
        # Determine blue residuals
        resid_vect_b, spec_synth_b = calc_synth_fit_resid_one_arm(
            teff, logg, feh, wave_b, spec_b, e_spec_b, bad_px_mask_b, ebv, rv, 
            bcor, idl, band_settings_b, do_polynomial_spectra_norm)
        
        # Normalise residuals by minimum rchi^2
        resid_vect_b /= resid_norm_fac["blue"]
    
        # Update flag
        used_blue = True
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Fit colours
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if bc_interp is not None:
        resid_vect_bc, synth_phot, synth_bc = calc_bc_resid(
            teff, logg, feh, Mbol, stellar_phot, e_stellar_phot, phot_bands,
            bc_interp,)

        # Normalise residuals by minimum rchi^2
        resid_vect_bc /= resid_norm_fac["phot"]

        # Do additional scaling by multiplying by scale_fac. This increases the
        # weighting of the photometric colours, thus accounting for correlated
        # spectral pixels that add to the residuals, without adding extra
        # information to the fit.
        resid_vect_bc *= phot_scale_fac
    
        # Update flag
        used_phot = True

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Make final residual vector
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resid_vect = np.concatenate(
        (resid_vect_b, resid_vect_r, resid_vect_bc))

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Print diagnostics
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if not suppress_fit_diagnostics:
        # Print stellar param update
        line = ("Teff = {:0.5f} K, logg = {:0.05f}, [Fe/H] = {:+0.05f}, "
                "mbol={:+9.05f}, RV={:+7.2f} km/s, E(B-V)={:0.5f}\t")
        print(line.format(teff, logg, feh, Mbol, rv, ebv), end="")
        
        # Print synthetic colour update
        if bc_interp is not None:
            for pband, psynth in zip(phot_bands, synth_phot):
                print("{} = {:6.3f}, ".format(pband, psynth), end="")

        # Print whether we're using blue, red, and phot information
        print("\t[b={}, r={}, c={}]".format(
            used_blue, used_red, used_phot), end="")
        
        # And finally the rchi^2
        rchi2 = np.sum(resid_vect**2) / (len(resid_vect)-len(params))
        print(" --> rchi^2 = {:0.5f}".format(rchi2))

    if return_synth_models:
        return resid_vect, spec_synth_b, spec_synth_r, synth_phot, synth_bc
    else:
        return resid_vect


def do_synthetic_fit(
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r, 
    params, 
    bcor,
    idl,
    band_settings_r,
    fit_for_params={
        "teff":True,
        "logg":True,
        "feh":True,
        "Mbol":True,
        "rv":False,
        "ebv":False,},
    band_settings_b=None,
    wave_b=None, 
    spec_b=None, 
    e_spec_b=None, 
    bad_px_mask_b=None,
    stellar_phot=None,
    e_stellar_phot=None,
    phot_bands=None,
    phot_bands_all=None,
    feh_offset=10,
    scale_threshold={"blue":1,"red":1,"phot":1},
    fit_for_resid_norm_fac=False,
    phot_scale_fac=1,
    suppress_fit_diagnostics=False,
    do_polynomial_spectra_norm=False,
    teff_bounds=(2800,8000),
    logg_bounds=(4,5.5),
    feh_bounds=(-2,0.5),
    mbol_bounds=(-10,100),
    rv_bounds=(-500,500),
    ebv_bounds=(0,10),
    ls_scale=np.array([1,1,1,1,1,1]),
    ls_step=np.array([0.1,0.1,0.1,0.1,0.01,0.01]),):
    """Performs least squares fitting (using scipy.optimize.least_squares) on
    the combined residual vectors of:
        a) Observed vs synthetic spectrum (red arm)
        b) Observed vs synthetic spectrum (blue arm)
        c) Observed vs synthetic photometry
    
    Where any of
    This rescaling is done by doing a separate
    fit for each of a), b) and c) to find the global minimum chi^2. After this,
    a final simultaneous fit is done.

    Note: currently do not rescale blue residuals.

    TODO: use parameters as a dictionary, rather than individually as we 
    currently do.

    Parameters
    ----------
    wave_r, spec_r, e_spec_r: float array
        The red wavelength scale, spectrum, and uncertainties vectors.

    bad_px_mask_r: boolean array
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    params: dict
        Dictionary containing values for the parameters involved in the fit,
        can be either initial guesses or fixed values, with the interpretation
        set by the dictionary fit_for_params. Currently supported keys are:
        ['teff', 'logg', 'feh',  'Mbol']

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

    ebv: float
        Reddening of form E(B-V).

    fit_for_params: dict, 
        default: {'teff':True, 'logg':True, 'feh':True, 'Mbol':True}
        Dictionary mapping stellar parameters to a boolean, where a True
        indicates the parameter will be fitted for as part of the least sqaures
        fit, and False indicates the parameter will be fixed.

    band_settings_b: dict, default: None
        Dictionary with settings for WiFeS blue band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    wave_b, spec_b, e_spec_b: float array
        The blue wavelength scale, spectrum, and uncertainties vectors.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for blue arm
        corresponding to wave_b.

    stellar_phot, e_stellar_phot: float array or None
        Array of observed stellar photometry and photometric uncertainties
        corresponding to colour_bands.

    phot_bands: string array or None
        Photometric filters to use in fit, which we *must* have observed
        equivalents for.

    phot_bands_all: string array or None
        Photometric filters we want to generate synethetic magnitudes for at
        the conclusion of the fit, which we don't need observed equivalents.
    
    feh_offset: float
        Arbitrary offset added to [Fe/H] so that it never goes below zero to
        improve compatability with diff_step in least_squares. 

    scale_threshold: dict, default: {'blue':1,'red':1,'phot':1}
        Dictionary with keys ['red', 'blue', 'phot'] corresponding to the 
        minimum chi^2 for which we rescale that specific residual vector.

    phot_scale_fac: float, default: 1
        Scaling factor to better put photometry and spectroscopy on the same
        scale, taken to mean that every phot_scale_fac spectroscopic pixels 
        are correlated. Ballpark value to set this to, at least for cool 
        dwarfs, is ~20.

    suppress_fit_diagnostics: bool, default: False
        Whether to suppress printed diagnostics. 

    do_polynomial_spectra_norm: bool, default: False
        Whether to normalise spectra by a polynomial, mostly for testing
        purposes.

    teff_bounds: float array, default:(2800,8000)
        Lower and upper limits on Teff when performing least squares fitting.

    logg_bounds: float array, default:(4,5.5)
        Lower and upper limits on logg when performing least squares fitting.

    feh_bounds: float array, default:(-2,0.5)
        Lower and upper limits on [Fe/H] when performing least squares fitting.
    
    mbol_bounds: float array, default:(-10,100)
        Lower and upper limits on mbol when performing least squares fitting.
    
    rv_bounds: float array, default:(-500,500)
        Lower and upper limits on RV when performing least squares fitting.
    
    ebv_bounds: float array, default:(0,10)
        Lower and upper limits on E(B-V) when performing least squares fitting.
    
    ls_scale: float array, default:(1,1,1,1,1,1)
        Scaling factor to use for each parameter when doing least squares fit.

    ls_step: float array, default:(0.1,0.1,0.1,0.1,0.01,0.01)
        Fractional step size to use for each parameter when doing least squares 
        fit.

    Returns
    -------
    optimize_result: dict
        Dictionary of best fit results returned from 
        scipy.optimize.least_squares.
    """
    # Initalise boundary conditions for each parameter, applying [Fe/H] offset
    bounds = np.array([
        # lower bounds
        (teff_bounds[0], logg_bounds[0], feh_bounds[0]+feh_offset,
          mbol_bounds[0], rv_bounds[0], ebv_bounds[0],),
        # upper bounds
        (teff_bounds[1], logg_bounds[1], feh_bounds[1]+feh_offset,
          mbol_bounds[1], rv_bounds[1], ebv_bounds[1],)
    ])

    # Scale [Fe/H] if fitting for
    if fit_for_params["feh"]:
        params["feh"] += feh_offset

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialise photometry
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Intialises bc_interp if provided colours, otherwise None
    if stellar_phot is None or e_stellar_phot is None:
        bc_interp = None
        fit_for_mbol = False

    # If we've been given photometry, initialise synthetic colour interpolator
    else:
        bc_interp = SyntheticBCInterpolator()
        fit_for_mbol = True

    # Based on whether we have been provided photometry, determine whether 
    # we're fitting for Mbol
    fit_for_params["Mbol"] = fit_for_mbol

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Fit with red spectra alone, find min chi^2
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Only proceed if we don't already know how we're doing the rescaling
    if fit_for_resid_norm_fac:
        print("\nFinding global red chi^2 minima\n" + "-"*31)

        # Don't fit for Mbol
        fit_for_params["Mbol"] = False
        param_mask = list(fit_for_params.values())

        # Initialise resid_norm_fac to be ones
        resid_norm_fac = {'blue':1, 'red':1, 'phot':1,}

        # Initialise param initial guess *to be fit for* as a list, save keys
        params_init = [params[pp] for pp in params if fit_for_params[pp]]
        params_init_keys = np.array([pp for pp in params if fit_for_params[pp]])

        # Keep the rest in dictionary form
        params_fixed = {pp:params[pp] for pp in params if not fit_for_params[pp]}
        
        # Setup fit settings
        args = (wave_r, spec_r, e_spec_r, bad_px_mask_r, bcor , idl, 
                band_settings_r, params_init_keys, params_fixed, 
                band_settings_b, None, None, None,  None, None, 
                None, None, None, feh_offset, 
                resid_norm_fac, phot_scale_fac, suppress_fit_diagnostics,
                False, do_polynomial_spectra_norm,)

        # Do fit
        opt_res = least_squares(
            calc_synth_fit_resid, 
            params_init, 
            jac="3-point",
            bounds=bounds[:,param_mask],
            x_scale=ls_scale[param_mask],
            diff_step=ls_step[param_mask],
            args=args, 
        )

        # Calculate residual scaling for red spectra
        min_chi2_red = np.sum(opt_res["fun"]**2)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Fit with photometry alone, find min chi^2
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print("\nFinding global photometry chi^2 minima\n" + "-"*38)

        # Don't fit for Mbol
        fit_for_params["Mbol"] = True
        param_mask = list(fit_for_params.values())

        # Initialise param initial guess *to be fit for* as a list, save keys
        params_init = [params[pp] for pp in params if fit_for_params[pp]]
        params_init_keys = np.array([pp for pp in params if fit_for_params[pp]])

        # Keep the rest in dictionary form
        params_fixed = {pp:params[pp] for pp in params if not fit_for_params[pp]}
        
        # Setup fit settings
        args = (None, None, None, None, bcor , idl, 
                band_settings_r, params_init_keys, params_fixed, 
                band_settings_b, None, None, None,  None, stellar_phot, 
                e_stellar_phot, phot_bands, bc_interp, feh_offset, 
                resid_norm_fac, phot_scale_fac, suppress_fit_diagnostics,
                False, do_polynomial_spectra_norm,)

        # Do fit
        opt_res = least_squares(
            calc_synth_fit_resid, 
            params_init, 
            jac="3-point",
            bounds=bounds[:,param_mask],
            x_scale=ls_scale[param_mask],
            diff_step=ls_step[param_mask],
            args=args, 
        )

        # Calculate residual scaling for red spectra
        min_chi2_phot = np.sum(opt_res["fun"]**2)

        # Now setup residual scaling
        if min_chi2_red > scale_threshold["red"]:
            resid_norm_fac["red"] = min_chi2_red**0.5
        else:
            resid_norm_fac["red"] = 1
        
        if min_chi2_phot > scale_threshold["phot"]:
            resid_norm_fac["phot"] = min_chi2_phot**0.5
        else:
            resid_norm_fac["phot"] = 1

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Do combined fit and scale residuals
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("\nDoing final global\n" + "-"*18)
    # Setup a mask for slicing fit bounds/scaling/step
    param_mask = list(fit_for_params.values())

    # Initialise param initial guess *to be fit for* as a list, save keys
    params_init = [params[pp] for pp in params if fit_for_params[pp]]
    params_init_keys = np.array([pp for pp in params if fit_for_params[pp]])

    # Keep the rest in dictionary form
    params_fixed = {pp:params[pp] for pp in params if not fit_for_params[pp]}
    
    # Setup fit settings
    args = (wave_r, spec_r, e_spec_r, bad_px_mask_r, bcor , idl, 
            band_settings_r, params_init_keys, params_fixed, band_settings_b, 
            wave_b, spec_b, e_spec_b,  bad_px_mask_b, stellar_phot, 
            e_stellar_phot, phot_bands, bc_interp, feh_offset, 
            resid_norm_fac, phot_scale_fac, suppress_fit_diagnostics,
            False, do_polynomial_spectra_norm,)

    # Do fit
    opt_res = least_squares(
        calc_synth_fit_resid, 
        params_init, 
        jac="3-point",
        bounds=bounds[:,param_mask],
        x_scale=ls_scale[param_mask],
        diff_step=ls_step[param_mask],
        args=args, 
    )

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sort out fit results
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Get the length of each of the sub-residual vectors
    len_b = 0 if spec_b is None else len(spec_b)
    len_r = 0 if spec_r is None else len(spec_r)
    len_p = 0 if stellar_phot is None else len(stellar_phot)

    # Pull out the residuals
    res = opt_res["fun"]

    # Rescale the residuals in three components (blue, red, phot)
    unscaled_res = np.concatenate((
        res[:len_b] * resid_norm_fac["blue"],
        res[len_b:(len_b+len_r)] * resid_norm_fac["red"],
        res[-len_p:] * resid_norm_fac["phot"] / phot_scale_fac,
    ))

    assert len(unscaled_res) == len(res)

    # Calculate final scale factor to unscale the residuals. Note that this is
    # a bit of a hack, as using unscaled_res in calculation of std results in
    # a colossal variance, which then results in a huge std.
    unscale_fac = np.sum(unscaled_res) / np.sum(res)

    # Calculate RMS to scale uncertainties by
    rms = np.sqrt(np.sum(res**2)/np.sum(res != 0))

    # Calculate uncertainties
    jac = opt_res["jac"]
    cov = np.linalg.inv(jac.T.dot(jac))
    std = np.sqrt(np.diagonal(cov)) * rms# * unscale_fac
    opt_res["std"] = std

    # Teff
    ti = np.argwhere(params_init_keys=="teff")
    opt_res["teff"] = opt_res["x"][int(ti)] if len(ti) > 0 else params["teff"]
    opt_res["e_teff"] = opt_res["std"][int(ti)] if len(ti) > 0 else np.nan

    # logg
    gi = np.argwhere(params_init_keys=="logg")
    opt_res["logg"] = opt_res["x"][int(gi)] if len(gi) > 0 else params["logg"]
    opt_res["e_logg"] = opt_res["std"][int(gi)] if len(gi) > 0 else np.nan

    # [Fe/H]
    fi = np.argwhere(params_init_keys=="feh")
    opt_res["feh"] = opt_res["x"][int(fi)] if len(fi) > 0 else params["feh"]
    opt_res["e_feh"] = opt_res["std"][int(fi)] if len(fi) > 0 else np.nan

    # Mbol
    mi = np.argwhere(params_init_keys=="Mbol")
    opt_res["Mbol"] = opt_res["x"][int(mi)] if len(mi) > 0 else params["Mbol"]
    opt_res["e_Mbol"] = opt_res["std"][int(mi)] if len(mi) > 0 else np.nan

    # RV
    ri = np.argwhere(params_init_keys=="rv")
    opt_res["rv"] = opt_res["x"][int(ri)] if len(ri) > 0 else params["rv"]
    opt_res["e_rv"] = opt_res["std"][int(ri)] if len(ri) > 0 else np.nan

    # E(B-V)
    ei = np.argwhere(params_init_keys=="Mbol")
    opt_res["ebv"] = opt_res["x"][int(ei)] if len(ei) > 0 else params["ebv"]
    opt_res["e_ebv"] = opt_res["std"][int(ei)] if len(ei) > 0 else np.nan

    # Unscale [Fe/H] if fitted for
    if fit_for_params["feh"]:
        opt_res["feh"] -= feh_offset

    # Store the scaling params
    opt_res["resid_norm_fac_red"] = resid_norm_fac["red"]
    opt_res["resid_norm_fac_phot"] = resid_norm_fac["phot"]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Get synthetic spectra and colours at optimal params
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Generate, normalise and save synthetic spectra at optimal params
    if band_settings_b is not None:
        _, spec_synth_b = get_idl_spectrum(
            idl, 
            opt_res["teff"], 
            opt_res["logg"], 
            opt_res["feh"], 
            wl_min=band_settings_b["wl_min"], 
            wl_max=band_settings_b["wl_max"], 
            ipres=band_settings_b["inst_res_pow"],
            grid=band_settings_b["grid"],
            resolution=band_settings_b["wl_broadening"],
            norm="abs",
            do_resample=True, 
            wl_per_px=band_settings_b["wl_per_px"],
            rv_bcor=(opt_res["rv"]-bcor),
            )

        spec_synth_b_norm = spec.norm_spec_by_wl_region(
            wave_b, 
            spec_synth_b, 
            band_settings_b["arm"])
    else:
        spec_synth_b_norm = None

    if band_settings_r is not None:
        _, spec_synth_r = get_idl_spectrum(
            idl, 
            opt_res["teff"], 
            opt_res["logg"], 
            opt_res["feh"], 
            wl_min=band_settings_r["wl_min"], 
            wl_max=band_settings_r["wl_max"], 
            ipres=band_settings_r["inst_res_pow"],
            grid=band_settings_r["grid"],
            resolution=band_settings_r["wl_broadening"],
            norm="abs",
            do_resample=True, 
            wl_per_px=band_settings_r["wl_per_px"],
            rv_bcor=(opt_res["rv"]-bcor),
            )

        spec_synth_r_norm = spec.norm_spec_by_wl_region(
            wave_r, 
            spec_synth_r, 
            band_settings_r["arm"])
    
    else:
        spec_synth_r_norm = None

    # Generate bolometric corrections at the final params
    if bc_interp is not None:
        synth_bc = np.array(
            [bc_interp.compute_bc(
                (opt_res["teff"],opt_res["logg"],opt_res["feh"]),band) 
            for band in phot_bands_all])

        synth_phot = opt_res["Mbol"] - synth_bc

    else:
        synth_bc = None
        synth_phot = None

    # Add synthetic spectra and synthetic colours to return dict
    opt_res["spec_synth_b"] = spec_synth_b_norm
    opt_res["spec_synth_r"] = spec_synth_r_norm
    opt_res["synth_bc"] = synth_bc
    opt_res["synth_phot"] = synth_phot

    # Calculate rchi^2
    opt_res["rchi2"] = (np.sum(opt_res["fun"]**2) 
                        / (len(opt_res["fun"])-len(params_init)))

    return opt_res


def make_chi2_map(
    teff_actual,
    logg_actual,
    feh_actual,
    wave_r, 
    spec_r, 
    e_spec_r, 
    bad_px_mask_r, 
    rv, 
    bcor, 
    idl, 
    band_settings_r, 
    band_settings_b, 
    wave_b, 
    spec_b, 
    e_spec_b,  
    bad_px_mask_b, 
    stellar_phot, 
    e_stellar_phot, 
    phot_bands,
    teff_span=400,
    feh_span=1.0, 
    n_fits=100,
    feh_offset=0, 
    phot_scale_fac=1,
    teff_lims=(2500,8000),
    feh_lims=(-2,0.5),
    n_feh_valley_pts=20,
    scale_residuals={"blue":True,"red":True,"phot":False},
    scale_threshold={"blue":0,"red":0,"phot":1}):
    """Samples the chi^2 space in Teff and [Fe/H] in a box around central 
    literature values of Teff and [Fe/H].

    Note: This function predates the new implementation of do_synthetic_fit,
    and hasn't been refactored yet to suit and this likely doesn't work.

    Parameters
    ----------
    teff_actual, logg_actual, feh_actual: float
        Literature parameters for the star to be used as the centre of the
        chi^2 'box'.

    wave_r, spec_r, e_spec_r: float array
        The red wavelength scale, spectrum, and uncertainties vectors.

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

    params_fit_keys: string list
        List of parameters that are to be fitted for, either:
         - 'teff'
         - 'logg'
         - 'feh'

    params_fixed: dict
        Dictionary pairing of parameter ('teff', 'logg', 'feh') for those
        parameters that are fixed during fitting. Will contain those parameters
        not in params_fit_keys.

    band_settings_b, band_settings_r: dict
        Dictionary with settings for WiFeS B and R band, used when generating 
        synthetic spectra. Has keys: ["inst_res_pow", "wl_min", "wl_max",
        "n_px", "wl_per_px", "wl_broadening", "arm"]

    wave_b, spec_b, e_spec_b: float array
        The blue wavelength scale, spectrum, and uncertainties vectors.

    bad_px_mask_b: boolean, optional
        Array of bad pixels (i.e. bad pixels are True) for blue arm
        corresponding to wave_b.

    stellar_colours: float array, default: None
        Array of observed stellar colour corresponding to colour_bands. If None
        photometry is not used in the fit.

    e_stellar_colours: float array, default: None
        Array of observed stellar colour uncertainties. If None photometry is 
        not used in the fit.

    colour_bands: string array, default: ['Rp-J', 'J-H', 'H-K']
        Colour bands to use in the fit.

    teff_span, feh_spane: float, default: 400, 1.0
        Width of the chi^2 box in Teff and [Fe/H], centred on the literature
        values of Teff and [Fe/H].
    
    n_fits: int, default: 100
        Number of random samples to make in Teff-[Fe/H] for the chi^2 map.

    feh_offset: float, default: 10
        Arbitrary offset to add to [Fe/H] so that it never goes below zero to
        improve compatability with diff_step.

    scale_fac: int, default: 100
        Scaling factor to weight the residuals from photometry versus those 
        from the spectra.

    teff_lims, feh_lims: float tuple, default: (2500, 8000), (-2, 0.5)
        Limits for Teff and [Fe/H] when sampling.

    feh_min_step: float, default: 0.05
        Step size in [Fe/H] for mapping the valley floor.

    Returns
    -------
    teffs, fehs, rchi2s: float array
        Sampled Teffs, [Fe/H] and resulting rchi^2s.
    """
    # Set Teff bounds to not go outside of grid
    teff_smin = teff_actual - teff_span/2
    teff_smax = teff_actual + teff_span/2

    teff_min = teff_smin if teff_smin > teff_lims[0] else teff_lims[0]
    teff_max = teff_smax if teff_smax < teff_lims[1] else teff_lims[1]

    # Set [Fe/H] bounds to not go outside of grid
    feh_smin = feh_actual - feh_span/2
    feh_smax = feh_actual + feh_span/2

    feh_min = feh_smin if feh_smin > feh_lims[0] else feh_lims[0]
    feh_max = feh_smax if feh_smax < feh_lims[1] else feh_lims[1]

    # Generate n_fits realisations of our parameters
    teffs = np.linspace(teff_min, teff_max, n_fits+1)
    fehs = np.linspace(feh_min, feh_max, n_fits+1)

    # Meshgrid
    tt, ff = np.meshgrid(teffs, fehs)

    grid_teffs = tt.flatten()
    grid_fehs = ff.flatten()

    # Initialise our return vectors
    synth_spectra_r = []
    grid_resid = []

    # Initialise our synthetic colour interpolator
    bc_interp = SyntheticBCInterpolator()

    resid_norm_fac = {'blue':1, 'red':1, 'phot':1,}

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Compute residuals for our *grid* of Teff and [Fe/H]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for teff, feh in zip(tqdm(grid_teffs, desc="fitting grid"), grid_fehs):
        # Prepare the parameter structures
        """
        params_init = np.array([teff, feh, 0])
        params_init_keys = np.array(["teff", "feh", "Mbol"])
        params_fixed = {"logg":logg_actual,}

        # Compute residuals at given params
        
        resid, _, synth_spec_r, _, _ = calc_synth_fit_resid(
            params_init, wave_r, spec_r,  e_spec_r, bad_px_mask_r, rv, bcor , 
            idl, band_settings_r, params_init_keys, params_fixed, 
            band_settings_b, wave_b, spec_b, e_spec_b, bad_px_mask_b, 
            stellar_phot, e_stellar_phot, phot_bands, bc_interp, 
            feh_offset, resid_norm_fac=resid_norm_fac, phot_scale_fac=1, 
            suppress_fit_diagnostics=True, return_synth_models=True,)
        """
        params_init = {"teff":teff, "logg":logg_actual, "feh":feh, "Mbol":0}
        fit_for_params = OrderedDict([
            ("teff",False), ("logg",False), ("feh",False), ("Mbol",True)])

        opt_res = do_synthetic_fit(
            wave_r=wave_r,
            spec_r=spec_r,
            e_spec_r=e_spec_r,
            bad_px_mask_r=bad_px_mask_r,
            params=params_init, 
            rv=rv, 
            bcor=bcor,
            idl=idl,
            band_settings_r=band_settings_r,
            fit_for_params=fit_for_params,
            band_settings_b=band_settings_b,
            wave_b=wave_b, 
            spec_b=spec_b, 
            e_spec_b=e_spec_b, 
            bad_px_mask_b=bad_px_mask_b,
            stellar_phot=stellar_phot,
            e_stellar_phot=e_stellar_phot,
            phot_bands=phot_bands,
            resid_norm_fac=resid_norm_fac,
            phot_scale_fac=phot_scale_fac,
            suppress_fit_diagnostics=True,)

        synth_spectra_r.append(opt_res["spec_synth_r"])
        grid_resid.append(opt_res["fun"])

    grid_resid = np.stack(grid_resid)
    synth_spectra_r = np.stack(synth_spectra_r)

    # Now for each of our [Fe/H] grid points, find the optimal Teff, thus 
    # mapping out the valley floor
    # Setup for valley determination
    valley_fehs = np.linspace(feh_min, feh_max, n_feh_valley_pts)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Determine location of valley for blue spectra
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if spec_b is not None:
        # Intitialise
        valley_teffs_b = []
        valley_resid_b = []

        for feh in tqdm(valley_fehs, desc="Finding blue valley"):
            # Prepare the parameter structures
            params_init = {
                "teff":teff_actual, "logg":logg_actual, "feh":feh, "Mbol":0}
            fit_for_params = OrderedDict([
                ("teff",True), ("logg",False), ("feh",False), ("Mbol",False)])

            opt_res = do_synthetic_fit(
                wave_r=None, # Red wl
                spec_r=None, # Red spec
                e_spec_r=None, # Red uncertainties
                bad_px_mask_r=None,
                params=params_init, 
                rv=rv, 
                bcor=bcor,
                idl=idl,
                band_settings_r=None,
                fit_for_params=fit_for_params,
                band_settings_b=band_settings_b,
                wave_b=wave_b, 
                spec_b=spec_b, 
                e_spec_b=e_spec_b, 
                bad_px_mask_b=bad_px_mask_b,
                stellar_phot=None,
                e_stellar_phot=None,
                phot_bands=None,
                phot_scale_fac=1,
                suppress_fit_diagnostics=True,)

            valley_teffs_b.append(float(opt_res["x"]))
            valley_resid_b.append(opt_res["fun"])
        
        valley_teffs_b = np.array(valley_teffs_b)
        valley_resid_b = np.stack(valley_resid_b)

        # Calculate
        n_b = len(spec_b)

        min_chi2_b = np.min(np.sum(valley_resid_b**2, axis=1))

        if scale_residuals["blue"] and min_chi2_b > scale_threshold["blue"]:
            resid_scale_fac_b = min_chi2_b**0.5

        else:
            resid_scale_fac_b = 1

        grid_resid_b = grid_resid[:, :n_b] / resid_scale_fac_b
        grid_rchi2_b = np.sum(grid_resid_b**2, axis=1)# / (n_b - 2)
        
        valley_resid_b /=  resid_scale_fac_b
        valley_rchi2_b = np.sum(valley_resid_b**2, axis=1)# / (n_b - 2)

    # No blue, so default
    else:
        n_b = 0
        valley_teffs_b = np.ones_like(valley_fehs) * np.nan
        resid_scale_fac_b = np.nan

        grid_resid_b = np.empty((grid_resid.shape[0], 0))
        grid_rchi2_b = np.nan

        valley_resid_b = np.empty((len(valley_fehs), 0))
        valley_rchi2_b = np.ones_like(valley_fehs) * np.nan

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Determine location of valley for red spectra
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if spec_r is not None:
        # Intitialise
        valley_teffs_r = []
        valley_resid_r = []

        for feh in tqdm(valley_fehs, desc="Finding red valley"):
            # Prepare the parameter structures
            params_init = {
                "teff":teff_actual, "logg":logg_actual, "feh":feh, "Mbol":0}
            fit_for_params = OrderedDict([
                ("teff",True), ("logg",False), ("feh",False), ("Mbol",False)])

            opt_res = do_synthetic_fit(
                wave_r=wave_r,
                spec_r=spec_r,
                e_spec_r=e_spec_r,
                bad_px_mask_r=bad_px_mask_r,
                params=params_init, 
                rv=rv, 
                bcor=bcor,
                idl=idl,
                band_settings_r=band_settings_r,
                fit_for_params=fit_for_params,
                band_settings_b=None,
                wave_b=None, 
                spec_b=None, 
                e_spec_b=None, 
                bad_px_mask_b=None,
                stellar_phot=None,
                e_stellar_phot=None,
                phot_bands=None,
                phot_scale_fac=1,
                suppress_fit_diagnostics=True,)

            valley_teffs_r.append(float(opt_res["x"]))
            valley_resid_r.append(opt_res["fun"])
        
        valley_teffs_r = np.array(valley_teffs_r)
        valley_resid_r = np.stack(valley_resid_r)

        # Calculate
        n_r = len(spec_r)
        
        min_chi2_r = np.min(np.sum(valley_resid_r**2, axis=1))

        if scale_residuals["red"] and min_chi2_r > scale_threshold["red"]:
            resid_scale_fac_r = min_chi2_r**0.5
        else:
            resid_scale_fac_r = 1
        
        grid_resid_r = grid_resid[:, n_b:(n_b+n_r)] / resid_scale_fac_r
        grid_rchi2_r = np.sum(grid_resid_r**2, axis=1)# / (n_r - 2)
        
        valley_resid_r /=  resid_scale_fac_r
        valley_rchi2_r = np.sum(valley_resid_r**2, axis=1)# / (n_r - 2)

    # No red, so default
    else:
        n_r = 0
        valley_teffs_r = np.ones_like(valley_fehs) * np.nan
        resid_scale_fac_r = np.nan

        grid_resid_r = np.empty((grid_resid.shape[0], 0))
        grid_rchi2_r = np.nan

        valley_resid_r = np.empty((len(valley_fehs), 0))
        valley_rchi2_r = np.ones_like(valley_fehs) * np.nan

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Determine location of valley for photometry
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if stellar_phot is not None:
        # Intitialise
        valley_teffs_p = []
        valley_resid_p = []

        for feh in tqdm(valley_fehs, desc="Finding colour valley"):
            # Prepare the parameter structures
            params_init = {
                "teff":teff_actual, "logg":logg_actual, "feh":feh, "Mbol":0}
            fit_for_params = OrderedDict([
                ("teff",True), ("logg",False), ("feh",False), ("Mbol",True)])

            opt_res = do_synthetic_fit(
                wave_r=None,
                spec_r=None,
                e_spec_r=None,
                bad_px_mask_r=None,
                params=params_init, 
                rv=rv, 
                bcor=bcor,
                idl=idl,
                band_settings_r=None,
                fit_for_params=fit_for_params,
                band_settings_b=None,
                wave_b=None, 
                spec_b=None, 
                e_spec_b=None, 
                bad_px_mask_b=None,
                stellar_phot=stellar_phot,
                e_stellar_phot=e_stellar_phot,
                phot_bands=phot_bands,
                phot_scale_fac=1,
                suppress_fit_diagnostics=True,)

            valley_teffs_p.append(float(opt_res["x"][0]))
            valley_resid_p.append(opt_res["fun"])
        
        valley_teffs_p = np.array(valley_teffs_p)
        valley_resid_p = np.stack(valley_resid_p)

        # Calculate
        n_p = len(stellar_phot)

        min_chi2_p = np.min(np.sum(valley_resid_p**2, axis=1))

        if scale_residuals["phot"] and min_chi2_p > scale_threshold["phot"]:
            resid_scale_fac_p = min_chi2_p**0.5

        else:
            resid_scale_fac_p = 1
        
        grid_resid_p = grid_resid[:, -n_p:] / resid_scale_fac_p
        grid_rchi2_p = np.sum(grid_resid_p**2, axis=1)# / (n_p - 2)
        
        valley_resid_p /=  resid_scale_fac_p
        valley_rchi2_p = np.sum(valley_resid_p**2, axis=1)# / (n_p - 2)

    # No colours, so default
    else:
        n_p = 0
        valley_teffs_p = np.ones_like(valley_fehs) * np.nan
        resid_scale_fac_p = np.nan

        grid_resid_p = np.empty((grid_resid.shape[0], 0))
        grid_rchi2_p = np.nan

        valley_resid_p = np.empty((len(valley_fehs), 0))
        valley_rchi2_p = np.ones_like(valley_fehs) * np.nan

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Determine location of valley with everything
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Intitialise
    valley_teffs_all = []
    valley_resid_all = []

    # Initialise the scaling
    resid_norm_fac = {
        'blue':resid_scale_fac_b,
        'red':resid_scale_fac_r,
        'phot':resid_scale_fac_p,}

    for feh in tqdm(valley_fehs, desc="Finding final weighted valley"):
        # Prepare the parameter structures
        params_init = {
                "teff":teff_actual, "logg":logg_actual, "feh":feh, "Mbol":0}
        fit_for_params = OrderedDict([
            ("teff",True), ("logg",False), ("feh",False), ("Mbol",True)])

        opt_res = do_synthetic_fit(
            wave_r=wave_r,
            spec_r=spec_r,
            e_spec_r=e_spec_r,
            bad_px_mask_r=bad_px_mask_r,
            params=params_init, 
            rv=rv, 
            bcor=bcor,
            idl=idl,
            band_settings_r=band_settings_r,
            fit_for_params=fit_for_params,
            band_settings_b=band_settings_b,
            wave_b=wave_b, 
            spec_b=spec_b, 
            e_spec_b=e_spec_b, 
            bad_px_mask_b=bad_px_mask_b,
            stellar_phot=stellar_phot,
            e_stellar_phot=e_stellar_phot,
            phot_bands=phot_bands,
            resid_norm_fac=resid_norm_fac,
            phot_scale_fac=phot_scale_fac,
            suppress_fit_diagnostics=True,)

        valley_teffs_all.append(float(opt_res["x"][0]))
        valley_resid_all.append(opt_res["fun"])
    
    n_a = n_b + n_r + n_p
    valley_teffs_all = np.array(valley_teffs_all)
    valley_resid_all = np.stack(valley_resid_all)

    valley_rchi2_all = np.sum(valley_resid_all**2, axis=1)# / (n_a - 2)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Get broadband spectra at literature values, plus top and bottom of valley
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    black_body_synth_settings = {
        "ipres":2000,
        "wl_min":3000,
        "wl_max":60000,
        "wl_per_px":1.0,
        "grid":"full",
        "norm":"abs",
        "do_resample":True,
        "rv_bcor":(rv-bcor),
    }
    
    # Top of valley - i.e. metal rich
    _, spec_bb_high_feh = get_idl_spectrum(
        idl, valley_teffs_r[0], logg_actual, valley_fehs[0], 
        **black_body_synth_settings,)

    # Bottom of valley - i.e. metal poor
    _, spec_bb_low_feh = get_idl_spectrum(
        idl, valley_teffs_r[-1], logg_actual, valley_fehs[-1], 
        **black_body_synth_settings,)

    # And at literature value
    wave_black_body, spec_bb_lit = get_idl_spectrum(
        idl, teff_actual, logg_actual, feh_actual, **black_body_synth_settings)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Wrap up
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Determine minimum chi^2. Note that this assumes we'll never let logg 
    # float, as we're saying there're always 2 fitted params (Tef, [Fe/H])

    # Combine (and scale photometry here)
    grid_resid_all = np.hstack((
        grid_resid_b, grid_resid_r, grid_resid_p*phot_scale_fac))
    grid_rchi2_all = np.sum(grid_resid_all**2, axis=1)# / (len(grid_resid_all) - 2)

    chi2_map_dict = {
        # Grid coordinates, residuals, and rchi^2s
        "grid_teffs":grid_teffs,
        "grid_fehs":grid_fehs,
        "grid_resid_b":grid_resid_b,
        "grid_rchi2_b":grid_rchi2_b,
        "grid_resid_r":grid_resid_r,
        "grid_rchi2_r":grid_rchi2_r,
        "grid_resid_p":grid_resid_p,
        "grid_rchi2_p":grid_rchi2_p,
        "grid_resid_all":grid_resid_all,
        "grid_rchi2_all":grid_rchi2_all,
        # Valley Teff, [Fe/H], residuals, rchi^2s, and scaling factors
        "valley_fehs":valley_fehs,
        # Blue valley
        "valley_teffs_b":valley_teffs_b,
        "valley_resid_b":valley_resid_b,
        "valley_rchi2_b":valley_rchi2_b,
        "resid_scale_fac_b":resid_scale_fac_b,
        # Red valley
        "valley_teffs_r":valley_teffs_r,
        "valley_resid_r":valley_resid_r,
        "valley_rchi2_r":valley_rchi2_r,
        "resid_scale_fac_r":resid_scale_fac_r,
        # Colour valley
        "valley_teffs_p":valley_teffs_p,
        "valley_resid_p":valley_resid_p,
        "valley_rchi2_p":valley_rchi2_p,
        "resid_scale_fac_p":resid_scale_fac_p,
        # Combined valley
        "valley_teffs_all":valley_teffs_all,
        "valley_resid_all":valley_resid_all,
        "valley_rchi2_all":valley_rchi2_all,
        # Spectra
        "wave_black_body":wave_black_body,
        "spec_bb_low_feh":spec_bb_low_feh,
        "spec_bb_high_feh":spec_bb_high_feh,
        "spec_bb_lit":spec_bb_lit,
    }

    # All done, return
    return chi2_map_dict#, synth_spectra_r


def make_chi2_valley_interpolator(teffs, fehs, rchi2s, feh_slice_step):
    """Makes an interpolation function for the rchi^2 valley in terms of teff
    and [Fe/H].

    Parameters
    ----------
    teffs: float array
        List of teffs corresponding to fehs and rchi2s.

    fehs: float array
        List of fehs corresponding to teffs and rchi2s.

    rchi2s: float array
        List of rchi^2s corresponding to teffs and fehs.

    feh_slice_step: float
        Step size in [Fe/H] for mapping the valley floor.

    Returns
    -------
    calc_valley_teff: scipy.interpolate.interp1d
        1D interpolator taking [Fe/H] and producing the corresponding Teff 
        point along the rchi^2 minima valley.
    """
    # Ensure in numpy.array form
    teffs = np.array(teffs)
    fehs = np.array(fehs)
    rchi2s = np.array(rchi2s)

    # Initialise arrays to hold points in valley
    valley_fehs = []
    valley_teffs = []
    valley_rchi2s = []

    # In steps of feh_slice_step working along the [Fe/H] axis, determine the 
    # minimum rchi2 and save it and the corresponding Teff and [Fe/H]
    for feh_step in np.arange(fehs.min(), fehs.max(), feh_slice_step):
        slice_mask = np.logical_and(
            fehs > feh_step, 
            fehs < feh_step+feh_slice_step)

        rslice = rchi2s[slice_mask]

        # Skip if no points in this slice
        if len(rslice) == 0:
            continue
        else:
            min_i = int(np.argmin(rslice))

        valley_fehs.append(fehs[slice_mask][min_i])
        valley_teffs.append(teffs[slice_mask][min_i])
        valley_rchi2s.append(rchi2s[slice_mask][min_i])

    # Interpolator to calculate the temperature in the valley given [Fe/H]
    calc_valley_teff = interp1d(valley_fehs, valley_teffs, bounds_error=False)

    return calc_valley_teff

# -----------------------------------------------------------------------------
# Synthetic Photometry
# -----------------------------------------------------------------------------
def load_filter_profile(
    filt, 
    min_wl=1200, 
    max_wl=30000, 
    gaia_filt_path="data/GaiaDR2_Passbands.dat",
    tmass_filt_path="data/2mass_{}_profile.txt",
    wise_filt_path="data/wise_{}_profile.txt",
    skymapper_filt_path="data/skymapper_profiles.txt",
    do_zero_pad=True,):
    """Load in the specified filter profile and zero pad both ends in 
    preparation for feeding into an interpolation function.
    """
    #filt = filt.upper()

    filters_gaia = np.array(["G", "BP", "RP"])
    filters_skymapper = np.array(["u", "v", "g", "r", "i", "z"])
    filters_2mass = np.array(["J", "H", "K"])
    filters_wise = np.array(["W1", "W2", "W3", "W4"])

    all_filt = np.concatenate(
        (filters_gaia, filters_skymapper, filters_2mass, filters_wise))

    if filt not in all_filt:
        raise ValueError("Filter must be either {}".format(all_filt))

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
    
    elif filt in filters_skymapper:
        smf = pd.read_csv("data/skymapper_profiles.txt", delim_whitespace=True)

        wl = smf["wave_{}".format(filt)].values
        wl = wl[~np.isnan(wl)]

        filt_profile = smf[filt].values
        filt_profile = filt_profile[~np.isnan(filt_profile)]

    # WISE filter profiles
    elif filt in filters_wise:
        # Load the filter profile, and convert to Angstroms
        wpb = pd.read_csv(wise_filt_path.format(filt), delim_whitespace=True)

        wl = wpb["wl"].values * 10**4
        filt_profile = wpb["pb"].values

    # Pre- and post- append zeros from min_wl to max_wl
    if do_zero_pad:
        prepad_wl = np.arange(min_wl, wl[0], 1000)
        prepad = np.zeros_like(prepad_wl)
        
        postpad_wl = np.arange(wl[-1], max_wl, 1000)[1:]
        postpad = np.zeros_like(postpad_wl)
        
        wl_filt = np.concatenate((prepad_wl, wl, postpad_wl))
        filt_profile = np.concatenate((prepad, filt_profile, postpad))
        
        return wl_filt, filt_profile
    
    else:
        return wl, filt_profile


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

    Or this to compute SkyMapper vgriz:
        20 71  =  photometric system and filter (select from menu below)
        20 72  =  photometric system and filter (select from menu below)
        20 73  =  photometric system and filter (select from menu below)
        20 74  =  photometric system and filter (select from menu below)
        20 75  =  photometric system and filter (select from menu below)

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


def merge_casagrande_colour_grid(grid_fn_1, grid_fn_2,):
    """
    """
    grid_1 = load_casagrade_colour_grid(grid_fn_1)
    grid_2 = load_casagrade_colour_grid(grid_fn_2)

    # Drop duplicate columns
    grid_2.drop(columns=["id","logg","feh","teff","ebv"], inplace=True)

    grid_final = grid_1.join(grid_2, how="inner")

    grid_final.to_csv("data/synth_colour_grid.csv")


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


class SyntheticBCInterpolator():
    """Class to interpolate synthetic colours, meaning that you don't need to
    reload and set up the interpolator every time as you would with a function.
    """
    def __init__(self):
        """Constructor
        """
        param_cols = ["teff", "logg", "feh"]

        self.bc_grid = load_casagrade_colour_grid()
        self.VALID_COLOURS = ["Bp", "Rp", "J", "H", "K", "v", "g", "r", "i", "z"]

        # Gaia photometry
        self.calc_bc_bp =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_Bp"].values)

        self.calc_bc_rp =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_Rp"].values)

        # 2MASS photometry
        self.calc_bc_j =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_J"].values)

        self.calc_bc_h =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_H"].values)
        
        self.calc_bc_k =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_K"].values)

        # SkyMapper photometry
        self.calc_bc_v =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_v"].values)

        self.calc_bc_g =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_g"].values)

        self.calc_bc_r =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_r"].values)

        self.calc_bc_i =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_i"].values)
        
        self.calc_bc_z =  LinearNDInterpolator(
            self.bc_grid[param_cols].values, 
            self.bc_grid["BC_z"].values)


    def compute_bc(self, params, phot_band):
        """Computes a synthetic colour based on params and provided colour band

        Parameters
        ------
        params: float array
            Stellar parameters of form (teff, logg, [Fe/H]).

        phot_band: string
            The photometric filter, either 'Bp', 'Rp', 'J', 'H', or 'K'.

        Returns
        -------
        colour: float array
            Resulting synthetic stellar colour.
        """
        bc_func_dict = {
            "Bp":self.calc_bc_bp,
            "Rp":self.calc_bc_rp,
            "J":self.calc_bc_j,
            "H":self.calc_bc_h,
            "K":self.calc_bc_k,
            "v":self.calc_bc_v,
            "g":self.calc_bc_g,
            "r":self.calc_bc_r,
            "i":self.calc_bc_i,
            "z":self.calc_bc_z,
        }

        if phot_band in bc_func_dict.keys():
            bc_synth = bc_func_dict[phot_band](params[0], params[1], params[2])

        else:
            raise ValueError("Invalid filter, must be in {}".format(
                self.VALID_COLOURS))

        return bc_synth

# -----------------------------------------------------------------------------
# BT-Settl Spectra
# ----------------------------------------------------------------------------- 
def import_btsettl_spectra(
    bbtsettl_path="phoenix",
    btsettl_grid_point=(3200,5.0),
    blue_conv_res=0.77,
    red_conv_res=0.44,
    len_wl_b=2858,
    len_wl_r=3637,):
    """Import BT-Settl spectra (given Teff and logg), convolve, and regrid to
    WiFeS B3000 and R7000.
    """
    # Construct filename
    bts_fn = "lte0{:0.0f}.0-{:0.1f}-0.0a+0.0.BT-Settl.spec.7".format(
        btsettl_grid_point[0]/100,  # teff
        btsettl_grid_point[1],)     # logg
    bts_file = os.path.join(bbtsettl_path, bts_fn)

    print("Importing:", bts_file)

    try:
        # Import the BBT-Settl spectrum, sort, and put flux in proper units
        spec_bts = pd.read_csv(bts_file, delim_whitespace=True,
            names=["wl", "flux", "bb_flux"]+list(np.arange(0,23)))
        
        sort_bts = np.argsort(spec_bts["wl"].values)
        wl = spec_bts["wl"][sort_bts].values
        spec_bts = 10**(spec_bts["flux"][sort_bts].values-8)

    except:
        raise ValueError("Invalid BT-Settl grid point")

    wl_b_mask = np.logical_and(wl > 3500, wl < 5700)
    wl_r_mask = np.logical_and(wl > 5400, wl < 7000)

    wl_b_all = wl[wl_b_mask]
    spec_b_all = spec_bts[wl_b_mask]

    wl_r_all = wl[wl_r_mask]
    spec_r_all = spec_bts[wl_r_mask]

    # Convolve
    gc_b =  conv.Gaussian1DKernel(blue_conv_res, x_size=len(wl_b_all))
    gc_r =  conv.Gaussian1DKernel(red_conv_res, x_size=len(wl_r_all))

    spec_b_conv = conv.convolve_fft(spec_b_all, gc_b)
    spec_r_conv = conv.convolve_fft(spec_r_all, gc_r)

    # Regrid the wavelengths, dropping any values that don't evenly fold in
    n_b = int(np.round(len(wl_b_all) / len_wl_b))
    cutoff_b = len(wl_b_all) % n_b

    if cutoff_b == 0:
        wl_b_conv_rg = wl_b_all.reshape(-1, n_b).mean(axis=1)
        spec_b_conv_rg = spec_b_conv.reshape(-1, n_b).sum(axis=1)
    else:
        wl_b_conv_rg = wl_b_all[:-cutoff_b].reshape(-1, n_b).mean(axis=1)
        spec_b_conv_rg = spec_b_conv[:-cutoff_b].reshape(-1, n_b).sum(axis=1)

    n_r = int(np.round(len(wl_r_all) / len_wl_r))
    cutoff_r = len(wl_r_all) % n_r

    if cutoff_r == 0:
        wl_r_conv_rg = wl_r_all.reshape(-1, n_r).mean(axis=1)
        spec_r_conv_rg = spec_r_conv.reshape(-1, n_r).sum(axis=1)
    else:
        wl_r_conv_rg = wl_r_all[:-cutoff_r].reshape(-1, n_r).mean(axis=1)
        spec_r_conv_rg = spec_r_conv[:-cutoff_r].reshape(-1, n_r).sum(axis=1)
    
    return wl_b_conv_rg, spec_b_conv_rg, wl_r_conv_rg, spec_r_conv_rg