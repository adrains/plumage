"""Functions dealing with the computation of stellar parameters or photometry.
"""
import os
import numpy as np
import pandas as pd
from warnings import warn
import astropy.units as units
import astropy.constants as const
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
from scipy.integrate import simps
from collections import OrderedDict
from scipy.optimize import least_squares
from numpy.polynomial.polynomial import polyval as polyval
from scipy.interpolate import LinearNDInterpolator
#from dustmaps.leike_ensslin_2019 import LeikeEnsslin2019Query
#from dustmaps.leike2020 import Leike2020Query

LOCAL_BUBBLE_DIST_PC = 70

# -----------------------------------------------------------------------------
# Limb darkening
# -----------------------------------------------------------------------------
def load_claret17_ldc_grid(path="data/claret_4_term_ld.tsv"):
    """Paremeters
    ----------
    grid: pandas dataframe, default: None
        Pandas dataframe of grid (if already imported)

    Returns
    -------
    path: string, default: "data/claret_4_term_ld.tsv"
        Where to import the grid from if not provided
    """
    grid = pd.read_csv("data/claret_4_term_ld.tsv", sep="\t", comment="#")

    return grid


def get_claret17_limb_darkening_coeff(teff, logg, feh, grid=None, 
                                      path="data/claret_4_term_ld.tsv"):
    """Function to interpolate the four-term nonlinear limb darkening 
    coefficients given values of Teff, logg, & [Fe/H], which are specific to
    the TESS filter.
    
    Note: presently using PHOENIX model limb darkening grid which have only 
    been sampled at solar [Fe/H], so any metallicity value passed in is not
    used.

    TODO: consider non-linear interpolator

    The interpolated grid is from Claret 2017:
     - https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract
    
    Parameters
    ----------
    teff: float or float array
        Stellar effective temperature in Kelvin

    logg: float or float array
        Stellar surface gravity
        
    feh: float or float array
        Stellar metallicity, [Fe/H] (relative to Solar)
        
    grid: pandas dataframe, default: None
        Pandas dataframe of grid (if already imported)
        
    path: string, default: "data/claret_4_term_ld.tsv"
        Where to import the grid from if not provided
        
    Returns
    -------
    ldc: float or floar array
        Four term limb darkening coefficients
    """
    # Load in grid if not provided
    if grid is None:
        grid = load_claret17_ldc_grid(path)
    
    # Interpolate along logg and Teff for all entries for filter
    calc_ldc = LinearNDInterpolator(
        grid[["Teff", "logg"]],#, "Z"]], 
        grid[["a1LSM", "a2LSM", "a3LSM", "a4LSM"]])
    
    # Calculate and return
    ldc = calc_ldc(np.array(teff), np.array(logg))#, np.array(feh))
    
    return ldc

# -----------------------------------------------------------------------------
# Fundamental Parameter Empirical Relations
# -----------------------------------------------------------------------------
def compute_mann_2019_masses(k_mag_abs):
    """Calculates stellar masses based on absolute 2MASS Ks band magnitudes
    per the empirical relations in Table 6 of Mann et al. 2019. 

    The implemented relation is the 5th order polynomial fit without the 
    dependence on [Fe/H].

    Note that while this runs quickly, it is suggested to use the module/func
    provided by Mann+2019 as  mk_mass.posterior() which properly considers
    magnitude and distance uncertainties.

    Parameters
    ----------
    k_mag_abs: float array
        Array of absolute 2MASS Ks band magnitudes

    Returns
    -------
    masses: float array
        Resulting stellar masses in solar units.

    e_masses: float array
        Uncertainties on stellar masses in solar units.
    """
    # Zero point for the relation
    zp = 7.5

    # Fractional mass uncertainty
    frac_mass_sigma = 0.02

    # Coefficients for 5th order polynomial fit without [Fe/H] dependence
    coeff = np.array(
        [-0.642, -0.208, -8.43*10**-4, 7.87*10**-3, 1.42*10**-4, -2.13*10**-4])

    # Calculate masses + uncertainties
    masses = 10**polyval(k_mag_abs-zp, coeff)
    e_masses = masses * frac_mass_sigma

    return masses, e_masses
    

def compute_mann_2015_teff(
    colour,
    j_h=None,
    feh=None,
    relation="BP - RP, J - H",
    teff_file="data/mann_2015_teff.txt", 
    sigma_spec=60,
    enforce_bounds=True,
    teff_bounds=(2700, 4100),):
    """
    Calculates stellar effective temperatures based on the empircal relations
    in Table 2 of Mann et al. 2015.

    Paper:
        https://iopscience.iop.org/article/10.1088/0004-637X/804/1/64
    
    Erratum:
        https://iopscience.iop.org/article/10.3847/0004-637X/819/1/87

    Supported relations:
        BP - RP
        V - J
        V - Ic
        r - z
        r - J
        BP - RP, [Fe/H]
        BP - RP (DR3), [Fe/H]
        V - J, [Fe/H]
        V - Ic, [Fe/H]
        r - z, [Fe/H]
        r - J, [Fe/H]
        BP - RP, J - H
        V - J, J - H
        V - Ic, J - H
        r - z, J - H
        r - J, J - H

    TODO: enforce bounds on colour, rather than Teff.

    Parameters
    ----------
    colour: float array
        Photometric colour used for the relation, e.g. BP-RP.

    j_h: float array
        J-H colour used as a proxy for [Fe/H] in some relations. Defaults to 
        None.

    feh: float array
        Metallicities of the sample for use in some relations. Defaults to 
        None.

    relation: string
        Photometric relation to use.

    teff_file: string
        Location for the stored table.

    sigma_spec: string
        Spectroscopic uncertainty quoted in the paper added in quadrature with
        relation uncertainties, defaults to 60 K.
    
    Returns
    -------
    teffs: float array
        Array of calculated stellar effective temperatues

    e_teffs: float array
        Uncertainties on teff.
    """
    # Ensure everything is numpy arrays
    colour = np.atleast_1d(colour)
    if j_h is not None: j_h = np.atleast_1d(j_h)
    if feh is not None: feh = np.atleast_1d(feh)

    # Import the table of colour relations
    m15_teff = pd.read_csv(
        teff_file,
        delimiter="\t",
        comment="#", 
        index_col="X")

    # Check we've been given a valid relation
    if relation not in m15_teff.index.values:
        raise ValueError("Unsupported relation. Must be one of %s"
                         % m15_teff.index.values)

    # Now ensure we've been given the right combination of inputs
    if "J - H" in relation and j_h is None:
        raise ValueError("Must give value for J-H to use J-H relations")
    if "[Fe/H]" in relation and feh is None:
        raise ValueError("Must give value of [Fe/H] to use [Fe/H] relations")

    # Calculate non-metallicity component
    x_coeff = m15_teff.loc[relation][["a", "b", "c", "d", "e"]].values
    color_comp = polyval(colour, x_coeff)

    # Now calculate the metallicity component, which either uses [Fe/H] 
    # directly, or J-H as a proxy. These are mutually exclusive

    # syst_info.loc["r_planet_rearth", "value"])

    # J-H component
    if "J - H" in relation:
        jh_comp = (m15_teff.loc[relation, "f"] * j_h 
                   + m15_teff.loc[relation, "g"] * j_h**2)
        feh_comp = 0

    # [Fe/H] component
    elif "[Fe/H]" in relation:
        feh_comp = m15_teff.loc[relation, "f"] * feh
        jh_comp = 0

    # Using a single colour
    else:
        feh_comp = 0
        jh_comp = 0

    # Add components together, and scale by temperature pivot/zero point
    teffs = (color_comp + jh_comp + feh_comp) * 3500

    # Calculate errors by taking the uncertainty on the relation, added in
    # quadrature with the spectroscopic uncertainties on derived Teffs
    e_teff = np.sqrt(m15_teff.loc[relation]["sigma"]**2 + sigma_spec**2)
    e_teffs = np.ones_like(teffs) * e_teff

    if enforce_bounds:
        oob = np.logical_or(teffs < teff_bounds[0], teffs > teff_bounds[1])
        teffs[oob] = np.nan
        e_teffs[oob] = np.nan

    return teffs, e_teffs


def compute_mann_2015_radii(
    k_mag_abs,
    fehs=None,
    enforce_M_Ks_bounds=True,
    enforce_feh_bounds=True,):
    """Calculates stellar radii based on absolute 2MASS Ks band magnitudes
    per the empirical relations in Table 1 of Mann et al. 2015.

    Paper:
        https://iopscience.iop.org/article/10.1088/0004-637X/804/1/64
    
    Erratum:
        https://iopscience.iop.org/article/10.3847/0004-637X/819/1/87

    Two relations are implemented:
        a) 2nd order polynomial for R* in M_Ks (Eqn 4)
        b) 2nd order polynomial for R* in M_Ks with [Fe/H] term (Eqn 5)

    Eqn 4: R* = a + b*(M_Ks) + c*(M_Ks)^2
    Eqn 5: R* = [a + b*(M_Ks) + c*(M_Ks)^2] * (1 + f*[Fe/H])
        
    Note that the values of the coefficients were taken from the *erratum*
    Table 1. The relations are valid for:
        i)   4.6 < M_Ks < 9.8
        ii)  -0.6 < [Fe/H] < 0.5
        iii) 0.1 < R* < 0.7 R_sun

    With the paper noting that "However, because of sample selection biases,
    the range of metallicities is significantly smaller for stars of spectral 
    types later than M4 (mostly slightly metal-poor) and for the late-K dwarfs
    (mostly metal-rich). There are also only 15 stars in total with spectral
    types M5–M7 and only three stars with [Fe/H] < -0.5. These sample biases
    should be considered when applying these formulae."

    Parameters
    ----------
    k_mag_abs: float array
        Array of absolute 2MASS Ks band magnitudes

    fehs: float array, default: None
        Array of [Fe/H] corresponding to k_mag_abs. If provided, we instead
        use the M_Ks-[Fe/H] relation.

    enforce_M_Ks_bounds, enforce_feh_bounds: boolean, default: True
        If True, we enforce the M_Ks or [Fe/H] (if using) bounds of the 
        relations and set stars outside the limits to nan.

    Returns
    -------
    radii: float array
        Resulting stellar radii in solar units.

    e_masses: float array
        Uncertainties on stellar radii in solar units.
    """
    # Values for Eqn 4 in Mann+2015
    frac_radii_pc_M_Ks = 0.0289
    coeff_M_Ks = np.array([1.9515, -0.3520, 0.01680])

    # Values for Eqn 5 in Mann+2015
    frac_radii_pc_M_Ks_feh = 0.027
    coeff_M_Ks_feh = np.array([1.9305, -0.3466, 0.01647])
    coeff_feh = 0.04458     # Note: erratum value!!

    # 3rd order polynomial fit *without* [Fe/H] dependence
    if fehs is None:
        radii = polyval(k_mag_abs, coeff_M_Ks)
        e_radii = radii * frac_radii_pc_M_Ks

    # 3rd order polynomial fit *with* [Fe/H] dependence
    else:
        radii = polyval(k_mag_abs, coeff_M_Ks_feh) * (1 + coeff_feh * fehs)
        e_radii = radii * frac_radii_pc_M_Ks_feh

    # Default behaviour is to output NaN where M_Ks or [Fe/H] (if using) are
    # beyond the bounds of the original relation.
    if enforce_M_Ks_bounds:
        outside_bounds = np.logical_or(k_mag_abs < 4.6, k_mag_abs > 9.8)
        radii[outside_bounds] = np.nan
        e_radii[outside_bounds] = np.nan

    if fehs is not None and enforce_feh_bounds:
        outside_bounds = np.logical_or(fehs < -0.6, fehs > 0.5)
        radii[outside_bounds] = np.nan
        e_radii[outside_bounds] = np.nan

    return radii, e_radii


def compute_kesseli_2019_radii(
    k_mag_abs,
    fehs,
    enforce_M_Ks_bounds=True,
    enforce_feh_bounds=True,):
    """Calculates stellar radii based on the 2MASS M_Ks-[Fe/H] empirical 
    relation in Equation 7 of Kesseli et al. 2019.

    Paper:
        https://ui.adsabs.harvard.edu/abs/2019AJ....157...63K/abstract
    
    Equation 7 is as follows (noting that the 'b' term is negated):

        R* = [a - b*(M_Ks) + c*(M_Ks)^2)] * (1 + f*[Fe/H])
        
    Where:
        a = 1.875 ± 0.05
        b = −0.337 ± 0.01
        c = 0.0161 ± 0.0009
        f = 0.079 ± 0.01

    And the scatter on the residuals is 6%, which we adopt as the uncertainty.
        
    The relation is valid for:
        i)   4 < M_Ks < 11
        ii)  -2.0 < [Fe/H] < 0.5

    Parameters
    ----------
    k_mag_abs: float array
        Array of absolute 2MASS Ks band magnitudes

    fehs: float array
        Array of [Fe/H] corresponding to k_mag_abs. 

    enforce_M_Ks_bounds, enforce_feh_bounds: boolean, default: True
        If True, we enforce the M_Ks or [Fe/H] bounds of the relations and set
        stars outside the limits to nan.

    Returns
    -------
    radii: float array
        Resulting stellar radii in solar units.

    e_masses: float array
        Uncertainties on stellar radii in solar units.
    """
    # Values for Eqn 7 in Kesseli+2019
    frac_radii_pc = 0.06
    coeff = np.array([1.875, -0.337, 0.0161])
    coeff_feh = 0.079

    # Polynomial fit with [Fe/H] dependence
    radii = ((coeff[0] + coeff[1]*k_mag_abs + coeff[2]*k_mag_abs**2) 
             * (1 + coeff_feh * fehs))
    e_radii = radii * frac_radii_pc

    if enforce_M_Ks_bounds:
        outside_bounds = np.logical_or(k_mag_abs < 4, k_mag_abs > 11)
        radii[outside_bounds] = np.nan
        e_radii[outside_bounds] = np.nan

    if enforce_feh_bounds:
        outside_bounds = np.logical_or(fehs < -2.0, fehs > 0.5)
        radii[outside_bounds] = np.nan
        e_radii[outside_bounds] = np.nan

    return radii, e_radii


def compute_logg(masses, e_masses, radii, e_radii,):
    """Compute logg from mass and radius + propagate uncertainties.

    Parameters
    ----------
    masses, e_masses: float array
        Masses and mass uncertainties in solar units.

    raddi, e_radii: float array
        Radii and radii uncertanties in solar units.

    Returns
    -------
    logg, e_logg: float array
        logg and logg uncertainties in log(cgs) units
    """
    # Define constants. Note: must be in cgs units for logg.
    G = const.G.cgs.value
    M_sun = const.M_sun.cgs.value
    R_sun = const.R_sun.cgs.value  

    # Ensure we have arrays of floats in cgs units
    masses = np.atleast_1d(masses).astype(float) * M_sun
    e_masses = np.atleast_1d(e_masses).astype(float) * M_sun
    radii = np.atleast_1d(radii).astype(float) * R_sun
    e_radii = np.atleast_1d(e_radii).astype(float) * R_sun

    # Calculate the surface gravity and uncertainty before logging
    sg = ((G * masses) / radii**2)

    e_sg = np.sqrt(
        e_masses**2 * (G/radii**2)**2
        + e_radii**2 * ((-2 * G * masses) / radii**3)**2
        )

    # Calculate logg and e_logg    
    logg = np.log10(sg)
    e_logg = np.abs(e_sg / (sg*np.log(10)))

    return logg, e_logg


def compute_casagrande_2021_teff(
    colour,
    logg,
    feh,
    relation,
    teff_file="data/casagrande_colour_teff_2021_EDR3.dat",
    enforce_bounds=True,
    regime="dwarf",):
    """Calculates Teff using a given colour relation and measure of stellar 
    logg and [Fe/H] per the Casagrande+2021 relations.

    https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.2684C/abstract

    Parameters
    ----------
    colour: float or float array
        The stellar colour corresponding to 'relation'.
    
    logg: float or float array
        Stellar surface gravity in log cgs units.

    feh: float or float array
        Stellar metallciity relative to solar, i.e. [Fe/H].

    relation: string
        Colour relation to use. Valid options are:
            - (BP-RP)
            - (BP-J)
            - (BP-H)
            - (BP-Ks)
            - (RP-J)
            - (RP-H)
            - (RP-Ks)
            - (G-J)
            - (G-H)
            - (G-Ks)
            - (G-BP)
            - (G-RP)
        Where all colours have been corrected for reddening.
    
    teff_file: string
        File of coefficients to load.

    enforce_bounds: boolean, default: True
        If True, temperatures outside of teff_bounds are set to nan.

    regime: str, default: 'dwarf'
        Stellar evolution regime for implementing colour limits, either 'dwarf'
        or 'giant'. See Figure 4 of Casagrande+21.

    Returns
    -------
    teff, e_teff: float or float array
        Calculated stellar effective temperature and corresponding uncertainty.
    """
    if regime not in ("dwarf", "giant"):
        raise ValueError("Invalid value for 'regime'.")

    # Convert to arrays
    colour = np.atleast_1d(colour)
    loggs = np.atleast_1d(logg)
    fehs = np.atleast_1d(feh)
    
    n_star = len(colour)

    # Load in Casagrande+21 colour relations, tab separated
    c21_teff = pd.read_csv(
        teff_file,
        sep="\t",
        comment="#",
        index_col="colour",) 

    # Ensure relation is valid
    if relation not in c21_teff.index.values:
        raise ValueError("Unsupported relation. Must be one of %s"
                         % c21_teff.index.values)

    # Get the coefficients
    coeff_cols = ["a{}".format(i) for i in range(15)]
    coeff = c21_teff.loc[relation][coeff_cols].values

    # Calculate each term of the polynomial
    poly_terms = [
        np.full(n_star, coeff[0]),
        coeff[1]*colour,
        coeff[2]*colour**2,
        coeff[3]*colour**3,
        coeff[4]*colour**5,
        coeff[5]*loggs,
        coeff[6]*loggs*colour,
        coeff[7]*loggs*colour**2,
        coeff[8]*loggs*colour**3,
        coeff[9]*loggs*colour**5,
        coeff[10]*fehs,
        coeff[11]*fehs*colour,
        coeff[12]*fehs*colour**2,
        coeff[13]*fehs*colour**3,
        coeff[14]*fehs*loggs*colour,
    ]

    # And sum to get the final temperature
    teff = np.sum(poly_terms, axis=0)
    e_teff = np.full(n_star, c21_teff.loc[relation]["sigma_teff"])

    if enforce_bounds:
        c_min = c21_teff.loc[relation]["c_min_{}".format(regime)]
        c_max = c21_teff.loc[relation]["c_max_{}".format(regime)]

        oob = np.logical_or(colour < c_min, colour > c_max)
        teff[oob] = np.nan
        e_teff[oob] = np.nan

    if n_star == 1:
        teff = teff[0]
        e_teff = e_teff[0]

    return teff, e_teff


# -----------------------------------------------------------------------------
# Orbital parameters
# -----------------------------------------------------------------------------
def compute_semi_major_axis(
    mass, 
    e_mass, 
    period, 
    e_period, 
    radius=None, 
    e_radius=None):
    """Compute the semi-major axis of the planet given the mass of the star and
    orbital period. Optionally, also can scale this by the stellar radii (if
    provided).

    Parameters
    ----------
    mass, e_mass: float or float array
        Stellar mass and uncertainty in solar units.
    
    period, e_period: float or float array
        Planet orbital period and uncertainty in days.

    radius, e_radius: float, float array, or None, default: None
        Stellar radius and uncertainty in solar units.

    Returns
    -------
    sma, e_sma: float or float array
        Semi-major axis and uncertainty in metres.

    sma_rstar, e_sma_rstar: float or float array (optional)
        Semi-major axis and uncertainty scaled by stellar radius to be 
        dimensionless. Returned only if radius is provided.
    """
    G = const.G.si.value
    period = period * 60 * 60 * 24              # Convert days to seconds
    e_period = e_period * 60 * 60 * 24            # Convert days to seconds
    mass = mass * const.M_sun.si.value          # Convert M_sun units to kg
    e_mass = e_mass * const.M_sun.si.value      # Convert M_sun units to kg

    sma = ((G*mass*period**2) / (4*np.pi**2))**(1/3)

    # Calculate uncertainty
    X = (G / (4*np.pi**2))**(1/3)   # const

    e_sma = X * np.sqrt(((1/3) * mass**(-2/3) * period**(2/3) * e_mass)**2 
                        + ((2/3) * mass**(1/3) * period**(-1/3) * e_period)**2)

    # Scaled SMA by radius if provided
    if radius is not None and e_radius is not None:
        sma_rstar = sma / (radius * const.R_sun.si.value) # convert to m
        e_sma_rstar = sma_rstar * ((e_sma/sma)**2 + (e_radius/radius)**2)**0.5

        return sma, e_sma, sma_rstar, e_sma_rstar
    
    # Otherwise just return
    else:
        return sma, e_sma

# -----------------------------------------------------------------------------
# Fluxes, sampling parameters, and final parameters
# -----------------------------------------------------------------------------
def sample_params(
    teff, 
    e_teff, 
    logg, 
    e_logg, 
    feh,
    e_feh,
    dist,
    e_dist,
    mag_dict, 
    n_samples=1000):
    """Function to sample stellar parameters N times to enable Monte Carlo 
    uncertainties on final values of flux, luminosity, and stellar radius.

    Parameters
    ----------
    teff: float
        Stellar effective temperature in K.
    
    e_teff: float
        Uncertainty on stellar effective temperature in K.

    logg: float
        log of stellar surface gravity in cgs units.

    e_logg: float
        Uncertainty of log of stellar surface gravity in cgs units.

    feh: float
        Stellar metallicity relative to solar.

    e_feh: float
        Uncertainty on stellar metallicity relative to solar.
    
    dist: float
        Stellar distance in pc.

    e_dist: float
        Uncertainty on stellar distance in pc.

    mag_dict: dict
        Dictionary of form {band:(mag, e_mag),}. e.g. {"Rp_mag":(9.5,0.1),}

    n_samples: int, default: 1000
        Number of times to sample the parameters.

    Returns
    -------
    sampled_params: pandas.DataFrame
        Pandas dataframe with a row for every sampling iteration.
    """
    # First sample stellar params, as every star will have these
    data = {
        "teff":np.random.normal(teff, e_teff, n_samples),
        "logg":np.random.normal(logg, e_logg, n_samples),
        "feh":np.random.normal(feh, e_feh, n_samples),
        "dist":np.random.normal(dist, e_dist, n_samples)
    }

    # Initialise pandas dataframe
    sampled_params = pd.DataFrame(data)

    # Now go through and sample whatever magnitudes we have been given
    for mag in mag_dict.keys():
        sampled_params[mag] = np.random.normal(
            mag_dict[mag][0], 
            mag_dict[mag][1], 
            n_samples)

    return sampled_params


def sample_all_params(
    observations, 
    info_cat, 
    bc_path,
    filters=["Bp", "Rp", "J", "H", "K"],
    logg_col="logg_synth",
    filter_mask=[True,True,True,True,True]):
    """Sample parameters for all stars in observations that have a match in
    info_cat.

    Parameters
    ----------
    observations: pandas.DataFrame
        Pandas dataframe containing observational info and results of synthetic
        fitting for each star.
    
    info_cat: pandas.DataFrame
        Pandas dataframe containing literature photometry for targets in
        observations.

    bc_path: str
        Path to Luca Casagrande's bolometric-corrections software
    
    filters: list of str, default: ["Bp", "Rp", "J", "H", "K"]
        List of filter bands to sample photometry for. Note that this should
        correspond to the settings in bolometric-corrections.
    
    logg_col: str, default: 'logg_synth'
        Logg column to use, used to select between logg from empirical 
        relations or iterated logg.

    Returns
    -------
    all_sampled_params: collections.OrderedDict
        Ordered dictionary of pairing key of stellar ID to pandas dataframe of
        sampled parameters from sample_params.
    """
    # Initialise ordered dict to hold results
    all_sampled_params = OrderedDict()

    # Temporary join
    obs = observations.join(info_cat, "source_id", rsuffix="_info")  

    # Sample parameters for every star in observations
    for star_i, star_data in obs.iterrows():
        # Get source id
        source_id = star_data["source_id"]

        print("-"*40,"\n", star_i, "\n", "-"*40)
        
        # Construct mag dict
        mag_dict = OrderedDict()

        for filt in filters:
            mag_dict[filt] = (
                star_data["{}_mag".format(filt)],
                star_data["e_{}_mag".format(filt)])

        sampled_params = sample_params(
            teff=star_data["teff_synth"], 
            e_teff=star_data["e_teff_synth"], 
            logg=star_data[logg_col], 
            e_logg=star_data["e_{}".format(logg_col)], 
            feh=star_data["feh_synth"],
            e_feh=star_data["e_feh_synth"],
            dist=star_data["dist"],
            e_dist=star_data["e_dist"],
            mag_dict=mag_dict)
        
        # Do Casagrande sampling
        sample_casagrande_bc(
            sampled_params=sampled_params, 
            bc_path=bc_path, 
            star_id=source_id)

        # Force exception if we have duplicate targets, otherwise save
        if source_id in all_sampled_params.keys():
            raise Exception("Duplicate stars in observations ({}) - remove or"
                            "make code more generic.".format(source_id))
        else:
            all_sampled_params[source_id] = sampled_params

        # Compute instantaneous params
        compute_instantaneous_params(sampled_params, filters, filter_mask)

    return all_sampled_params


def sample_casagrande_bc(
    sampled_params, 
    bc_path, 
    star_id,
    filters=["Bp", "Rp", "J", "H", "K"],
    teff_lims=[2600, 8000],
    logg_lims=[-0.5, 5.5],
    feh_lims=[-4.0, +1.0],
    ):
    """Sample stellar parameters for use with the bolometric correction code
    from Casagrande & VandenBerg (2014, 2018a, 2018b). Resulting bolometric
    corrections are saved to sampled_params.

    Note that any samples falling outside of the provided limits will be 
    dropped from the dataframe (to avoid crashing the BC code).
    
    https://github.com/casaluca/bolometric-corrections
    
    Check that selectbc.data looks like this to compute Bp, Rp, J, H, K:
        1  = ialf (= [alpha/Fe] variation: select from choices listed below)
        5  = nfil (= number of filter bandpasses to be considered; maximum = 5)
        27 86  =  photometric system and filter (select from menu below)
        27 88  =  photometric system and filter (select from menu below)
        1 1  =  photometric system and filter (select from menu below)
        1 2  =  photometric system and filter (select from menu below)
        1 3  =  photometric system and filter (select from menu below)

    Parameters
    ----------
    sampled_params: pandas.DataFrame
        Pandas dataframe with a row for every sampling iteration.

    bc_path: str
        Path to Luca Casagrande's bolometric-corrections software
    
    star_id: str
        ID of the star.

    filters: list of str, default: ["Bp", "Rp", "J", "H", "K"]
        List of filter bands to sample photometry for. Note that this should
        correspond to the settings in bolometric-corrections.

    teff_lims: 2 element float array, default: [2600, 8000]
        Lower and upper Teff limits on the MARCS grid used for computing BCs
    
    logg_lims: 2 element float array, default: [-0.5, 5.5]
        Lower and upper logg limits on the MARCS grid used for computing BCs
    
    feh_lims: 2 element float array, default: [-4.0, +1.0]
        Lower and upper [Fe/H] limits on the MARCS grid used for computing BCs
    """
    # MARCS models and thus Casagrande BC grid have limits on params. Enforce
    # these by removing any sampled params that are beyond these
    rows_to_drop = []

    for sample_i, sample in sampled_params.iterrows():
        if sample["teff"] < teff_lims[0] or sample["teff"] > teff_lims[1]:
            rows_to_drop.append(sample_i)

        elif sample["logg"] < logg_lims[0] or sample["logg"] > logg_lims[1]:
            rows_to_drop.append(sample_i)

        elif sample["feh"] < feh_lims[0] or sample["feh"] > feh_lims[1]:
            rows_to_drop.append(sample_i)

    sampled_params.drop(rows_to_drop, axis=0, inplace=True)

    # Initialise the new columns
    bc_labels = ["BC_{}".format(filt) for filt in filters]

    for bc in bc_labels:
        sampled_params[bc] = 0

    # Generate a unique 'id' for every iteration
    n_bs = len(sampled_params)
    id_fmt = star_id + "_%0" + str(int(np.log10(n_bs)) + 1) + "i"
    ids = [id_fmt % s for s in np.arange(0, n_bs)]

    # Initialise the desired E(B-V) - TODO: actually account for reddening
    ebvs = np.zeros(n_bs)

    # Arrange data in format required of bc code
    data = np.vstack((
        ids, 
        sampled_params["logg"].values, 
        sampled_params["feh"].values,
        sampled_params["teff"].values, 
        ebvs,
    )).T

    # Save to BC folder
    np.savetxt("%s/input.sample.all" % bc_path, data, delimiter=" ", fmt="%s")

    # Run
    os.system("cd %s; ./bcall" % bc_path)

    # Load in the result
    results = pd.read_csv("%s/output.file.all" % bc_path, 
                            delim_whitespace=True)

    # Save the bolometric corrections
    bc_num_cols = ["BC_1", "BC_2", "BC_3", "BC_4", "BC_5"]
    sampled_params[bc_labels] = results[bc_num_cols].values


def compute_instantaneous_params(
    sampled_params, 
    filters=["Bp", "Rp", "J", "H", "K"], 
    filter_mask=[True,True,True,True,True]):
    """Compute instantaneous stellar parameters (fbol in various filters, plus
    stellar radius and luminosity). Instantaneous parameters are saved to
    sampled_params.

    Parameters
    ----------
    sampled_params: pandas.DataFrame
        Pandas dataframe with a row for every sampling iteration.
    
    filters: list of str, default: ["Bp", "Rp", "J", "H", "K"]
        List of filter bands to sample photometry for. Note that this should
        correspond to the settings in bolometric-corrections.
    
    filter_mask: list of str, default: [1,1,1,1,1]
        Mask corresponding to filters, any filter set to 0 will not be 
        included in the instantaneous average for fbol.
    """
    # Ensure the mask is full of booleans (otherwise you get weird results)
    filter_mask = np.array(filter_mask).astype(bool)

    # Calculate
    f_bol_bands = ["f_bol_{}".format(filt) for filt in filters]

    # Calculate bolometric flux for each filter band
    for filt in filters:
        sampled_params["f_bol_{}".format(filt)] = calc_f_bol(
            sampled_params["BC_{}".format(filt)],
            sampled_params[filt])

    # Now calculate average, using mask to determine which filters are used
    fbol_bands_in_avg = np.array(f_bol_bands)[filter_mask]
    sampled_params["f_bol_avg"] = np.mean(
        sampled_params[fbol_bands_in_avg], axis=1)

    # Define params as variables to make maths more readable
    fbol_avg = sampled_params["f_bol_avg"].values
    sigma = const.sigma_sb.cgs.value
    teff = sampled_params["teff"].values
    dist = sampled_params["dist"].values * const.pc.cgs

    # Calculate stellar radius (in cgs units)
    sampled_params["radius"] = dist * np.sqrt(fbol_avg / (sigma*teff**4))

    # Calculate luminosity (in cgs units)
    sampled_params["lum"] = 4 * np.pi * fbol_avg * dist**2


def calc_f_bol(bc, mag):
    """Calculate the bolometric flux from a bolometric correction and mag.

    Parameters
    ----------
    bc: float or float array
        Bolometric correction corresponding to mag.

    mag: float or float array
        Magnitude in given filter band to compute fbol from.

    Returns
    -------
    f_bol: float or float array
        Resulting bolometric flux.
    """
    L_sun = const.L_sun.cgs.value # erg s^-1
    au = const.au.cgs.value       # cm
    M_bol_sun = 4.75
    
    exp = -0.4 * (bc - M_bol_sun + mag - 10)
    
    f_bol = (np.pi * L_sun / (1.296 * 10**9 * au)**2) * 10**exp
    
    return f_bol


def calc_f_bol_from_mbol(mbol, e_mbol):
    """Calculate the bolometric flux from a bolometric magnitude.
    Parameters
    ----------
    mbol: float or float array
        Bolometric magnitude of the star.

    Returns
    -------
    f_bol: float or float array
        Resulting bolometric flux.
    """
    L_sun = const.L_sun.cgs.value # erg s^-1
    au = const.au.cgs.value       # cm
    M_bol_sun = 4.75
    
    # Constant
    A = (np.pi * L_sun / (1.296 * 10**9 * au)**2)

    f_bol = A * 10**(-0.4 * (mbol - M_bol_sun - 10))
    

    B = -0.4* np.log(10)
    f_bol_dot = A * 10**(0.4*(M_bol_sun+10)) * B * np.e**(B*mbol)

    e_fbol = np.abs(f_bol_dot) * e_mbol

    return f_bol, e_fbol


def calc_radii(teff, e_teff, fbol, e_fbol, dist, e_dist,):
    """
    """
    # Constants
    pc = const.pc.cgs.value
    r_sun = const.R_sun.cgs.value
    sigma = const.sigma_sb.cgs.value

    # Calculate radii
    radii = dist * pc * np.sqrt(fbol / (sigma*teff**4)) / r_sun

    # Calculate uncertainty on radius (assuming no covariance)
    e_radii = radii * ((e_dist/dist)**2 + (e_fbol/(4*fbol))**2 
                        + (2*e_teff/teff)**2)**0.5
    #e_radii = np.sqrt(np.sum([
    #    (sigma**(-0.5) * fbol**0.5 * teff**-2)**2 * e_dist_cm**2,
    #    (0.5*sigma**(-0.5) * dist_cm * fbol**-0.5 * teff**-2)**2 * e_dist_cm**2,
    #    (-2*sigma**(-0.5) * dist_cm * fbol**0.5 * teff**-3)**2 * e_teff**2,
    #], axis=0)) / r_sun

    return radii, e_radii


def calc_L_star(fbol, e_fbol, dist, e_dist):
    """Calculate the stellar luminosity using the bolometric flux 
    """
    L_sun = const.L_sun.cgs.value # erg s^-1
    pc = const.pc.cgs.value       # cm
    
    L_star = 4 * np.pi * fbol * (dist*pc)**2 / L_sun

    e_L_star = L_star * ((e_fbol/fbol)**2 + 4*(e_dist/dist)**2)**0.5

    return L_star, e_L_star


def compute_final_params(
    observations, 
    all_sampled_params, 
    filters=["Bp", "Rp", "J", "H", "K"]):
    """Computes final stellar parameters and uncertainties from mean and std
    values of each fbol, plus radius, and luminosity. Final parameters are
    saved to observations.

    Parameters
    ----------
    observations: pandas.DataFrame
        Pandas dataframe containing observational info and results of synthetic
        fitting for each star.
    
    all_sampled_params: collections.OrderedDict
        Ordered dictionary of pairing key of stellar ID to pandas dataframe of
        sampled parameters from sample_params.
    
    filters: list of str, default: ["Bp", "Rp", "J", "H", "K"]
        List of filter bands to sample photometry for. Note that this should
        correspond to the settings in bolometric-corrections.
    """
    # Assemble columns for bolometric fluxes
    f_bol_cols= ["f_bol_{}".format(filt) for filt in filters] + ["f_bol_avg"]
    e_f_bol_cols = ["e_f_bol_{}".format(filt) for filt in filters]
    e_f_bol_cols += ["e_f_bol_avg"]

    f_bol_cols_interleaved = [val for pair in zip(f_bol_cols, e_f_bol_cols) 
                              for val in pair] 

    result_cols = ["radius", "e_radius", "lum", "e_lum"]
    result_cols = f_bol_cols_interleaved + result_cols

    # Make placeholder results dataframe
    result_df = pd.DataFrame(
        data=np.full((len(observations), len(result_cols)), np.nan), 
        index=observations.index, 
        columns=result_cols)

    for star_i, star_data in observations.iterrows():
        # Get source id
        source_id = star_data["source_id"]

        # Compute final fluxes and uncertainties
        result_df.loc[star_i][f_bol_cols] = np.mean(
            all_sampled_params[source_id][f_bol_cols], axis=0)

        result_df.loc[star_i][e_f_bol_cols] = np.std(
            all_sampled_params[source_id][f_bol_cols], axis=0)

        # Compute final radii and uncertainties
        result_df.loc[star_i]["radius"] = np.mean(
            all_sampled_params[source_id]["radius"], axis=0)

        result_df.loc[star_i]["e_radius"] = np.std(
            all_sampled_params[source_id]["radius"], axis=0)

        # Compute final luminosities and uncertainties
        result_df.loc[star_i]["lum"] = np.mean(
            all_sampled_params[source_id]["lum"], axis=0)

        result_df.loc[star_i]["e_lum"] = np.std(
            all_sampled_params[source_id]["lum"], axis=0)
    
    # Convert radii to solar units
    result_df["radius"] /= const.R_sun.cgs.value
    result_df["e_radius"] /= const.R_sun.cgs.value

    # Convert luminosity to solar units
    result_df["lum"] /= const.L_sun.cgs 
    result_df["e_lum"] /= const.L_sun.cgs 

    # Add results to observations
    for col in result_df.columns:
        observations[col] = result_df[col]


# -----------------------------------------------------------------------------
# Dustmaps/Extinction
# -----------------------------------------------------------------------------
def calculate_A_G(
    ra,
    dec,
    dist_pc,
    dm_query=None,
    force_local_bubble=True,
    local_bubble_real_a_g_threshold=0.01,
    n_pc_step=2,
    dustmap="leike_glatzle_ensslin_2020",
    verbose=False,
    do_plot=False,):
    """
    Computes the extinction in the Gaia G band, A_G, using the dustmaps python 
    package: https://dustmaps.readthedocs.io/en/latest/index.html

    Implemented dustmaps are currently the 3D dust maps from:
    a) Leike & Enßlin 2019
        https://ui.adsabs.harvard.edu/abs/2019A%26A...631A..32L/abstract
    b) Leike, Glatzle, & Enßlin 2020
        https://ui.adsabs.harvard.edu/abs/2020A%26A...639A.138L/abstract

    Note that the 2020 predicts more dust than the 2019 paper, with the paper
    describing the 2019 paper as underpredicting dust, and the 2020 paper 
    perhaps overpredicting dust. The default here is the 2020 paper.

    Parameters
    ----------
    ra, dec: float
        Right ascension and declination in degrees.

    dist_pc: float
        Distance of star in pc.

    dm_query: DustMap object, default None
        DustMap object, either LeikeEnsslin2019Query or Leike2020Query. If not
        passed in, is initialised here.

    force_local_bubble: boolean, default True
        Whether to set A_G of stars with non-significant extinction inside the
        local bubble to zero.

    local_bubble_real_a_g_threshold: float, default 0.01
        Level below which we consider local bubble stars to be unreddened.

    dustmap: str
        Dustmap to import, either leike_ensslin_2019 or 
        leike_glatzle_ensslin_2020.

    verbose, do_plot: boolean, default: False
        Whether to print summary and plot

    Returns
    -------
    A_G: float array
        Vector of calculcated extinctions in Gaia G band, A_G.
    """
    if dm_query is None:
        if dustmap == "leike_ensslin_2019":
            dm_query = LeikeEnsslin2019Query()
        elif dustmap == "leike_glatzle_ensslin_2020":
            dm_query = Leike2020Query()
        else:
            raise ValueError("Invalid dustmap provided, must be in".format(
                valid_dustmaps))

    # Have to integrate, so query the grid in n_pc_step pc steps, then add on
    # the actual distance
    dists = np.arange(0, dist_pc, n_pc_step)

    if dist_pc not in dists:
        dists = np.concatenate([dists, [dist_pc]])
    
    # Initialise array to hold extinctions
    ext_e_folds = []

    # Get extinction density for every distance step, then integrate
    for dist in dists:
        coords = SkyCoord(ra*units.deg, dec*units.deg, distance=dist*units.pc)
        ext_e_folds.append(dm_query.query(coords=coords))

    ext_e_folds = np.array(ext_e_folds)

    # Integrate
    total_ext_e_folds = simps(ext_e_folds, x=dists)

    # Convert to fraction of light getting through
    total_ext_frac = np.exp(-total_ext_e_folds)

    # Calculate A_G, and change log base to 10 to convert to magnitudes
    A_G = -2.5*np.log10(np.exp(-total_ext_e_folds))

    # If enforcing the limits of the Local Bubble, don't apply reddening to 
    # stars within it *unless* they exceed local_bubble_real_a_g_threshold,
    # which is set to 0.01 by default - i.e. anything below this we consider
    # to be noise.
    if (force_local_bubble and dist < LOCAL_BUBBLE_DIST_PC 
        and A_G < local_bubble_real_a_g_threshold):
        # Set A_G to 0
        A_G = 0

    # Do printing and plotting if requested
    if verbose:
        print(
            "{:6.2f} pc -->".format(dist_pc),
            total_ext_e_folds,
            total_ext_frac,
            "A_G = {:0.4f}".format(A_G))

    if do_plot:
        import matplotlib.pyplot as plt
        plt.plot(dists, np.array(np.exp(-ext_e_folds)))

    return A_G
    

def calculate_A_G_all(
    info_cat,
    dustmap="leike_glatzle_ensslin_2020",
    verbose=False,
    do_plot=False,):
    """For every star in info_cat, use RA, DEC, and distance to calculate the
    extinction A_G in the Gaia band.
    """
    A_G_all = []

    valid_dustmaps = ["leike_ensslin_2019", "leike_glatzle_ensslin_2020"]

    # Select which dust map to load in
    if dustmap == "leike_ensslin_2019":
        dm_query = LeikeEnsslin2019Query()
    elif dustmap == "leike_glatzle_ensslin_2020":
        dm_query = Leike2020Query()
    else:
        raise ValueError("Invalid dustmap provided, must be in".format(
            valid_dustmaps))

    # For each star, integrate extinction densities for A_G
    if not verbose:
        for star_i, star in tqdm(info_cat.iterrows(), desc="Calculating A_G",
            total=len(info_cat)):
            A_G_all.append(calculate_A_G(
                ra=star["ra"],
                dec=star["dec"],
                dist_pc=star["dist"],
                dm_query=dm_query,
                verbose=verbose,
                do_plot=do_plot,))
    else:
        for star_i, star in info_cat.iterrows():
            A_G_all.append(calculate_A_G(
                ra=star["ra"],
                dec=star["dec"],
                dist_pc=star["dist"],
                dm_query=dm_query,
                verbose=verbose,
                do_plot=do_plot,))

    A_G_all = np.asarray(A_G_all)

    return A_G_all


def calculate_per_band_reddening(A_G):
    """Computes E(B-V) and A_zeta, the extinction in filter band zeta, given
    a value of A_G. 

    Notes on extinction coefficients:
     - SkyMapper coefficients from Casagrande+18 (2019MNRAS.482.2770C)
     - Gaia coefficients from Casagrande+21 (arXiv:2011.02517)
     - 2MASS coefficients from Casagrande & VandenBerg (2014MNRAS.444..392C)

    Note that while some bands show a dependence on Teff, only the nominal 
    coefficients are implemented here.

    SkyMapper coefficients (other than u) vary by < 0.1 over 3,500-10,000 K. u
    shows a dependence on Teff.

    For Gaia, all bands show a Teff dependence, but Luca doesn't go cool enough
    for the fitted function to be valid for the coolest stars. As such, this 
    function here adopts the mean Bp-Rp of the TESS sample of ~2.03 which sits
    in a region where the interpolation is reliable.

    2MASS coefficients have weak dependence on Teff.

    Parameters
    ----------
    A_G: float array
        Array of magnitudes of extinction in Gaia G band
    
    Returns
    -------
    ebv: float array
        E(B-V) values corresponding to A_G.

    A_zeta: dict
        Dictionary with keys [u, v, g, r, i, z, BP, RP, G, K, H, K] which pair
        to arrays of the corresponding A_zeta values.
    """
    # Dictionary of our extinction coefficients R_zeta
    ext_coeff = {
        "u":4.88,   # Note this has a strong dependence on Teff
        "v":4.55,
        "g":3.43,
        "r":2.73,
        "i":1.99,
        "z":1.47,
        "BP":2.98,  # Teff dependent, computed for Bp-Rp = 2.03
        "RP":1.93,  # Teff dependent, computed for Bp-Rp = 2.03
        "G":2.26,   # Teff dependent, computed for Bp-Rp = 2.03
        "J":0.899,
        "H":0.567,
        "K":0.366,
    }

    # Calculate E(B-V)
    ebv = A_G / ext_coeff["G"]

    A_zeta = {}

    # Now calculate A_zeta for each other band
    for filt_zeta in ext_coeff.keys():
        A_zeta[filt_zeta] = ebv * ext_coeff[filt_zeta]

    return ebv, A_zeta
        

# -----------------------------------------------------------------------------
# Photometric [Fe/H] relation
# -----------------------------------------------------------------------------
def calc_photometric_feh_with_coeff_import(
    bp_k,
    k_mag_abs,
    bp_rp,
    isolated_ms_star_mask,
    ms_coeff_path="data/phot_feh_rel_ms_coeff.csv",
    offset_coeff_path="data/phot_feh_rel_offset_coeff.csv",
    e_phot_feh=0.19,
    bp_rp_bounds=(1.51,3.3)):
    """
    Parameters
    ----------
    bp_k: float array
        (Bp-Ks) colour for each star.

    k_mag_abs: float array
        Absolute Ks band magnitude.
    
    bp_rp: float array
        (Bp-Rp) colour for each star.

    isolated_ms_star_mask: boolean array
        Array indicating which stars are able to be used for the photometric
        [Fe/H] relation, True where valid, and False where not.

    ms_coeff_path, offset_coeff_path: str
        Paths to the save polynomial coefficients for the mean main sequence,
        and fitted offset in (Bp-Ks). Expected file format is a single line 
        CSV, with coefficients order from lowest order to highest order.

    e_phot_feh: float, default: 0.19
        Uncertainty on the predicted [Fe/H]

    bp_rp_bounds: float array, default: (1.51,3.4)
        The bounds of the relation in (Bp-Rp)

    Returns
    -------
    phot_fehs, e_phot_fehs: float array
        Predicted photometric [Fe/H] plus associated uncertainties.
    """
    # Import both sets of coefficients
    ms_coeff = np.loadtxt(ms_coeff_path)
    offset_coeff = np.loadtxt(offset_coeff_path)

    # Make the MS polynomial
    main_seq = np.polynomial.polynomial.Polynomial(ms_coeff)

    # Calculate photometric [Fe/H]
    phot_fehs = calc_photometric_feh(
        offset_coeff, 
        bp_k, 
        k_mag_abs, 
        main_seq, 
        use_bp_k_offset=True,)

    # Assign uncertainties
    e_phot_fehs = np.ones_like(phot_fehs) * e_phot_feh

    # Finally, assign nans to anything suspected of not being an isolated main
    # sequence star (using the provided mask) *or* outside of the valid Bp-Rp
    # range
    valid_bp_rp_mask = np.logical_and(
        bp_rp > bp_rp_bounds[0], 
        bp_rp < bp_rp_bounds[1])

    valid_star_mask = np.logical_and(isolated_ms_star_mask, valid_bp_rp_mask)

    phot_fehs[~valid_star_mask] = np.nan
    e_phot_fehs[~valid_star_mask] = np.nan

    return phot_fehs, e_phot_fehs


def calc_photometric_feh(
    params, 
    bp_k, 
    k_mag_abs, 
    main_seq, 
    use_bp_k_offset,
    offset_poly_order_colour=None,
    offset_poly_order_mks=None,
    bp_rp=None,):
    """Calculate [Fe/H] given Bp-K, M_Ks, the mean main sequence, and 
    polynomial coefficients.
    """
    # Compute offsets
    delta_mk = k_mag_abs - main_seq(bp_k)
    delta_bp_k = bp_k - main_seq(k_mag_abs)

    # Use absolute K band offset from MS (per Johnson & Apps 2009)
    if not use_bp_k_offset:
        delta_mk = k_mag_abs - main_seq(bp_k)

        feh_poly = np.polynomial.polynomial.Polynomial(params)
        feh_fit = feh_poly(delta_mk)

    # Use (Bp-K) colour offset from MS (per Schlaufman & Laughlin 2010)
    elif use_bp_k_offset:
        # Do old way using only a single polynomial in colour
        if offset_poly_order_colour is None or offset_poly_order_mks is None:
            feh_poly = np.polynomial.polynomial.Polynomial(params)
            feh_fit = feh_poly(delta_bp_k)

        # Otherwise use two polynomials
        else:
            # Compute the colour component of the polynomial. There is no
            # zeroeth order term here as it is added in later.
            if offset_poly_order_colour > 0:
                coeff = np.concatenate(
                    ([0], params[:offset_poly_order_colour]))
                feh_poly_colour = np.polynomial.polynomial.Polynomial(coeff)
                feh_poly_colour_val = feh_poly_colour(bp_k)
            else:
                feh_poly_colour_val = 0

            # Compute the colour component of the polynomial. There is no
            # zeroeth order term here as it is added in later.
            if offset_poly_order_mks > 0:
                coeff = np.concatenate(
                    ([0], params[offset_poly_order_colour:-1]))
                feh_poly_mks = np.polynomial.polynomial.Polynomial(coeff)
                feh_poly_mks_val = feh_poly_mks(k_mag_abs)
            else:
                feh_poly_mks_val = 0

            # Calculate [Fe/H] from the addition of both polynomials, plus the
            # constant
            feh_fit = feh_poly_colour_val + feh_poly_mks_val + params[-1]

    return feh_fit


def calc_feh_resid(
    params, 
    bp_k, 
    k_mag_abs, 
    feh, 
    e_feh, 
    main_seq, 
    use_bp_k_offset,
    offset_poly_order_colour,
    offset_poly_order_mks,
    bp_rp):
    """Calculate residuals from [Fe/H] fit for optimisation
    """
    feh_fit = calc_photometric_feh(
        params=params, 
        bp_k=bp_k, 
        k_mag_abs=k_mag_abs, 
        main_seq=main_seq, 
        use_bp_k_offset=use_bp_k_offset,
        offset_poly_order_colour=offset_poly_order_colour,
        offset_poly_order_mks=offset_poly_order_mks,
        bp_rp=bp_rp,)

    resid = (feh - feh_fit) / e_feh

    return resid


def fit_feh_model(
    bp_k, 
    k_mag_abs, 
    feh, 
    e_feh, 
    main_seq,
    use_bp_k_offset,
    offset_poly_order=None,
    offset_poly_order_colour=None,
    offset_poly_order_mks=None,
    bp_rp=None,):
    """Fit [Fe/H] model
    """
    # For compatability, allow use of offset_poly_order
    if offset_poly_order is not None:
        # Warn user, then set defaults
        warn(("'offset_poly_order' is deprecated, use "
            "'offset_poly_order_colour' and 'offset_poly_order_mks' instead."))
        offset_poly_order_colour = offset_poly_order
        offset_poly_order_mks = 0

        init_params = np.ones(offset_poly_order + 1)
        
    # Otherwise our polynomial coefficients will be composed of both colour and
    # absolute magnitude terms.
    else:
        init_params = np.ones(offset_poly_order_colour + offset_poly_order_mks + 1)

    # Setup parameter vector
    args = (bp_k, k_mag_abs, feh, e_feh, main_seq, use_bp_k_offset, 
        offset_poly_order_colour, offset_poly_order_mks, bp_rp)

    
    #diff_step = np.ones(offset_poly_order+1) * 0.01
    opt_res = least_squares(
        calc_feh_resid,
        init_params,
        jac="2-point",
        args=args,
        #method="lm",
        #diff_step=diff_step,
    )

    return opt_res

# -----------------------------------------------------------------------------
# Saving and loading sampled params
# -----------------------------------------------------------------------------
def save_sampled_params(all_sampled_params, label, path):
    """Save the sampled params to a multi-HDU fits file, which has N equals
    the number of stars sampled, plus the empty primary HDU. The saved file
    will have the name format 'sampled_params_<label>.fits'.

    Parameters
    ----------
    all_sampled_params: OrderedDict
        An OrderedDict with keys of stellar ID mapping to the table of sampled
        parameters

    label: string
        Unique identifier for the file.
    
    path: string
        The path to where the fits file is stored, either relative or absolute.
    """
    # Intialise HDU List
    hdu = fits.HDUList()
    
    # Make a fits table HDU for every star
    for star in all_sampled_params.keys():
        
        sampled_params = fits.BinTableHDU(
            Table.from_pandas(all_sampled_params[star]))
        sampled_params.header["EXTNAME"] = (str(star), "sampled params")
        hdu.append(sampled_params)

    # Done, save
    save_path = os.path.join(path,  "sampled_params_{}.fits".format(label))
    hdu.writeto(save_path, overwrite=True)


def load_sampled_params(label, path):
    """Load in the sampled params from the saved fits file, which has N equals
    the number of stars sampled, plus the empty primary HDU. The saved file
    should have the name format 'sampled_params_<label>.fits'.

    Parameters
    ----------
    label: string
        Unique identifier for the file.
    
    path: string
        The path to where the fits file is stored, either relative or absolute.

    Returns
    -------
    all_sampled_params: OrderedDict
        An OrderedDict with keys of stellar ID mapping to the table of sampled
        parameters
    """
    # Load in the fits file
    fits_path = os.path.join(path, "sampled_params_{}.fits".format(label))

    all_sampled_params = OrderedDict()

    # For every fits extension load the table in as a pandas dataframe
    with fits.open(fits_path) as fits_file: 
        for star_i in range(1,len(fits_file)):
            name = fits_file[star_i].name
            sampled_params = Table(fits_file[star_i].data).to_pandas()
            all_sampled_params[name] = sampled_params
        
    return all_sampled_params