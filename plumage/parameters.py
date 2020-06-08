"""
"""
import numpy as np
import pandas as pd
import astropy.constants as const
import plumage.spectra as spec
from numpy.polynomial.polynomial import polyval as polyval
from scipy.interpolate import LinearNDInterpolator

def compare_to_standard_spectrum(spec_sci, spec_std):
    """
    """
    # Get a wavelength mask
    wl_mask = spec.make_wavelength_mask(spec_sci[0,:], True,
                                        mask_sky_emission=True)

    # Get the chi^2 of the fit
    chi2 = np.nansum(((spec_sci[1,:][wl_mask] - spec_std[1,:][wl_mask]) 
                        / spec_std[2,:][wl_mask])**2)

    return chi2


def compare_sci_to_all_standards(spec_sci, spec_stds, std_params):
    """
    """
    chi2_all = np.ones(len(spec_stds))

    for std_i, spec_std in enumerate(spec_stds):
        chi2 = compare_to_standard_spectrum(spec_sci, spec_std)
        chi2_all[std_i] = chi2

    sorted_std_idx = np.argsort(chi2_all)

    print(chi2_all[sorted_std_idx][:3])
    print(std_params.iloc[sorted_std_idx][:3])

    #wl_mask = spec.make_wavelength_mask(spec_sci[0,:], True,
    #                                    mask_sky_emission=True)

    #plt.plot(spec_sci[0,:][wl_mask], spec_sci[1,:][wl_mask], label="Science")
    #plt.plot(spec_stds[0,:][wl_mask], spec_sci[1,:][wl_mask], label="Science")


def compare_fits_to_lit(observations, std_info):
    """Compares synthetic fits to literatue values
    """
    diff = []
    source = []
    n_matches = 0
    n_dups = 0
    dups = []

    for i in range(len(observations)):
        # ID
    
        sid = observations.iloc[i]["uid"]
        is_obs = std_info["observed"].values
        star_info = std_info[is_obs][std_info[is_obs]["source_id"]==sid]

        if len(star_info) < 1:
            diff.append([np.nan, np.nan, np.nan])
            source.append(["",""])
        elif len(star_info) > 1:
            diff.append([np.nan, np.nan, np.nan])
            n_dups += 1
            dups.append(sid)
            source.append(["",""])
        else:
            n_matches += 1
            star_info = star_info.iloc[0]
            synth_params = observations.iloc[i][[
                "teff_synth", "logg_synth", "feh_synth"]].values
            lit_params = star_info[["teff", "logg", "feh"]].values
            diff.append(synth_params-lit_params)
            source.append([star_info["kind"],star_info["source"]])
    
    print("{} matches".format(n_matches))
    print("{} duplicates".format(n_dups))
    
    diff = np.array(diff).astype(float)

    observations["teff_diff"] = diff[:,0]
    observations["logg_diff"] = diff[:,1] 
    observations["feh_diff"] = diff[:,2] 

    return diff, np.array(source)

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
    e_mass = 0.02

    # Coefficients for 5th order polynomial fit without [Fe/H] dependence
    coeff = np.array(
        [-0.642, -0.208, -8.43*10**-4, 7.87*10**-3, 1.42*10**-4, -2.13*10**-4]
        )

    # Calculate masses
    masses = 10**polyval(k_mag_abs-zp, coeff)
    e_masses = np.ones_like(masses) * e_mass

    return masses, e_masses
    

def compute_mann_2015_teff(
    colour,
    j_h=None,
    feh=None,
    relation="BP - RP, J - H",
    teff_file="data/mann_2015_teff.txt", 
    sigma_spec=60,
    ):
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
        V - J, [Fe/H]
        V - Ic, [Fe/H]
        r - z, [Fe/H]
        r - J, [Fe/H]
        BP - RP, J - H
        V - J, J - H
        V - Ic, J - H
        r - z, J - H
        r - J, J - H

    Parameters
    ----------
    colour: float array
        Photometric colour used for the relation, e.g. Bp-Rp.

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
    # Import the table of colour relations
    m15_teff = pd.read_csv(teff_file, delimiter="\t", comment="#", 
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
    x_coeff = m15_teff.loc[relation][["a", "b", "c", "d", "e"]]
    color_comp = polyval(colour, x_coeff)

    # Now calculate the metallicity component, which either uses [Fe/H] 
    # directly, or J-H as a proxy. These are mutually exclusive

    # J-H component
    if "J - H" in relation:
        jh_comp = (m15_teff.loc[relation]["f"] * j_h 
                   + m15_teff.loc[relation]["g"] * j_h**2)
        feh_comp = 0

    # [Fe/H] component
    elif "[Fe/H]" in relation:
        feh_comp = m15_teff.loc[relation][["f"]] * feh
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

    return teffs, e_teffs


def compute_mann_2015_radii(k_mag_abs):
    """Calculates stellar radii based on absolute 2MASS Ks band magnitudes
    per the empirical relations in Table 1 of Mann et al. 2015. 

    Paper:
        https://iopscience.iop.org/article/10.1088/0004-637X/804/1/64
    
    Erratum:
        https://iopscience.iop.org/article/10.3847/0004-637X/819/1/87

    The implemented relation is the 3rd order polynomial fit without the 
    dependence on [Fe/H].

    Parameters
    ----------
    k_mag_abs: float array
        Array of absolute 2MASS Ks band magnitudes

    Returns
    -------
    radii: float array
        Resulting stellar radii in solar units.

    e_masses: float array
        Uncertainties on stellar radii in solar units.
    """
    # Percentage uncertainty on the result
    e_radii_pc = 0.0289

    # Coefficients for 3rd order polynomial fit without [Fe/H] dependence
    coeff = np.array([1.9515, -0.3520, 0.01680])

    # Calculate radii
    radii = polyval(k_mag_abs, coeff)
    e_radii = radii * e_radii_pc

    return radii, e_radii

# -----------------------------------------------------------------------------
# Other params
# -----------------------------------------------------------------------------
def compute_semi_major_axis(mass, period):
    """
    """
    G = const.G.si.value
    period = period * 60 * 60 * 24          # Convert days to seconds
    mass = mass * const.M_sun.si.value      # Convert M_sun units to kg

    sma = ((G*mass*period**2) / (4*np.pi**2))**(1/3)

    return sma, np.nan