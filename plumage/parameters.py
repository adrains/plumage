"""Functions dealing with the computation of stellar parameters or photometry.
"""
import os
import numpy as np
import pandas as pd
from warnings import warn
import astropy.units as units
import astropy.constants as const
from astropy.coordinates import SkyCoord
import plumage.spectra as spec
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
from numpy.polynomial.polynomial import Polynomial

LOCAL_BUBBLE_DIST_PC = 70

# -----------------------------------------------------------------------------
# Benchmark star systematics, uncertainties, and citations
# -----------------------------------------------------------------------------
# TODO Import this from a file rather than having it globally here
# [Fe/H] offsets from Mann+13
FEH_OFFSETS = {
    "TW":0.0,
    "VF05":0.0,
    "CFHT":0.0,
    "C01":0.02,
    "M04" :0.04,
    "LH05":0.02,
    "T05":0.01,
    "B06":0.07,
    "Ra07":0.08,
    "Ro07":0.00,
    "F08":0.03,
    "C11":0.00,
    "S11":0.03,
    "N12":0.00,
    "M18":-0.03,    # Computed from np.nanmedian(cpm_selected["[Fe/H]_vf05"]-cpm_selected["Fe_H_m18"])
    "Sou06":0.0,    # TODO Confirm this
    "Sou08":0.0,    # TODO Confirm this
    "Soz09":0.0,    # TODO Confirm this
    "M14":0.0,      # Probably safe to assume this is zero?
    "RA12":0.01,   # Computed from np.nanmedian(obs_join["feh_m15"] - obs_join["feh_ra12"])
    "M13":0.0,
    "R21":0.0,      # Photometric. Assumed = 0 since built from binaries.
}

# Adopted uncertainties from VF05 Table 6
VF05_ABUND_SIGMA = {
    "M_H":0.029,
    "Na_H":0.032,
    "Si_H":0.019,
    "Ti_H":0.046,
    "Fe_H":0.03,
    "Si_H":0.03,
}

# [Ti/H] offsets --> TODO
TIH_OFFSETS = {
    "VF05":0.0,
    "M18":0.0,
    "A12":0.00,
}

# Citations
FEH_CITATIONS = {
    "TW":"mann_spectro-thermometry_2013",
    "M13":"mann_spectro-thermometry_2013",
    "VF05":"valenti_spectroscopic_2005",
    "CFHT":"",
    "C01":"",
    "M04" :"mishenina_correlation_2004",
    "LH05":"luck_stars_2005",
    "T05":"",
    "B06":"bean_accurate_2006",
    "Ra07":"ramirez_oxygen_2007",
    "Ro07":"robinson_n2k_2007",
    "F08":"fuhrmann_nearby_2008",
    "C11":"casagrande_new_2011",
    "S11":"da_silva_homogeneous_2011",
    "N12":"neves_metallicity_2012",
    "M18":"montes_calibrating_2018",
    "Sou06":"sousa_spectroscopic_2006",
    "Sou08":"sousa_spectroscopic_2008",
    "Soz09":"sozzetti_keck_2009",
    "M14":"mann_prospecting_2014",
}

# -----------------------------------------------------------------------------
# Spectrum comparison. TODO: unsure if these are used anymore.
# -----------------------------------------------------------------------------
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


def compute_logg(masses, e_masses, radii, e_radii,):
    """

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


def compute_casagrande_2020_teff(
    colour,
    logg,
    feh,
    relation,
    teff_file="data/casagrande_colour_teff_2020.dat",):
    """Calculates Teff using a given colour relation and measure of stellar 
    logg and [Fe/H] per the Casagrande+2020 relations.

    Parameters
    ----------
    colour: float
        The stellar colour corresponding to 'relation'.
    
    logg: float
        Stellar surface gravity in log cgs units.

    feh: float
        Stellar metallciity relative to solar, i.e. [Fe/H].

    relation: string
        Colour relation to use. Valid options are:
            - (Bp-Rp)
            - (Bp-J)
            - (Bp-H)
            - (Bp-K)
            - (Rp-J)
            - (Rp-H)
            - (Rp-K)
            - (G-J)
            - (G-H)
            - (G-K)
            - (G-Bp)
            - (G-Rp)
        Where all colours have been corrected for reddening. Strings should be
        entered in all lowercase, e.g. "bp-rp".
    
    teff_file: string
        File of coefficients to load.

    Returns
    -------
    teff, e_teff: float
        Calculated stellar effective temperature and corresponding uncertainty.
    """
    # Load in 
    c20_teff = pd.read_csv(
        teff_file,
        delim_whitespace=True,
        comment="#",
        index_col="colour",) 

    # Ensure relation is valid
    if relation not in c20_teff.index.values:
        raise ValueError("Unsupported relation. Must be one of %s"
                         % c20_teff.index.values)

    # Get the coefficients
    coeff_cols = ["a{}".format(i) for i in range(15)]
    coeff = c20_teff.loc[relation][coeff_cols].values

    # Calculate each term of the polynomial
    poly_terms = [
        coeff[0],
        coeff[1]*colour,
        coeff[2]*colour**2,
        coeff[3]*colour**3,
        coeff[4]*colour**5,
        coeff[5]*logg,
        coeff[6]*logg*colour,
        coeff[7]*logg*colour**2,
        coeff[8]*logg*colour**3,
        coeff[9]*logg*colour**5,
        coeff[10]*feh,
        coeff[11]*feh*colour,
        coeff[12]*feh*colour**2,
        coeff[13]*feh*colour**3,
        coeff[14]*feh*logg*colour,
    ]

    # And sum to get the final temperature
    teff = np.sum(poly_terms)
    e_teff = c20_teff.loc[relation]["sigma_teff"]

    return teff, e_teff


# -----------------------------------------------------------------------------
# Label preparation for benchmark stars
# -----------------------------------------------------------------------------
def prepare_labels(
    obs_join,
    n_labels=3,
    e_teff_quad=60,
    max_teff=4200,
    abundance_labels=[],
    abundance_trends=None,
    calc_x_fe_abund=False,):
    """Prepare our set of training labels using our hierarchy of parameter 
    source preferences.

    Teff: Prefer interferometric measurements, otherwise take the uniform Teff
    scale from Rains+21 which has been benchmarked to the interferometric Teff
    scale. Add Rains+21 uncertainties in quadrature with standard M+15 
    uncertainties to ensure that interferometric benchmarks are weighted more
    highly. Enforce a max Teff limit to avoid warmer stars.

    Logg: uniform Logg from Rains+21 (Mann+15 intial guess, updated from fit)

    [Fe/H]: Prefer CPM binary benchmarks, then M+15, then RA+12, then [Fe/H]
    from other NIR relations (e.g. T+15, G+14), then just default for Solar 
    Neighbourhood with large uncertainties.

    Systematics, uncertainties, and citations are pulled from FEH_OFFSETS, 
    VF05_ABUND_SIGMA, and FEH_CITATIONS at the top of the file respectively.

    Parameters
    ----------
    obs_join: pandas DataFrame
        Pandas dataframe crossmatch containing observation information, Gaia
        2MASS, and benchmark information.

    n_labels: int, default: 3
        The number of labels for the Cannon model.

    e_teff_quad: int, default: 60
        Teff uncertainty to add in quadrature with the statistical 
        uncertainties.

    max_teff: int, default: 4200
        Maximum allowable benchmark temperature.

    abundance_labels: list, default: []
        List of string abundances (e.g. 'Ti_H') to use.

    abundance_trends: pandas DataFrame, default: None
        Polynomial coefficients of [X/H] abundance trend w.r.t [Fe/H] for each
        of [Na_H,Mg_H,Al_H,Si_H,Ca_H,Sc_H,Ti_H,V_H,Cr_H,Mn_H,Co_H,Ni_H] fitted
        to the Montes+18 sample where the abundances are columns and the rows
        are polyomial coefficents. Only linear trends have been tested.

    calc_x_fe_abund: boolean, default: False
        Indicates whether to determine [X/Fe] (True) or [X/H] (False).

    Returns
    -------
    label_values, label_sigma: float array
        Float array of shape [N_star, N_label] containing the adopted label
        values or uncertainties respectively. Missing values default to NaN.
    
    std_mask: boolean array
        Boolean mask of shape [N_star] which is True where a star is useful as
        a benchmark and has appropriate labels. 
    
     label_sources: string array
        String array containing paper abbreviations denoting the source of each
        adopted label.
    """
    # Intialise mask
    std_mask = np.full(len(obs_join), True)

    # Initialise label vector
    label_values = np.full( (len(obs_join), n_labels), np.nan)
    label_sigma = np.full( (len(obs_join), n_labels), np.nan)

    # Initialise record of label source/s
    label_sources = np.full( (len(obs_join), n_labels), "").astype(object)

    # And initialise record of whether we've assigned a default parameter
    label_nondefault = np.full( (len(obs_join), n_labels), False)

    # Go through one star at a time and select labels
    for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
        # Only accept properly vetted stars with consistent Teffs
        if not star_info["in_paper"]:
            std_mask[star_i] = False
            continue

        # Only accept interferometric, M+15, RA+12, NIR other, & CPM standards
        elif not (~np.isnan(star_info["teff_int"]) 
            or ~np.isnan(star_info["teff_m15"])
            or ~np.isnan(star_info["teff_ra12"])
            or ~np.isnan(star_info["feh_nir"])
            or star_info["is_cpm"]):
            std_mask[star_i] = False
            continue
        
        # Enforce our max temperature for interferometric standards
        elif star_info["teff_int"] > max_teff:
            std_mask[star_i] = False
            continue
        
        # ---------------------------------------------------------------------
        # Teff
        # ---------------------------------------------------------------------
        # Teff: interferometric > Rains+21
        if not np.isnan(star_info["teff_int"]):
            label_values[star_i, 0] = star_info["teff_int"]
            label_sigma[star_i, 0] = star_info["e_teff_int"]
            label_sources[star_i, 0] = star_info["int_source"]
            label_nondefault[star_i, 0] = True

        else:
            label_values[star_i, 0] = star_info["teff_synth"]
            label_sigma[star_i, 0] = (
                star_info["e_teff_synth"]**2 + e_teff_quad**2)**0.5
            label_sources[star_i, 0] = "R21"
            label_nondefault[star_i, 0] = True

        # ---------------------------------------------------------------------
        # logg
        # ---------------------------------------------------------------------
        # logg: Rains+21
        label_values[star_i, 1] = star_info["logg_synth"]
        label_sigma[star_i, 1] = star_info["e_logg_synth"]
        label_sources[star_i, 1] = "R21"
        label_nondefault[star_i, 1] = True

        # ---------------------------------------------------------------------
        # [Fe/H] - binary
        # ---------------------------------------------------------------------
        # VF+05 > M18 > others
        if star_info["is_cpm"]:
            # Valenti & Fischer 2005 --> Mann+15 base [Fe/H] scale
            if ~np.isnan(star_info["Fe_H_vf05"]):
                ref = "VF05"
                feh_corr = star_info["Fe_H_vf05"] + FEH_OFFSETS[ref]
                e_feh_corr = VF05_ABUND_SIGMA["Fe_H"]
                citation = FEH_CITATIONS[ref]

            # Montes+2018
            elif ~np.isnan(star_info["Fe_H_m18"]):
                ref = "M18"
                feh_corr = star_info["Fe_H_m18"] + FEH_OFFSETS[ref]
                e_feh_corr = star_info["eFe_H_m18"]
                citation = FEH_CITATIONS["M18"]

            # Sousa+08 (this is the base sample for Adibekyan+12 abundances)
            elif ~np.isnan(star_info["feh_s08"]):
                ref = "Sou08"
                feh_corr = star_info["feh_s08"] + FEH_OFFSETS[ref]
                e_feh_corr = star_info["e_feh_s08"]
                citation = FEH_CITATIONS["Sou08"]

            # Mann+2014 for VLM dwarfs
            elif ~np.isnan(star_info["feh_m14"]):
                ref = "M14"
                feh_corr = star_info["feh_m14"] + FEH_OFFSETS[ref]
                e_feh_corr = star_info["e_feh_m14"]
                citation = FEH_CITATIONS[ref]

            # -----------------------------------------------------------------
            # Note that after here things are a bit ~misc~

            # Mann+13
            elif ~np.isnan(star_info["feh_prim_m13"]):
                ref = star_info["ref_feh_prim_m13"]
                feh_corr = star_info["feh_prim_m13"] + FEH_OFFSETS[ref]
                e_feh_corr = star_info["e_feh_prim_m13"]
                citation = FEH_CITATIONS[ref]

            # Newton+14
            elif ~np.isnan(star_info["feh_prim_n14"]):
                ref = star_info["feh_prim_ref_n14"]
                feh_corr = star_info["feh_prim_n14"] + FEH_OFFSETS["VF05"]
                e_feh_corr = star_info["e_feh_prim_n14"]
                citation = FEH_CITATIONS[star_info["feh_prim_ref_n14"]]

            # Then check other entries
            elif star_info["feh_prim_ref_other"] == "Ra07":
                ref = "Ra07"
                feh_corr = star_info["feh_prim_other"] + FEH_OFFSETS[ref]
                e_feh_corr = star_info["e_feh_prim_other"]
                citation = FEH_CITATIONS["Ra07"]
            
            # In case we're missing a value, alert
            else:
                print("Missing CPM value")

            # Take final value
            label_values[star_i, 2] = feh_corr
            label_sigma[star_i, 2] = e_feh_corr
            label_sources[star_i, 2] = ref
            label_nondefault[star_i, 2] = True

        # ---------------------------------------------------------------------
        # [Fe/H] - single star
        # ---------------------------------------------------------------------
        # M+15 > RA+12 > NIR other > Rains+21 > default
        # Mann+15
        elif not np.isnan(star_info["feh_m15"]):
            label_values[star_i, 2] = star_info["feh_m15"]
            label_sigma[star_i, 2] = star_info["e_feh_m15"]
            label_sources[star_i, 2] = "M15"
            label_nondefault[star_i, 2] = True

        # Rojas-Ayala+2012
        elif not np.isnan(star_info["feh_ra12"]):
            label_values[star_i, 2] = \
                star_info["feh_ra12"] + FEH_OFFSETS["RA12"]
            label_sigma[star_i, 2] = star_info["e_feh_ra12"]
            label_sources[star_i, 2] = "RA12"
            label_nondefault[star_i, 2] = True

        # Other NIR relations
        elif not np.isnan(star_info["feh_nir"]):
            label_values[star_i, 2] = star_info["feh_nir"]
            label_sigma[star_i, 2] = star_info["e_feh_nir"]
            label_sources[star_i, 2] = star_info["nir_source"]
            label_nondefault[star_i, 2] = True

        # Rains+21 photometric [Fe/H]
        elif not np.isnan(star_info["phot_feh"]):
            label_values[star_i, 2] = star_info["phot_feh"]
            label_sigma[star_i, 2] = star_info["e_phot_feh"]
            label_sources[star_i, 2] = "R21"
            label_nondefault[star_i, 2] = True

        # Solar neighbourhood default
        else:
            label_values[star_i, 2] = -0.14 # Mean for Solar Neighbourhood
            label_sigma[star_i, 2] = 2.0    # Default uncertainty
            label_nondefault[star_i, 2] = False
            print("Assigned default [Fe/H] to {}".format(source_id))

        # Note the adopted [Fe/H]
        feh_adopted = label_values[star_i, 2]

        # ---------------------------------------------------------------------
        # Abundances
        # ---------------------------------------------------------------------
        # Finally setup [X/H] one element at a time. We bypass this loop if we
        # don't have any abundances.
        for abundance_i, abundance in enumerate(abundance_labels):
            label_i = 3 + abundance_i

            # Only allow supported abundances
            SUPPORTED_ABUNDANCES = ["Ti_H"]

            if abundance not in SUPPORTED_ABUNDANCES:
                raise ValueError("Unsupported abundance!")

            m18_abundance = "{}_m18".format(abundance)
            vf05_abundance = "{}_vf05".format(abundance)
            a12_abundance = "{}_a12".format(abundance)
            
            # Boolean to check if we have assigned a lit abundance
            lit_abund_assigned = False

            # Since VF05 is our base [Fe/H] reference, we'll use it as our base
            # reference for abundances as well. It also has lower uncertainties
            # than other references. TODO: account for abundance systematics.
            if not np.isnan(star_info[vf05_abundance]):
                label_values[star_i, label_i] = star_info[vf05_abundance]
                label_sigma[star_i, label_i] = VF05_ABUND_SIGMA[abundance]
                label_sources[star_i, label_i] = "VF05"
                label_nondefault[star_i, label_i] = True

                lit_abund_assigned = True

            # Montes+18
            elif not np.isnan(star_info[m18_abundance]):
                label_values[star_i, label_i] = star_info[m18_abundance]
                label_sigma[star_i, label_i] = \
                    star_info["e{}".format(m18_abundance)]
                label_sources[star_i, label_i] = "M18"
                label_nondefault[star_i, label_i] = True

                lit_abund_assigned = True

            # Adibekyan+2012. Note that we're currently doing what is probably
            # a hysically unreasonable HACK in just averaging the Ti I/H and 
            # Ti II/H abundances.
            elif not np.isnan(star_info["Fe_H_a12"]):
                if abundance == "Ti_H":
                    ti_i_abund = a12_abundance.replace("_H_", "I_H_")
                    ti_ii_abund = a12_abundance.replace("_H_", "II_H_")

                    abund = \
                        (star_info[ti_i_abund] + star_info[ti_ii_abund])/2
                    e_abund = np.sqrt(
                        star_info["e_{}".format(ti_i_abund)]**2
                        + star_info["e_{}".format(ti_ii_abund)]**2)
                else:
                    abund = star_info[a12_abundance]
                    e_abund = star_info["e_{}".format(a12_abundance)]

                label_values[star_i, label_i] = abund
                label_sigma[star_i, label_i] = e_abund
                label_sources[star_i, label_i] = "A12"
                label_nondefault[star_i, label_i] = True

                lit_abund_assigned = True
            
            # If we've assigned a literature abundance and we're producing
            # [X/Fe] then caculate [X/Fe] and propagate uncertainties
            if lit_abund_assigned and calc_x_fe_abund:
                # Grab labels
                fe_h_log10 = label_values[star_i, 2]
                e_fe_h_log10 = label_sigma[star_i, 2]

                x_h_log10 = label_values[star_i, label_i]
                e_x_h_log10 = label_sigma[star_i, label_i]

                # Unlog
                fe_h = 10**fe_h_log10
                e_fe_h = fe_h * np.log(10) * e_fe_h_log10

                x_h = 10**x_h_log10
                e_x_h = x_h * np.log(10) * e_x_h_log10

                # Calculate [X/Fe]
                x_fe = x_h / fe_h
                e_x_fe = x_fe * np.sqrt((e_x_h/x_h)**2+(e_fe_h/fe_h)**2)

                # Relog, save
                # TODO: this overwrites our existing adopted labels.
                label_values[star_i, label_i] = np.log10(x_fe)
                label_sigma[star_i, label_i] = e_x_fe / (x_fe * np.log(10))
                label_nondefault[star_i, label_i] = False

            # If we've *not* assigned a literature abundance, but we're
            # producing [X/Fe] values, calculate a default [X/Fe] value
            elif not lit_abund_assigned and calc_x_fe_abund:
                # Note: this is specically for [Ti/Fe] as fitted to GALAH DR2
                label_values[star_i, label_i] = -0.332 * feh_adopted + 0.078
                label_sigma[star_i, label_i] = 0.16
                label_nondefault[star_i, label_i] = False

            # If we've not assigned a literature abundance, but we're opting to
            # work in [X/H] space, produce a default [X/H] value
            elif not lit_abund_assigned and not calc_x_fe_abund:
                poly = Polynomial(
                    abundance_trends[abundance.replace("_m18", "")].values)
                X_H = poly(feh_adopted)
                label_values[star_i, label_i] = X_H
                label_sigma[star_i, label_i] = 2.0
                label_nondefault[star_i, label_i] = False

            # If we've assigned a literature abundance and are working in [X/H]
            # space, we don't need to do anything.
            elif lit_abund_assigned and not calc_x_fe_abund:
                continue
                
    return label_values, label_sigma, std_mask, label_sources, label_nondefault


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
            print("Using original method.")
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