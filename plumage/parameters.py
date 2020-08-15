"""
"""
import os
import numpy as np
import pandas as pd
import astropy.constants as const
import plumage.spectra as spec
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
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