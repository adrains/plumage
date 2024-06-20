"""Functions related to selecting literature benchmark parameters to train the
Cannon on.
"""
import numpy as np
import scipy.odr as odr
from numpy.polynomial.polynomial import Polynomial

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
    "M18":0.02,     # Computed from np.nanmedian(Fe_H_vf05-Fe_H_m18) in cpm_prim, 46 stars
    "Sou06":0.0,    # TODO Confirm this
    "Sou08":0.0,    # TODO Confirm this
    "Soz09":0.0,    # TODO Confirm this
    "M14":0.0,      # Probably safe to assume this is zero?
    "RA12":0.01,    # Computed from np.nanmedian(obs_join["feh_m15"] - obs_join["feh_ra12"])
    "G14":0.0,      # Assumed to be on the same scale as M15
    "M13":0.0,
    "R21":np.nan,
    "B16":0.0,      # Computed from np.nanmedian(Fe_H_vf05-Fe_H_b16) in cpm_prim, 30 stars
    "RB20":0.00,    # Computed from np.nanmedian(Fe_H_vf05-Fe_H_rb16) in cpm_prim, 18 stars
    "L18":0.0,      # TODO, compute the actual value
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

# Adopted uncertainties from Table 6 of B16
B16_ABUND_SIGMA = {
    "C_H":0.026,
    "N_H":0.042,
    "O_H":0.036,
    "Na_H":0.014,
    "Mg_H":0.012,
    "Al_H":0.028,
    "Si_H":0.008,
    "Ca_H":0.014,
    "Ti_H":0.012,
    "V_H":0.034,
    "Cr_H":0.014,
    "Mn_H":0.020,
    "Fe_H":0.010,
    "Ni_H":0.012,
    "Y_H":0.03,
}

# Adopted uncertainties from Table 4 of RB20 for pre-2004 sample
RB20_ABUND_SIGMA = {
#   [X/H]:sigma     limits
    "C_H":0.05,     # −0.60–0.64
    "N_H":0.09,     # −0.86–0.84
    "O_H":0.07,     # −0.36–0.77
    "Na_H":0.06,    # −1.09–0.78
    "Mg_H":0.03,    # −0.70–0.54
    "Al_H":0.05,    # −0.66–0.58
    "Si_H":0.03,    # −0.65–0.57
    "Ca_H":0.03,    # −0.73–0.54
    "Ti_H":0.03,    # −0.71–0.52
    "V_H":0.04,     # −0.85–0.46
    "Cr_H":0.03,    # −1.07–0.52
    "Mn_H":0.05,    # −1.40–0.66
    "Fe_H":0.02,    # −0.99–0.57
    "Ni_H":0.03,    # −0.97–0.63
    "Y_H":0.07,     # −0.87–1.35
}

# TODO: Read the paper and figure out what the actual uncertainties are
L18_ABUND_SIGMA = {
#   [X/H]:sigma     limits
    "Ti_H":0.00,
}

# [Ti/H] offsets, computed from complete cpm_primary table
TIH_OFFSETS = {
    "B16":-0.02,        # np.nanmedian(tih_vf05 - tih_b16), 30 stars
    "VF05":0.0,         # Reference sample
    "RB20":0.00,        # np.nanmedian(tih_vf05 - tih_rb20), 18 stars
    "M18":-0.03,        # np.nanmedian(tih_vf05 - tih_m18), 46 stars
    "A12":0.00,         # TODO: Not computed
    "L18":0.0,         # TODO: Not computed
    
}

# [Ti/Fe] offsets, computed from the *adopted* labels
Ti_Fe_OFFSETS = {
    "Monty":-0.03,     # np.median(Ti_Fe_vf05 - Ti_Fe_monty)
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
    "B16":"brewer_spectral_2016",
    "RB20":"rice_stellar_2020",
}

# -----------------------------------------------------------------------------
# Label preparation for benchmark stars
# -----------------------------------------------------------------------------
def prepare_labels(
    obs_join,
    e_teff_quad=60,
    max_teff=4200,
    synth_params_available=False,):
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

    e_teff_quad: float
        Sigma value to add in quadrature with the non-interferometric Teff
        statistical uncertainties from Rains+21.

    max_teff: int, default: 4200
        Maximum allowable benchmark temperature.

    Updated
    -------
    obs_join: DataFrame
        New columns added to DataFrame corresponding to adopted labels.
    """
    SUPPORTED_LABELS = ["teff", "logg", "feh", "Ti_H", "Ti_Fe"]
    N_LABELS = len(SUPPORTED_LABELS)

    # Intialise mask
    std_mask = np.full(len(obs_join), True)

    # Initialise label vector
    label_values = np.full( (len(obs_join), N_LABELS), np.nan)
    label_sigmas = np.full( (len(obs_join), N_LABELS), np.nan)

    # Initialise record of label source/s
    label_sources = np.full( (len(obs_join), N_LABELS), "").astype(object)

    # Initialise record of whether we've assigned a default parameter
    label_nondefault = np.full( (len(obs_join), N_LABELS), False)

    # And initialise vector of predicted defaults
    predicted_label_values = np.full( (len(obs_join), N_LABELS-3), np.nan)
    predicted_label_sigmas = np.full( (len(obs_join), N_LABELS-3), np.nan)

    # Go through one star at a time and select labels
    for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
        # ---------------------------------------------------------------------
        # Prepare mask
        # ---------------------------------------------------------------------
        # Only accept properly vetted stars with consistent Teffs
        #if not star_info["in_paper"]:
        #    std_mask[star_i] = False
        #    continue

        # Only accept interferometric, M+15, RA+12, NIR other, & CPM standards
        if not (~np.isnan(star_info["teff_int"])
            or ~np.isnan(star_info["teff_m15"])
            or ~np.isnan(star_info["teff_g14"])
            or ~np.isnan(star_info["teff_ra12"])
            or ~np.isnan(star_info["feh_nir"])
            or star_info["is_mid_k_dwarf"]
            or star_info["is_cpm"]):
            std_mask[star_i] = False
            continue
        
        # Enforce our max temperature for interferometric standards
        elif star_info["teff_int"] > max_teff:
            std_mask[star_i] = False
            continue
        
        # Final check, only proceed if we have a Mann+19 mass (and thus a logg)
        # This enforces that we only run on mid-K through M dwarfs
        elif np.isnan(star_info["mass_m19"]):
            std_mask[star_i] = False
            continue

        # ---------------------------------------------------------------------
        # Teff
        # ---------------------------------------------------------------------
        teff_value, teff_sigma, teff_source, teff_nondefault = \
            select_teff_label(
                star_info,
                e_teff_quad=e_teff_quad,
                synth_params_available=synth_params_available)

        label_values[star_i, 0] = teff_value
        label_sigmas[star_i, 0] = teff_sigma
        label_sources[star_i, 0] = teff_source
        label_nondefault[star_i, 0] = teff_nondefault

        # ---------------------------------------------------------------------
        # logg
        # ---------------------------------------------------------------------
        logg_value, logg_sigma, logg_source, logg_nondefault = \
            select_logg_label(star_info, synth_params_available)

        label_values[star_i, 1] = logg_value
        label_sigmas[star_i, 1] = logg_sigma
        label_sources[star_i, 1] = logg_source
        label_nondefault[star_i, 1] = logg_nondefault

        # ---------------------------------------------------------------------
        # [Fe/H]
        # ---------------------------------------------------------------------
        feh_value, feh_sigma, feh_source, feh_nondefault = \
            select_Fe_H_label(star_info)
        
        label_values[star_i, 2] = feh_value
        label_sigmas[star_i, 2] = feh_sigma
        label_sources[star_i, 2] = feh_source
        label_nondefault[star_i, 2] = feh_nondefault

        # ---------------------------------------------------------------------
        # [Ti/H] and [Ti/Fe]
        # ---------------------------------------------------------------------
        Ti_dict = select_Ti_label(star_info, feh_value, feh_sigma)
        
        # Update [Ti/H]
        label_values[star_i, 3] = Ti_dict["Ti_H_value_adopted"]
        label_sigmas[star_i, 3] = Ti_dict["Ti_H_sigma_adopted"]
        label_sources[star_i, 3] = Ti_dict["Ti_source"]
        label_nondefault[star_i, 3] = Ti_dict["Ti_nondefault"]

        # Update [Ti/Fe]
        label_values[star_i, 4] = Ti_dict["Ti_Fe_value_adopted"]
        label_sigmas[star_i, 4] = Ti_dict["Ti_Fe_sigma_adopted"]
        label_sources[star_i, 4] = Ti_dict["Ti_source"]
        label_nondefault[star_i, 4] = Ti_dict["Ti_nondefault"]

        # Store predicted values for record
        predicted_label_values[star_i, 0] = Ti_dict["Ti_H_value_predicted"]
        predicted_label_sigmas[star_i, 0] = Ti_dict["Ti_H_sigma_predicted"]

        predicted_label_values[star_i, 1] = Ti_dict["Ti_Fe_value_predicted"]
        predicted_label_sigmas[star_i, 1] = Ti_dict["Ti_Fe_sigma_predicted"]

        # ---------------------------------------------------------------------
        # Final check
        # ---------------------------------------------------------------------
        if np.sum(np.isnan(label_values[star_i])) > 0:
            std_mask[star_i] = False

    # -------------------------------------------------------------------------
    # Update Dataframe with selected labels
    # -------------------------------------------------------------------------
    # Compute variance
    label_var_all = label_sigmas**2

    # Add the adopted labels to the dataframe
    for lbl_i, lbl in enumerate(SUPPORTED_LABELS):
        obs_join["label_adopt_{}".format(lbl)] = label_values[:,lbl_i]
        obs_join["label_adopt_sigma_{}".format(lbl)] = label_sigmas[:,lbl_i]
        obs_join["label_adopt_var_{}".format(lbl)] = label_var_all[:,lbl_i]
        obs_join["label_source_{}".format(lbl)] = label_sources[:,lbl_i]
        obs_join["label_nondefault_{}".format(lbl)] = label_nondefault[:,lbl_i]

    # Add predicted labels. We currently assume that everything other than
    # Teff, logg, and [Fe/H] has a predicted value.
    for lbl_i, lbl in enumerate(SUPPORTED_LABELS[3:]):
        obs_join["label_predicted_{}".format(lbl)] = \
            predicted_label_values[:,lbl_i]
        obs_join["label_predicted_sigma_{}".format(lbl)] = \
            predicted_label_sigmas[:,lbl_i]

    # Add corresponding mask to the dataframe
    obs_join["has_complete_label_set"] = std_mask

    # Combine this mask with our quality cut mask to get the adopted benchmarks
    obs_join["is_cannon_benchmark"] = np.logical_and(
        obs_join["passed_quality_cuts"],
        obs_join["has_complete_label_set"],)
    
    # All done!


def select_teff_label(star_info, e_teff_quad, synth_params_available):
    """Produces our adopted Teff values for this specific star.

    Our current logg heirarchy is:
     1: Interferometry
     2: Rains+21

    Parameters
    ----------
    star_info: Pandas Series
        Single row of our obs_info DataFrame corresponding to a single star.

    e_teff_quad: float
        Sigma value to add in quadrature with the non-interferometric Teff
        statistical uncertainties from Rains+21.

    synth_params_available: TODO

    Returns
    -------
    teff_value, teff_sigma: float
        Adopted value and sigma for Teff.
        
    teff_source: str
        Source publication code (e.g. R21) for adopted Teff.
    
    teff_nondefault: bool
        False if the adopted value comes from an empirical relation, True
        otherwise.
    """
    # First preference: interferometric Teff
    if not np.isnan(star_info["teff_int"]):
        teff_value = star_info["teff_int"]
        teff_sigma = star_info["e_teff_int"]
        teff_source = star_info["int_source"]
        teff_nondefault = True

    # Otherwise uniform Teff from Rains+21
    elif synth_params_available:
        teff_value = star_info["teff_synth"]
        teff_sigma = (
            star_info["e_teff_synth"]**2 + e_teff_quad**2)**0.5
        teff_source = "R21"
        teff_nondefault = True

    # Teff from empirical relations. M dwarfs will have Teff from the Mann+2015
    # relations, and K dwarfs and up will have Teff from the Casagrande+2021
    # relations, but for the overlap sample (late-K dwarfs) that have both,
    # we will prioritise the Casagrande+2021 value.
    else:
        # Grab booleans for convenience/readability
        has_M15_teff = not np.isnan(star_info["teff_M15_BP_RP_feh"])
        has_C21_teff = not np.isnan(star_info["teff_C21_BP_Ks_logg_feh"])

        # M-dwarfs
        if has_M15_teff and not has_C21_teff:
            teff_value = star_info["teff_M15_BP_RP_feh"]
            teff_sigma = star_info["e_teff_M15_BP_RP_feh"]
            teff_source = "M15er"
            teff_nondefault = True

        # Late-K dwarfs
        elif has_M15_teff and has_C21_teff:
            teff_value = star_info["teff_C21_BP_Ks_logg_feh"]
            teff_sigma = star_info["e_teff_C21_BP_Ks_logg_feh"]
            teff_source = "C21"
            teff_nondefault = True

        # Mid-K and warmer
        elif not has_M15_teff and has_C21_teff:
            teff_value = star_info["teff_C21_BP_Ks_logg_feh"]
            teff_sigma = star_info["e_teff_C21_BP_Ks_logg_feh"]
            teff_source = "C21"
            teff_nondefault = True
        
        # Problem!
        else:
            raise ValueError("No valid Teff value, {}".format(star_info.name))

    return teff_value, teff_sigma, teff_source, teff_nondefault
            

def select_logg_label(star_info, synth_params_available):
    """Produces our adopted logg values for this specific star.

    Our current logg heirarchy is:
     1: Rains+21

    Parameters
    ----------
    star_info: Pandas Series
        Single row of our obs_info DataFrame corresponding to a single star.

    synth_params_available: TODO

    Returns
    -------
    logg_value, logg_sigma: float
        Adopted value and sigma for logg.
        
    logg_source: str
        Source publication code (e.g. R21) for adopted logg.
    
    logg_nondefault: bool
        False if the adopted value comes from an empirical relation, True
        otherwise.
    """
    # logg from Rains+21 (+Rains+24 Cannon model)
    if synth_params_available:
        logg_value = star_info["logg_synth"]
        logg_sigma = star_info["e_logg_synth"]
        logg_source = "R21"
        logg_nondefault = True

    # logg from Mann+2015 and Mann+2019 empirical relations
    else:
        logg_value = star_info["logg_m19"]
        logg_sigma = star_info["e_logg_m19"]
        logg_source = "M19"
        logg_nondefault = True

    return logg_value, logg_sigma, logg_source, logg_nondefault


def select_Fe_H_label(star_info,):
    """Produces our adopted [Fe/H] values for this specific star.

    Our current [Fe/H] heirarchy is:
     1: Cool K/M secondaries in binary systems
        i:      Brewer+2016
        ii:     Rice & Brewer 2020
        iii:    Valenti & Fischer 2005
        iv:     Montes+2018
        v:      Sousa+08
        vi:     Mann+2014 (for VLM dwarfs)
        vii:    Mann+2013
        viii:   Newton+14
        ix:     Other (TODO)
     2: Early-mid K dwarf (with directly measured chemistry)
        i:      Brewer+2016
        ii:     Rice & Brewer 2020
        iii:    Valenti & Fischer 2005
        iv:     Montes+2018
        v:      Sousa+08
        vi:     Luck+08
     3: late-K dwarfs and M-dwarfs
        i:      Mann+15
        ii:     Gaidos+14
        iii:    Rojas-Ayala+2012
        iv:     Other NIR relations (TODO)
        v:      Other (TODO)

    Parameters
    ----------
    star_info: Pandas Series
        Single row of our obs_info DataFrame corresponding to a single star.

    Returns
    -------
    feh_value, feh_sigma: float
        Adopted value and sigma for [Fe/H].
        
    feh_source: str
        Source publication code (e.g. VF05) for adopted [Fe/H].
    
    feh_nondefault: bool
        False if the adopted value comes from an empirical relation, True
        otherwise.
    """
    # -------------------------------------------------------------------------
    # [Fe/H] - binary
    # -------------------------------------------------------------------------
    # B+16 > RB20 > VF+05 > M18 > others
    if star_info["is_cpm"]:
        # Brewer+2016 --> follow-up from VF+05 on same scale with > precision
        if ~np.isnan(star_info["Fe_H_b16_prim"]):
            ref = "B16"
            feh_corr = star_info["Fe_H_b16_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = B16_ABUND_SIGMA["Fe_H"]

        # Rice & Brewer 2020 --> follow-up from VF+05 and B+16, but with Cannon
        elif ~np.isnan(star_info["Fe_H_rb20_prim"]):
            ref = "RB20"
            feh_corr = star_info["Fe_H_rb20_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = RB20_ABUND_SIGMA["Fe_H"]

        # Valenti & Fischer 2005 --> Mann+15 base [Fe/H] scale
        elif ~np.isnan(star_info["Fe_H_vf05_prim"]):
            ref = "VF05"
            feh_corr = star_info["Fe_H_vf05_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = VF05_ABUND_SIGMA["Fe_H"]

        # Montes+2018 --> large sample, but low precision on some stars
        elif ~np.isnan(star_info["Fe_H_m18_prim"]):
            ref = "M18"
            feh_corr = star_info["Fe_H_m18_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = star_info["eFe_H_m18_prim"]

        # Sousa+08 (this is the base sample for Adibekyan+12 abundances)
        elif ~np.isnan(star_info["feh_s08_prim"]):
            ref = "Sou08"
            feh_corr = star_info["feh_s08_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = star_info["e_feh_s08_prim"]

        # Mann+2014 for VLM dwarfs
        elif ~np.isnan(star_info["feh_m14_prim"]):
            ref = "M14"
            feh_corr = star_info["feh_m14_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = star_info["e_feh_m14_prim"]

        # -----------------------------------------------------------------
        # Note that after here things are a bit ~misc~

        # Mann+13
        elif ~np.isnan(star_info["feh_prim_m13_prim"]):
            ref = star_info["ref_feh_prim_m13_prim"]
            feh_corr = star_info["feh_prim_m13_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = star_info["e_feh_prim_m13_prim"]

        # Newton+14
        elif ~np.isnan(star_info["feh_prim_n14_prim"]):
            ref = star_info["feh_prim_ref_n14_prim"]
            feh_corr = star_info["feh_prim_n14_prim"] + FEH_OFFSETS["VF05"]
            e_feh_corr = star_info["e_feh_prim_n14_prim"]

        # Then check other entries
        elif star_info["feh_prim_ref_other_prim"] == "Ra07":
            ref = "Ra07"
            feh_corr = star_info["feh_prim_other_prim"] + FEH_OFFSETS[ref]
            e_feh_corr = star_info["e_feh_prim_other_prim"]
        
        # In case we're missing a value, alert
        else:
            print("Missing CPM value")

        # Take final value
        feh_value = feh_corr
        feh_sigma = e_feh_corr
        feh_source = ref
        feh_nondefault = True

    # -------------------------------------------------------------------------
    # [Fe/H] - single K-dwarfs
    # -------------------------------------------------------------------------
    # Brewer+2016 --> follow-up from VF+05 on same scale with > precision
    elif ~np.isnan(star_info["Fe_H_b16"]):
        feh_value = star_info["Fe_H_b16"] + FEH_OFFSETS["B16"]
        feh_sigma = B16_ABUND_SIGMA["Fe_H"]
        feh_source = "B16"
        feh_nondefault = True

    # Rice & Brewer 2020 --> follow-up from VF+05 and B+16, but with Cannon
    # Note: RB2020 reports params for M dwarfs, but these are *extremely*, so
    # we have to make sure to only take the useful (i.e. mid-K and up) stars.
    elif ~np.isnan(star_info["Fe_H_rb20"]) and star_info["useful_rb20"]:
        feh_value = star_info["Fe_H_rb20"] + FEH_OFFSETS["RB20"]
        feh_sigma = RB20_ABUND_SIGMA["Fe_H"]
        feh_source = "RB20"
        feh_nondefault = True

    # VF05 (K-dwarfs)
    elif ~np.isnan(star_info["Fe_H_vf05"]):      # TODO: feh_vf05 =/= Fe_H_vf05
        feh_value = star_info["Fe_H_vf05"]
        feh_sigma = VF05_ABUND_SIGMA["Fe_H"]
        feh_source = "VF05"
        feh_nondefault = True

    # Montes+2018 --> large sample, but low precision on some stars
    elif ~np.isnan(star_info["Fe_H_m18"]):
        feh_value = star_info["Fe_H_m18"] + FEH_OFFSETS["M18"]
        feh_sigma = star_info["eFe_H_m18"]
        feh_source = "M18"
        feh_nondefault = True

    # Sousa+08 (K-dwarfs)
    elif ~np.isnan(star_info["feh_s08"]):
        feh_value = star_info["feh_s08"]
        feh_sigma = star_info["e_feh_s08"]
        feh_source = "Sou08"
        feh_nondefault = True

    # Luck+2018 (K-dwarfs)
    elif not np.isnan(star_info["feh_L18"]):
        feh_value = star_info["feh_L18"]
        feh_sigma = np.nan    # TODO
        feh_source = "L18"
        feh_nondefault = True

    # -------------------------------------------------------------------------
    # [Fe/H] - single M-dwarfs
    # -------------------------------------------------------------------------
    # M+15 > G+14 > RA+12 > NIR other > Rains+21 > default

    # Mann+15
    elif not np.isnan(star_info["feh_m15"]):
        feh_value = star_info["feh_m15"]
        feh_sigma = star_info["e_feh_m15"]
        feh_source = "M15"
        feh_nondefault = True
    
    # Gaidos+14
    elif not np.isnan(star_info["feh_g14"]):
        feh_value = star_info["feh_g14"]
        feh_sigma = star_info["e_feh_g14"]  # TODO: check, but should be M15 scale
        feh_source = "G14"
        feh_nondefault = True

    # Rojas-Ayala+2012
    elif not np.isnan(star_info["feh_ra12"]):
        feh_value = \
            star_info["feh_ra12"] + FEH_OFFSETS["RA12"]
        feh_sigma = star_info["e_feh_ra12"]
        feh_source = "RA12"
        feh_nondefault = True

    # Other NIR relations
    elif not np.isnan(star_info["feh_nir"]):
        feh_value = star_info["feh_nir"]
        feh_sigma = star_info["e_feh_nir"]
        feh_source = star_info["nir_source"]
        feh_nondefault = True

    # TODO: eventually have the Cannon from Rains+24 predict [Fe/H]
    else:
        feh_value = np.nan
        feh_sigma = np.nan
        feh_source = ""
        feh_nondefault = False

    """
    # Rains+21 photometric [Fe/H]
    elif not np.isnan(star_info["phot_feh"]):
        feh_value = star_info["phot_feh"]
        feh_sigma = star_info["e_phot_feh"]
        feh_source = "R21"
        feh_nondefault = True

    # Solar neighbourhood default
    else:
        feh_value = -0.14 # Mean for Solar Neighbourhood
        feh_sigma = 2.0    # Default uncertainty
        feh_source = "R21"
        feh_nondefault = False
        #print("Assigned default [Fe/H] to {}".format(source_id))
    """

    return feh_value, feh_sigma, feh_source, feh_nondefault


def select_Ti_label(star_info, feh_adopted, feh_sigma_adopted):
    """Produces our adopted [Ti/H] and [Ti/Fe] abundances for this specific
    star.

    Our current Ti heirarchy is:
     1: Brewer+2016
     2. Rice & Brewer 2020
     3: Valenti & Fischer 2005 --> adopted reference scale
     4: Montes+2018
     5: Adibekyan+2012
     6: Empirical relation based on adopted [Fe/H]
      a: Chemo-kinematic relationship based in [Fe/H]-V_phi-[Ti/Fe] space built
         from GALAH DR2 and Gaia DR3 data.
      b: Simple linear fit in [Fe/H]-[Ti/Fe] space to GALAH DR2 data.
      c: Simple linear fit in [Fe/H]-[Ti/H] space to Montes+18 data. 

    Valenti & Fischer 2005 is the reference scale, and though we prioritise 
    Brewer+2016 and Rice & Brewer 2020 over it, they are a) follow-up work and
    thus nominally on the same scale, and b) have higher [X/Fe] precision.

    Parameters
    ----------
    star_info: Pandas Series
        Single row of our obs_info DataFrame corresponding to a single star.

    feh_adopted, feh_adopted: float
        Adopted values for [Fe/H] and sigma_[Fe/H].

    Returns
    -------
    Ti_dict: dict
        Dictionary with keys ["Ti_H_value_predicted", "Ti_H_sigma_predicted",
        "Ti_Fe_value_predicted", "Ti_Fe_sigma_predicted", "Ti_H_value_adopted", 
        "Ti_H_sigma_adopted", "Ti_Fe_value_adopted", "Ti_Fe_sigma_adopted",
        "Ti_source", "Ti_nondefault"] containing the adopted values for Ti.
    """
    # -------------------------------------------------------------------------
    # Calculate empirical values for [Ti/H] and [Ti/Fe]
    # -------------------------------------------------------------------------
    # [Ti/H] - Fitted to Montes+18 sample
    poly = Polynomial([0.02030864, 0.65459293])
    Ti_H_value_predicted = poly(feh_adopted)
    Ti_H_sigma_predicted = 2.0
    Ti_H_source = "R22c"

    # [Ti/Fe] from Monty GALAH + Gaia fits --> Preferred
    if not np.isnan(star_info["Ti_Fe_monty"]):
        Ti_Fe_value_predicted = star_info["Ti_Fe_monty"] + Ti_Fe_OFFSETS["Monty"]
        Ti_Fe_sigma_predicted = star_info["e_Ti_Fe_monty"]
        Ti_Fe_source = "R22a"

    # [Ti/Fe]--simple linear in [Fe/H] to GALAH DR2
    else:
        Ti_Fe_value_predicted = -0.332 * feh_adopted + 0.078
        Ti_Fe_sigma_predicted = 0.16
        Ti_Fe_source = "R22b"

    # -------------------------------------------------------------------------
    # [Ti/H]
    # -------------------------------------------------------------------------
    # Since B+20 / VF+05 / RB20 are our base [Fe/H] reference, we'll use them 
    # as our base reference for abundances as well. They also have lower 
    # uncertainties than other references. TODO: account for abundance systematics.
    if (not np.isnan(star_info["Ti_H_b16_prim"]) 
        or not np.isnan(star_info["Ti_H_b16"])):
        # Prefer chemistry from primary star
        if not np.isnan(star_info["Ti_H_b16_prim"]):
            col = "Ti_H_b16_prim"
        else:
            col = "Ti_H_b16"

        Ti_H_value = star_info[col] + TIH_OFFSETS["B16"]
        Ti_H_sigma = B16_ABUND_SIGMA["Ti_H"]
        Ti_source = "B16"
        Ti_nondefault = True

        lit_abund_assigned = True

    # RB 20
    # Note: RB2020 reports params for M dwarfs, but these are *extremely* wrong
    # so we also need to make a BP-RP cut here
    elif (not np.isnan(star_info["Ti_H_rb20_prim"])
        or not np.isnan(star_info["Ti_H_rb20"])):
        # Prefer chemistry from primary star
        if not np.isnan(star_info["Ti_H_rb20_prim"]):
            col = "Ti_H_rb20_prim"
        else:
            col = "Ti_H_rb20"

        Ti_H_value = star_info[col] + TIH_OFFSETS["RB20"]
        Ti_H_sigma = RB20_ABUND_SIGMA["Ti_H"]
        Ti_source = "RB20"
        Ti_nondefault = True

        lit_abund_assigned = True

    # VF+05
    elif (not np.isnan(star_info["Ti_H_vf05_prim"])
        or not np.isnan(star_info["Ti_H_vf05"])):
        # Prefer chemistry from primary star
        if not np.isnan(star_info["Ti_H_vf05_prim"]):
            col = "Ti_H_vf05_prim"
        else:
            col = "Ti_H_vf05"

        Ti_H_value = star_info[col] + TIH_OFFSETS["VF05"]
        Ti_H_sigma = VF05_ABUND_SIGMA["Ti_H"]
        Ti_source = "VF05"
        Ti_nondefault = True

        lit_abund_assigned = True

    # Montes+18
    elif (not np.isnan(star_info["Ti_H_m18_prim"])
        or not np.isnan(star_info["Ti_H_m18"])):
        # Prefer chemistry from primary star
        if not np.isnan(star_info["Ti_H_m18_prim"]):
            col = "Ti_H_m18_prim"
        else:
            col = "Ti_H_m18"

        Ti_H_value = star_info[col] + TIH_OFFSETS["M18"]
        Ti_H_sigma =  star_info["e{}".format(col)]
        Ti_source = "M18"
        Ti_nondefault = True

        lit_abund_assigned = True

    # Adibekyan+2012, TODO: compute [Ti/H] offset
    # Note that for precision reasons we'll use the Ti I abundance which is 
    # computed from more lines (even if it does suffer from more LTE effects)
    # than the Ti II abundances, and will take the value *corrected* for Teff
    # systematics.
    elif (not np.isnan(star_info["Fe_H_a12_prim"])
        or not np.isnan(star_info["Fe_H_a12"])):
        # Prefer chemistry from primary star
        if not np.isnan(star_info["Fe_H_a12_prim"]):
            col = "TiI_Hc_a12_prim"
            e_col = "e_TiI_H_a12_prim"
        else:
            col = "TiI_Hc_a12"
            e_col = "e_TiI_H_a12"

        Ti_H_value = star_info[col]       # Corrected Ti I abund
        Ti_H_sigma = star_info[e_col]

        Ti_source = "A12"
        Ti_nondefault = True

        lit_abund_assigned = True

    # Luck+18 --- Note: currently only for early-mid K dwarfs
    elif not np.isnan(star_info["Ti_H_L18"]):
        col = "Ti_H_L18"

        Ti_H_value = star_info[col] + TIH_OFFSETS["L18"]
        Ti_H_sigma =  L18_ABUND_SIGMA["Ti_H"]
        Ti_source = "L18"
        Ti_nondefault = True

        lit_abund_assigned = True

    # Otherwise assign [Ti/H] from our adopted empirical relation
    else:
        Ti_H_value = Ti_H_value_predicted
        Ti_H_sigma =  Ti_H_sigma_predicted
        Ti_source = Ti_H_source
        Ti_nondefault = False

        lit_abund_assigned = False

    # -------------------------------------------------------------------------
    # [Ti/Fe]
    # -------------------------------------------------------------------------
    # [TODO] In the event of literature values of [Ti/Fe]
    if False:
        pass

    # If we've assigned a literature value for [Ti/H], calculate [X/Fe] from it
    # and our adopted [Fe/H] value and propagate uncertainties
    elif lit_abund_assigned:
        # Grab labels
        fe_h_log10 = feh_adopted
        e_fe_h_log10 = feh_sigma_adopted

        x_h_log10 = Ti_H_value
        e_x_h_log10 = Ti_H_sigma

        # Unlog
        fe_h = 10**fe_h_log10
        e_fe_h = fe_h * np.log(10) * e_fe_h_log10

        x_h = 10**x_h_log10
        e_x_h = x_h * np.log(10) * e_x_h_log10

        # Calculate [X/Fe]
        x_fe = x_h / fe_h
        e_x_fe = x_fe * np.sqrt((e_x_h/x_h)**2+(e_fe_h/fe_h)**2)

        # Relog, save
        Ti_Fe_value = np.log10(x_fe)
        Ti_Fe_sigma = e_x_fe / (x_fe * np.log(10))

    # If we've *not* assigned a literature abundance, but we're
    # producing [X/Fe] values, calculate a default [X/Fe] value
    elif not lit_abund_assigned:
        Ti_Fe_value = Ti_Fe_value_predicted
        Ti_Fe_sigma =  Ti_Fe_sigma_predicted
        Ti_source = Ti_Fe_source
        Ti_nondefault = False

    # -------------------------------------------------------------------------
    # Prepare return dict and return
    # -------------------------------------------------------------------------
    # Prepare return dict
    Ti_dict = {
        "Ti_H_value_predicted":Ti_H_value_predicted, 
        "Ti_H_sigma_predicted":Ti_H_sigma_predicted,
        "Ti_Fe_value_predicted":Ti_Fe_value_predicted,
        "Ti_Fe_sigma_predicted":Ti_Fe_sigma_predicted,
        "Ti_H_value_adopted":Ti_H_value, 
        "Ti_H_sigma_adopted":Ti_H_sigma,
        "Ti_Fe_value_adopted":Ti_Fe_value,
        "Ti_Fe_sigma_adopted":Ti_Fe_sigma,
        "Ti_source":Ti_source,
        "Ti_nondefault":Ti_nondefault,
    }

    return Ti_dict


def compute_systematic(
    label_values_pred,
    label_sigma_stat_pred,
    label_values_lit,
    label_sigma_lit,
    label,):
    """Function to compute polynomial fit to residuals to compute an error
    weighted systematic. 
    
    https://stackoverflow.com/questions/22670057/
    linear-fitting-in-python-with-uncertainty-in-both-x-and-y-coordinates

    Note: nominally functional, but not well tested or implemented, so consider
    very TODO.
    """
    # Compute residual and residual uncertainty
    resid = label_values_lit - label_values_pred
    resid_std = np.full(label_values_pred.shape, np.std(resid))
    resid_sigma = \
        (label_sigma_stat_pred**2 + resid_std**2 + label_sigma_lit**2)**0.5

    # Fitting function
    def linear_fit(p,x):
        m, c = p
        y = m*x + c
        return y
    
    # Create a model for fitting
    linear_model = odr.Model(linear_fit)

    # Create a RealData object
    data = odr.RealData(
        x=label_values_lit,
        y=resid,
        sx=label_sigma_lit,
        sy=resid_sigma,
    )

    # Run regression
    odr_obj = odr.ODR(data, linear_model, beta0=[0,1])
    odr_out = odr_obj.run()

    # Compute systematic from line of best fit
    corr = linear_fit(odr_out.beta, label_values_lit)
    print(np.median(corr))

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.close("all")
    fig, ax = plt.subplots()
    ax.errorbar(
        x=label_values_lit, 
        y=resid,
        xerr=label_sigma_lit,
        yerr=resid_sigma,
        fmt=".",
        ecolor="k",
        alpha=0.8)
    
    # Plot line of best fit
    # Create line for plotting
    xx = np.linspace(np.min(label_values_lit), np.max(label_values_lit), 100)
    yy = linear_fit(odr_out.beta, xx)
    ax.plot(xx, yy, color="r")

    text = r"resid $ = {:0.2f}\times {:s} + {:0.2f}$".format(
        odr_out.beta[0], label, odr_out.beta[1])

    ax.text(
        x=0.50,
        y=0.90,
        s=text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,)

    return odr_out