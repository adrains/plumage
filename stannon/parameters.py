"""Functions related to selecting literature benchmark parameters to train the
Cannon on.
"""
import numpy as np
import scipy.odr as odr

# -----------------------------------------------------------------------------
# Label preparation for benchmark stars
# -----------------------------------------------------------------------------
def prepare_labels(
    obs_join,
    abund_order_k,
    abund_order_m,
    abund_order_binary,
    abundance_labels,
    synth_params_available=False,
    mid_K_BP_RP_bound=1.7,
    mid_K_MKs_bound=4.6,):
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
    VF05_ABUND_SIGMA, and ABUND_CITATIONS at the top of the file respectively.

    Parameters
    ----------
    obs_join: pandas DataFrame
        Pandas dataframe crossmatch containing observation information, Gaia
        2MASS, and benchmark information.

    abund_order_k, abund_order_m, abund_order_binary: str list
        Abund order of preference (highest priority to lowest) for K dwarfs,
        M dwarfs, and binaries respectively.

    abundance_labels: str list
        The abundance labels to prepare, e.g. ['Fe_H'] at the most basic.

    synth_params_available: boolean, default: False
        Used if we want to use the Teff scale from R21. TODO: remove, as the
        new combined photometric Teff scale is far superior and accessible to
        anything with BP-RP, M_Ks, and [Fe/H].

    mid_K_BP_RP_bound: float, default: 1.5
        Upper BP-RP bound beyond which we no longer consider direct 
        determination of [Fe/H] or [X/Fe] from high-R spectroscopy reliable.

    mid_K_MKs_bound: float
        Upper MKs bound beyond which we no longer consider direct determination
        of [Fe/H] or [X/Fe] from high-R spectroscopy reliable.

    Updated
    -------
    obs_join: DataFrame
        New columns added to DataFrame corresponding to adopted labels.
    """
    BASE_LABELS =  ["teff", "logg",]
    SUPPORTED_LABELS = BASE_LABELS + abundance_labels
    N_LABELS = len(SUPPORTED_LABELS)

    # Intialise mask
    has_base_label_set = np.full(len(obs_join), True)      # Teff, logg, [Fe/H]
    has_complete_label_set = np.full(len(obs_join), True)  # All labels

    # Initialise label vector
    label_values = np.full( (len(obs_join), N_LABELS), np.nan)
    label_sigmas = np.full( (len(obs_join), N_LABELS), np.nan)

    # Initialise record of label source/s
    label_sources = np.full( (len(obs_join), N_LABELS), "").astype(object)

    # Go through one star at a time and select labels
    for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
        # ---------------------------------------------------------------------
        # Teff
        # ---------------------------------------------------------------------
        teff_value, teff_sigma, teff_source = select_teff_label(
            star_info,
            synth_params_available=synth_params_available)

        label_values[star_i, 0] = teff_value
        label_sigmas[star_i, 0] = teff_sigma
        label_sources[star_i, 0] = teff_source

        # ---------------------------------------------------------------------
        # logg
        # ---------------------------------------------------------------------
        logg_value, logg_sigma, logg_source = select_logg_label(
            star_info, synth_params_available)

        label_values[star_i, 1] = logg_value
        label_sigmas[star_i, 1] = logg_sigma
        label_sources[star_i, 1] = logg_source

        # ---------------------------------------------------------------------
        # [Fe/H] and abundances
        # ---------------------------------------------------------------------
        for abund_i, abund in enumerate(abundance_labels):
            abund_value, abund_sigma, abund_source = select_abund_label(
                    star_info=star_info,
                    abund=abund,
                    mid_K_BP_RP_bound=mid_K_BP_RP_bound,
                    mid_K_MKs_bound=mid_K_MKs_bound,
                    abund_order_k=abund_order_k,
                    abund_order_m=abund_order_m,
                    abund_order_binary=abund_order_binary,)

            label_values[star_i, 2+abund_i] = abund_value
            label_sigmas[star_i, 2+abund_i] = abund_sigma
            label_sources[star_i, 2+abund_i] = abund_source

        # ---------------------------------------------------------------------
        # Final check
        # ---------------------------------------------------------------------
        if np.sum(np.isnan(label_values[star_i, :3])) > 0:
            has_base_label_set[star_i] = False

        if np.sum(np.isnan(label_values[star_i])) > 0:
            has_complete_label_set[star_i] = False

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

    # Add corresponding mask to the dataframe
    obs_join["has_base_label_set"] = has_base_label_set
    obs_join["has_complete_label_set"] = has_complete_label_set

    # Combine this mask with our quality cut mask to get the adopted benchmarks
    obs_join["is_cannon_benchmark"] = np.logical_and(
        obs_join["passed_quality_cuts"],
        obs_join["has_base_label_set"],)
    
    # All done!


def select_teff_label(star_info, synth_params_available):
    """Produces our adopted Teff values for this specific star.

    TODO: generalise similar to the abundance function to accept a list of
    potential reference/source suffixes.

    Parameters
    ----------
    star_info: Pandas Series
        Single row of our obs_info DataFrame corresponding to a single star.

    synth_params_available: boolean, default: False
        Used if we want to use the Teff scale from R21. TODO: remove, as the
        new combined photometric Teff scale is far superior and accessible to
        anything with BP-RP, M_Ks, and [Fe/H].

    Returns
    -------
    teff_value, teff_sigma: float
        Adopted value and sigma for Teff.
        
    teff_source: str
        Source publication code (e.g. R21) for adopted Teff.
    """
    # First preference: interferometric Teff
    if not np.isnan(star_info["teff_int"]):
        teff_value = star_info["teff_int"]
        teff_sigma = star_info["e_teff_int"]
        teff_source = star_info["int_source"]

    # Otherwise uniform Teff from Rains+21
    elif synth_params_available:
        # Sigma value to add in quadrature with the non-interferometric Teff
        # statistical uncertainties from Rains+21.
        e_teff_quad = 60

        teff_value = star_info["teff_synth"]
        teff_sigma = (
            star_info["e_teff_synth"]**2 + e_teff_quad**2)**0.5
        teff_source = "R21"

    # Teff from empirical relations. M dwarfs will have Teff from the Mann+2015
    # relations, and K dwarfs and up will have Teff from the Casagrande+2021
    # relations, but for the overlap sample (late-K dwarfs) that have both,
    # we will prioritise the Casagrande+2021 value.
    else:
        # Grab booleans for convenience/readability
        has_M15_teff = not np.isnan(star_info["teff_M15_BP_RP_feh"])
        has_C21_teff = not np.isnan(star_info["teff_C21_BP_RP_logg_feh"])

        # M-dwarfs
        if has_M15_teff and not has_C21_teff:
            teff_value = star_info["teff_M15_BP_RP_feh"]
            teff_sigma = star_info["e_teff_M15_BP_RP_feh"]
            teff_source = "M15er"

        # Late-K dwarfs
        elif has_M15_teff and has_C21_teff:
            teff_value = star_info["teff_C21_BP_RP_logg_feh"]
            teff_sigma = star_info["e_teff_C21_BP_RP_logg_feh"]
            teff_source = "C21"

        # Mid-K and warmer
        elif not has_M15_teff and has_C21_teff:
            teff_value = star_info["teff_C21_BP_RP_logg_feh"]
            teff_sigma = star_info["e_teff_C21_BP_RP_logg_feh"]
            teff_source = "C21"
        
        # Otherwise we don't have a Teff from either of the photometric
        # relations, which presently means the star does not have a [Fe/H]
        # since we're solely using the [Fe/H] sensitive relations.
        else:
            teff_value = np.nan
            teff_sigma = np.nan
            teff_source = ""

    return teff_value, teff_sigma, teff_source
            

def select_logg_label(star_info, synth_params_available):
    """Produces our adopted logg values for this specific star.

    TODO: generalise similar to the abundance function to accept a list of
    potential reference/source suffixes.

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
    """
    # logg from Rains+21 (+Rains+24 Cannon model)
    if synth_params_available:
        logg_value = star_info["logg_synth"]
        logg_sigma = star_info["e_logg_synth"]
        logg_source = "R21"

    # logg from Mann+2015 and Mann+2019 empirical relations
    else:
        logg_value = star_info["logg_m19"]
        logg_sigma = star_info["e_logg_m19"]
        logg_source = "M19"

    return logg_value, logg_sigma, logg_source


def select_abund_label(
    star_info,
    abund,
    mid_K_BP_RP_bound,
    mid_K_MKs_bound,
    abund_order_k,
    abund_order_m,
    abund_order_binary,):
    """Produces our adopted [Fe/H] or [X/Fe] values for this specific star
    based on our ranking of literature sources where we preferentially take
    sources earlier in the list. 

    Parameters
    ----------
    star_info: Pandas Series
        Single row of our obs_info DataFrame corresponding to a single star.

    abund: str
        The abundance labels to prepare, e.g. 'Fe_H'.
        
    mid_K_BP_RP_bound: float
        Upper BP-RP bound beyond which we no longer consider direct 
        determination of [Fe/H] or [X/Fe] from high-R spectroscopy reliable.

    mid_K_MKs_bound: float
        Upper MKs bound beyond which we no longer consider direct determination
        of [Fe/H] or [X/Fe] from high-R spectroscopy reliable.

    abund_order_k, abund_order_m, abund_order_binary: str list
        Abund order of preference (highest priority to lowest) for K dwarfs,
        M dwarfs, and binaries respectively.

    Returns
    -------
    abund_value, abund_sigma: float
        Adopted value and sigma for the selected abundance.
        
    abund_source: str
        Source publication code (e.g. 'VF05') for adopted abundance.
    """
    # First check that the star is below the mid-K BP-RP boundary, which
    # corresponds to us trusting *directly* measured [Fe/H] and [X/Fe] from
    # high-resolution spectra.
    if (star_info["K_mag_abs"] < mid_K_MKs_bound 
        and star_info["BP-RP_dr3"] < mid_K_BP_RP_bound):
        is_mid_K = True
    else:
        is_mid_K = False

    # Initialise [Fe/H] output values to defaults
    abund_value = np.nan
    abund_sigma = np.nan
    abund_source = ""

    #=========================================
    # [Fe/H] - binary
    #=========================================
    # Check all CPM references in order of ranking
    if star_info["is_cpm"]:
        for ref in abund_order_binary:
            # Construct the column name and check it exists
            abund_col = "{}_{}_prim".format(abund, ref)
            e_abund_col = "e_{}_{}_prim".format(abund, ref)

            if abund_col not in star_info.keys():
                continue

            if ~np.isnan(star_info[abund_col]):
                abund_source = ref
                abund_value = star_info[abund_col]
                abund_sigma = star_info[e_abund_col]
                
                # We've found our parameter, break out of the loop
                break
            
        # TODO: default?
        if abund_source == "":
            print("Missing CPM [{}] for {} ({})".format(
                abund.replace("_", "/"), 
                star_info["simbad_name"], 
                star_info.name))

    #=========================================
    # K dwarfs
    #=========================================
    # We have to enforce our colour cut on K dwarfs as we can't trust directly
    # determined abundances below a certain threshold.
    elif is_mid_K:
        for ref in abund_order_k:
            # Construct the column name and check it exists
            abund_col = "{}_{}".format(abund, ref)
            e_abund_col = "e_{}_{}".format(abund, ref)

            if abund_col not in star_info.keys():
                continue

            if ~np.isnan(star_info[abund_col]):
                abund_source = ref
                abund_value = star_info[abund_col]
                abund_sigma = star_info[e_abund_col]
                
                # We've found our parameter, break out of the loop
                break
            
        # TODO: default?
        if abund_source == "":
            pass

    #=========================================
    # M dwarfs
    #=========================================
    # Otherwise search our M-dwarf specific references, which largely have only
    # [Fe/H] or [M/H] from empirical relations.
    else:
        for ref in abund_order_m:
            # Construct the column name and check it exists
            abund_col = "{}_{}".format(abund, ref)
            e_abund_col = "e_{}_{}".format(abund, ref)

            if abund_col not in star_info.keys():
                continue

            if ~np.isnan(star_info[abund_col]):
                abund_source = ref
                abund_value = star_info[abund_col]
                abund_sigma = star_info[e_abund_col]
                
                # We've found our parameter, break out of the loop
                break
            
        # TODO: default?
        if abund_source == "":
            pass

    return abund_value, abund_sigma, abund_source


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


def compute_X_Fe(chem_df, species, samples,):
    """Computes [X/Fe] and propagates uncertainties from [X/H] and [Fe/H] for
    each requested literature source and adds the appropriate new columns to
    the DataFrame in-place.

    Parameters
    ----------
    chem_df: pandas DataFrame
        Standardised dataframe of literature abundances. Index is Gaia DR3 ID,
        and columns are BP-RP plus X_H_<ref> and e_X_H_<ref> for each species
        and reference, where <ref> is e.g. 'VF05'.

    species: str list
        List of species to compute [X/Fe] for, e.g. ['Na', 'Si', 'Ti', 'Fe'].

    samples: str list
        List of literature reference abbreviations, e.g. ['VF05', 'R07', 'A12].
    """
    # Don't run on Fe or M
    species = np.array(species)[~np.isin(species, ["M", "Fe"])]

    # If we don't have any species, no sense continuing
    if len(species) == 0:
        print("No species left to compute [X/Fe] for!")
        return

    for ref in samples:
        for xh in species:
            # Grab columns for this molecule
            Fe_H_col = "Fe_H_{}".format(ref)
            e_Fe_H_col = "e_Fe_H_{}".format(ref)

            X_H_col = "{}_H_{}".format(xh, ref)
            e_X_H_col = "e_{}_H_{}".format(xh, ref)

            # If we don't have an X_H column, skip
            if X_H_col not in chem_df.columns.values:
                continue

            # Grab labels
            Fe_H_log10 = chem_df[Fe_H_col].values
            e_Fe_H_log10 = chem_df[e_Fe_H_col].values

            X_H_log10 = chem_df[X_H_col].values
            e_X_H_log10 = chem_df[e_X_H_col].values

            # Unlog
            Fe_H = 10**Fe_H_log10
            e_Fe_H = Fe_H * np.log(10) * e_Fe_H_log10

            X_H = 10**X_H_log10
            e_X_H = X_H * np.log(10) * e_X_H_log10

            # Calculate [X/Fe]
            X_Fe = X_H / Fe_H
            e_X_Fe = X_Fe * np.sqrt((e_X_H/X_H)**2+(e_Fe_H/Fe_H)**2)

            # Relog
            X_Fe_value = np.log10(X_Fe)
            X_Fe_sigma = e_X_Fe / (X_Fe * np.log(10))

            # Insert into dataframe immediately after [X/H] and e_[X/H] columns
            X_Fe_col = "{}_Fe_{}".format(xh, ref)
            e_X_Fe_col = "e_{}_Fe_{}".format(xh, ref)

            col_i = chem_df.columns.get_loc(e_X_H_col)
            chem_df.insert(loc=col_i+1, column=X_Fe_col, value=X_Fe_value)
            chem_df.insert(loc=col_i+2, column=e_X_Fe_col, value=X_Fe_sigma)