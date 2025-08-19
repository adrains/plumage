"""Script to assess systematics and overlap between different literature
chemistry benchmarks. While we need a bespoke approach (to a limited extent) to
import each literature catalogue, we put measured abundances into a standard
format and single table which greatly simplifies working with chemical
benchmarks subsequently.

TODO: perform fit and correction on *unlogged* abundances and uncertainties.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import stannon.parameters as sp

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def correct_abundance_trends(
    chem_df,
    species_to_correct,
    comp_ref,
    references_to_compare,
    comp_ref_secondary=None,
    poly_order=4,
    outlier_dex=0.3,
    outer_points_to_drop=2,):
    """Function to put separate and *overlapping* chemical samples on the same
    scale by performing polynomial fits to the residuals between a literature
    and reference abundance samples. These trends are then corrected, and the
    abundances updated in-place in the DataFrame. The original uncorrected
    abundances are then stored as <abund>_<ref>_uc in the DataFrame, where 'uc'
    stands for 'uncorrected abundance'.

    TODO: perform the polynomial fit and correction taking into account the
    fact that abundances are logarithmic values. 

    Parameters
    ----------
    chem_df: pandas DataFrame
        Pandas DataFrame of literature abundances information. Has columns 
        BP-RP, and then <abund>_<ref>, and e_<abund>_<ref> for all literature
        sources and abundances.

    species_to_correct: str list
        List of species to correct for systematics, e.g. ['Fe_H', 'Ti_H'].

    comp_ref: str
        The reference abundance scale to correct to, e.g. 'VF05'.

    references_to_compare: str list
        List of literature sources to correct to our reference abundance scale.

    comp_ref_secondary: str, default: None
        A secondary reference abundance scale to adopt in the case our first
        preference in comp_ref is not available. 

    poly_order: int, default: 4
        Polynomial order for the correction.

    outlier_dex: default: 0.3
        Residuals beyond +/- outlier_dex are excluded when fitting the 
        polynomial in order to prevent outliers affecting the fit.

    outer_points_to_drop: int, default 2
        In order to prevent potentially sparsely sampled edge points from 
        influencing the fit, we drop the first and last outer_points_to_drop
        points when fitting.

    Returns
    -------
    fit_dict: dict
        Dictionary containing <ref>:(<abund>, Polynomial Obj) pairs for each
        corrected reference/abundance combination.
    """
    # Create dict for storing polynomials for use when plotting later
    fit_dict = {}

    # Loop over all comparisons
    for ref_i, ref in enumerate(references_to_compare):
        for species in species_to_correct:
            # Grab value + sigma column names
            cols = chem_df.columns.values

            # Use primary reference scale where we can
            if "{}_{}".format(species, comp_ref) in cols:
                comp_ref_adopt = comp_ref

            # Check the secondary reference if it was provided
            elif ("{}_{}".format(species, comp_ref_secondary) in cols
                  and comp_ref_secondary is not None):
                comp_ref_adopt = comp_ref_secondary
                
            # Otherwise we can't make this comparison, so just continue
            else:
                continue

            # Check that ref =/= comp_ref_adopt. This can happen if we end up
            # using the secondary reference
            if ref == comp_ref_adopt:
                continue
            
            # Initial checks passed, construct our column names for reference
            abund_ref = "{}_{}".format(species, comp_ref_adopt)
            e_abund_ref = "e_{}_{}".format(species, comp_ref_adopt)

            # Grab the comparison values
            abund_comp = "{}_{}".format(species, ref)
            e_abund_comp = "e_{}_{}".format(species, ref)

            # Continue if we don't have this species to correct
            if abund_comp not in chem_df.columns.values:
                continue

            #=========================================
            # Fit residuals with polynomial
            #=========================================
            n_abund_before = np.sum(~np.isnan(chem_df[abund_comp].values))

            # Compute the residuals and the combined uncertainties
            resid = chem_df[abund_ref].values - chem_df[abund_comp].values
            e_resid = np.sqrt(chem_df[e_abund_ref].values**2
                              + chem_df[e_abund_comp].values**2)
            
            # Perform polynomial fitting
            n_overlap = np.sum(~np.isnan(resid))
            resid_mask = ~np.isnan(resid)
            fit_mask = np.logical_and(resid_mask, np.abs(resid) < outlier_dex)

            bp_rp = chem_df["BP_RP"].values
            
            poly = np.polynomial.Polynomial.fit(
                bp_rp[fit_mask], resid[fit_mask], poly_order)
            
            #=========================================
            # Correct systematics below/within/above bounds
            #=========================================
            # The polynomial correction is only valid within the BP-RP bounds
            # of the overlapping sample. However, for our correction we'll
            # evaluate it several points in from each end to prevent outliers
            # from having too much of an influence.
            bp_rp_sorted = bp_rp[resid_mask].copy()
            bp_rp_sorted.sort()

            i_min = outer_points_to_drop
            i_max = len(bp_rp_sorted) - outer_points_to_drop

            min_BP_RP = bp_rp_sorted[i_min]
            max_BP_RP = bp_rp_sorted[i_max]
            
            # Store bounds in polynomial object for plotting later
            poly.BP_RP_bounds = (min_BP_RP, max_BP_RP)

            # Work out beyond bounds
            abund = chem_df[abund_comp].values

            is_below = bp_rp < min_BP_RP
            within_bounds = np.logical_and(
                bp_rp >= min_BP_RP, bp_rp <= max_BP_RP)
            is_above = bp_rp > max_BP_RP

            has_data = ~np.isnan(abund)
            n_below = np.sum(np.logical_and(has_data, is_below))
            n_above = np.sum(np.logical_and(has_data, is_above))
            poly.n_beyond_bounds = (n_below, n_above)

            # Correct for the fit (below, overlapping, & above BP-RP bounds)
            abund_corr = np.full_like(abund, np.nan)
            abund_corr[is_below] = abund[is_below] + poly(min_BP_RP)
            abund_corr[within_bounds] = \
                abund[within_bounds] + poly(bp_rp[within_bounds])
            abund_corr[is_above] = abund[is_above] + poly(max_BP_RP)

            n_abund_after = np.sum(~np.isnan(abund_corr))

            # Store uncorrected abundances, and save corrected abundances
            chem_df["{}_uc".format(abund_comp)] = abund.copy()
            chem_df[abund_comp] = abund_corr

            # TODO: recalculate residuals properly considering logarithms 
            pass

            # Store polynomial
            fit_dict[(ref, species)] = poly
        
        print(ref, n_abund_before, n_abund_after)

    return fit_dict


def plot_abundance_trends(
    chem_df,
    fit_dict,
    species_to_correct,
    comp_ref,
    references_to_compare,
    bp_rp_lims,
    abund_Y_lims,
    comp_ref_secondary=None,
    do_limit_y_extent=True,
    fn_label="",
    bp_rp_ticks=(0.1,0.05),
    figsize=(16, 10),):
    """Function to plot before and after correcting for abundance trends and
    putting all samples on the same reference abundance scale. For every 
    species, we plot two panels per reference showing the residuals fit with a
    polynomial trend, and then the corrected residuals.

    Parameters
    ----------
    chem_df: pandas DataFrame
        Pandas DataFrame of literature abundances information. Has columns 
        BP-RP, and then <abund>_<ref>, and e_<abund>_<ref> for all literature
        sources and abundances.

    fit_dict: dict
        Dictionary containing <ref>:(<abund>, Polynomial Obj) pairs for each
        corrected reference/abundance combination.

    species_to_correct: str list
        List of species to correct for systematics, e.g. ['Fe_H', 'Ti_H'].

    comp_ref: str
        The reference abundance scale to correct to, e.g. 'VF05'.

    references_to_compare: str list
        List of literature sources to correct to our reference abundance scale.

    bp_rp_lims: float list
        Two element list of the minimum and maximum BP-RP limits to plot, of
        form [min, max] 

    abund_Y_lims: float
        Symmetric limits for the Y abundance axis.

    comp_ref_secondary: str, default: None
        A secondary reference abundance scale to adopt in the case our first
        preference in comp_ref is not available. 

    do_limit_y_extent: boolean, default: True
        Whether or not to limit the Y extent plotted.

    fn_label: str, default: ''
        Plots by default are saved as paper/chemical_trends_<abund>.<pdf/png>
        but this changes it to paper/chemical_trends_<label>_<abund>.<pdf/png>.

    bp_rp_ticks: float array, default: (0.1, 0.05)
        Major and minor x tick values for BP-RP.

    figsize: float tuple, default: (16, 10)
        Size of the plotted figure for each species.
    """
    # Loop over all species
    for species in species_to_correct:
        # First we need to count the number of references that actually have
        # this species so that we can initialise the plot.
        comp_mask = np.array(["{}_{}".format(species, cr) in chem_df.columns 
            for cr in references_to_compare])

        # If our primary reference does not have this species, we need to check
        # the secondary reference and make sure to remove it from being
        # compared with itself. We can then plot one fewer panel.
        if ("{}_{}".format(species, comp_ref) not in chem_df.columns
            and comp_ref_secondary is not None):
            comp_ref_2nd_i = int(np.where(
                references_to_compare_K == comp_ref_secondary)[0])
            comp_mask[comp_ref_2nd_i] = False

        plt.close("all")
        fig, axes = plt.subplots(
            nrows=np.sum(comp_mask),
            ncols=2,
            sharex=True,
            sharey="row",
            figsize=figsize)

        # In the event we only have one comparison (e.g. in the chemodynamic
        # case) ensure the axes array is 2D for consistent indexing.
        if len(axes.shape) == 1:
            axes = np.atleast_2d(axes)
        
        fig.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.98,
            top=0.97,
            hspace=0.01,
            wspace=0.01)

        # Loop over all valid comparisons
        for ref_i, ref in enumerate(references_to_compare[comp_mask]):
            # Grab value + sigma column names
            cols = chem_df.columns.values

            # Use primary reference scale where we can
            if "{}_{}".format(species, comp_ref) in cols:
                comp_ref_adopt = comp_ref

            # Check the secondary reference if it was provided
            elif ("{}_{}".format(species, comp_ref_secondary) in cols
                  and comp_ref_secondary is not None):
                comp_ref_adopt = comp_ref_secondary

            # Shouldn't be able to get here?
            else:
                raise Exception("Something has gone wong?")

            # Grab value + sigma column names
            abund_ref = "{}_{}".format(species, comp_ref_adopt)
            e_abund_ref = "e_{}_{}".format(species, comp_ref_adopt)

            abund_comp = "{}_{}".format(species, ref)
            e_abund_comp = "e_{}_{}".format(species, ref)

            #=========================================
            # Compute residuals
            #=========================================
            # Compute the residuals and the combined uncertainties
            # TODO: use abundances uncertainties corrected in log-space
            abund_comp_uc = "{}_uc".format(abund_comp)
            #e_abund_comp_uc = "e_{}_uc".format(abund_comp)

            resid = chem_df[abund_ref].values - chem_df[abund_comp_uc].values
            e_resid = np.sqrt(chem_df[e_abund_ref].values**2
                              + chem_df[e_abund_comp].values**2)
            
            # Compute the corrected residuals
            resid_corr = \
                chem_df[abund_ref].values - chem_df[abund_comp].values

            # Compute statistics before and after
            med = np.nanmedian(resid)
            std = np.nanstd(resid)

            med_corr = np.nanmedian(resid_corr)
            std_corr = np.nanstd(resid_corr)

            n_overlap = np.sum(~np.isnan(resid_corr))

            #=========================================
            # Left Hand Panels: raw residuals + fit
            #=========================================
            # Plot polynomial correction within bounds
            poly = fit_dict[(ref, species)]
            xx = np.linspace(poly.BP_RP_bounds[0], poly.BP_RP_bounds[1], 100)
            axes[ref_i, 0].plot(
                xx, poly(xx), color="r", linewidth=1.0, zorder=10)

            # Plot and annotate correction *below* polynomial bounds
            axes[ref_i, 0].hlines(
                y=poly(poly.BP_RP_bounds[0]),
                xmin=bp_rp_lims[0],
                xmax=poly.BP_RP_bounds[0],
                color="r",
                linewidth=1.0,
                linestyle="--",)

            axes[ref_i, 0].text(
                x=0.025,
                y=0.925,
                s=r"$\leftarrow {:0.0f}$".format(poly.n_beyond_bounds[0]),
                horizontalalignment="center",
                verticalalignment="center",
                color="r",
                transform=axes[ref_i, 0].transAxes,)
            
            # Plot and annotate correction *above* polynomial bounds
            axes[ref_i, 0].hlines(
                y=poly(poly.BP_RP_bounds[1]),
                xmin=poly.BP_RP_bounds[1],
                xmax=bp_rp_lims[1],
                color="r",
                linewidth=1.0,
                linestyle="--",)
            
            axes[ref_i, 0].text(
                x=0.975,
                y=0.925,
                s=r"${:0.0f} \rightarrow$".format(poly.n_beyond_bounds[1]),
                horizontalalignment="center",
                verticalalignment="center",
                color="r",
                transform=axes[ref_i, 0].transAxes,)
            
            # Plot the residuals
            axes[ref_i, 0].errorbar(
                x=chem_df["BP_RP"].values,
                y=resid,
                yerr=e_resid,
                fmt="o",
                alpha=0.8,
                ecolor="k",
                label=r"{}$ - ${} (N={})".format(
                    comp_ref_adopt, ref, n_overlap))
            
            axes[ref_i, 0].legend(loc="lower right")
            
            axes[ref_i, 0].set_ylabel(
                r"$\Delta$[{}]".format(species.replace("_", "/")))

            # Plot resid=0 line
            axes[ref_i, 0].hlines(
                y=0,
                xmin=bp_rp_lims[0],
                xmax=bp_rp_lims[1],
                linestyles="--",
                colors="k",
                linewidth=0.5,)
            
            # Annotate statistics
            txt = r"${:0.2f} \pm {:0.2f}\,$dex".format(med, std)
            txt = txt.replace("-0.00", "0.00")

            axes[ref_i, 0].text(
                x=0.5,
                y=0.25,
                s=txt,
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[ref_i, 0].transAxes,
                bbox=dict(facecolor="grey", edgecolor="None", alpha=0.5),)
            
            # Display polynomial coefficients
            coefs = poly.coef[::-1]
            exponents = np.arange(3, 0, -1)
            ft = r"{:0.3}\cdot(BP-RP)^{:0.0f}"

            fit_list = \
                [ft.format(cc, ee) for (cc, ee) in zip(coefs, exponents)]
            fit_list.append("{:0.3f}".format(coefs[-1]))
            fit_txt = r"${}$".format("+".join(fit_list).replace("+-", "-"))

            # Also display BP-RP limits
            lims = "\t$[{:0.2f}, {:0.2f}]$".format(*poly.BP_RP_bounds)

            axes[ref_i, 0].text(
                x=0.01,
                y=0.04,
                s=fit_txt + lims,
                horizontalalignment="left",
                verticalalignment="center",
                color="r",
                fontsize="x-small",
                transform=axes[ref_i, 0].transAxes,)

            #=========================================
            # Right Hand Panels: corrected residuals
            #=========================================
            axes[ref_i, 1].errorbar(
                x=chem_df["BP_RP"].values,
                y=resid_corr,
                yerr=e_resid,
                fmt="o",
                alpha=0.8,
                ecolor="k",
                label=r"{}$ - ${} (N={})".format(
                    comp_ref_adopt, ref, n_overlap))
            
            axes[ref_i, 1].legend(loc="lower right")

            axes[ref_i, 1].hlines(
                y=0,
                xmin=bp_rp_lims[0],
                xmax=bp_rp_lims[1],
                linestyles="--",
                colors="k",
                linewidth=0.5,)
            
            txt = r"${:0.2f} \pm {:0.2f}\,$dex".format(med_corr, std_corr)
            txt = txt.replace("-0.00", "0.00")

            axes[ref_i, 1].text(
                x=0.5,
                y=0.2,
                s=txt,
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[ref_i, 1].transAxes,
                bbox=dict(facecolor="grey", edgecolor="None", alpha=0.5),)
            
            if do_limit_y_extent:
                axes[ref_i, 1].set_ylim(abund_Y_lims[0], abund_Y_lims[1])

            axes[ref_i,0].yaxis.set_major_locator(
                plticker.MultipleLocator(base=0.2))
            axes[ref_i,0].yaxis.set_minor_locator(
                plticker.MultipleLocator(base=0.1))

        axes[0,0].set_title("Best-fit Residuals")
        axes[0,1].set_title("Corrected Residuals")

        axes[ref_i, 0].set_xlim(bp_rp_lims[0], bp_rp_lims[1])
        axes[ref_i, 0].set_xlabel(r"$BP-RP$")
        axes[ref_i, 1].set_xlabel(r"$BP-RP$")

        axes[ref_i,0].xaxis.set_major_locator(
            plticker.MultipleLocator(base=bp_rp_ticks[0]))
        axes[ref_i,0].xaxis.set_minor_locator(
            plticker.MultipleLocator(base=bp_rp_ticks[1]))

        lbl = "" if fn_label == "" else "{}_".format(fn_label)

        plt.savefig("paper/chemical_trends_{}{}.pdf".format(lbl, species))
        plt.savefig(
            "paper/chemical_trends_{}{}.png".format(lbl, species),
            dpi=200)


def plot_chemodynamic_one_to_one_recovery(
    chem_df,
    species_to_plot,
    comp_refs,
    figsize=(16,10),):
    """Function to plot 1:1 recovery of a literature [X/Fe] sample with the
    chemodynamic equivalent values.
    """
    plt.close("all")
    fig, axes = plt.subplots(
        nrows=len(species_to_plot),
        ncols=1,
        figsize=figsize)
    
    fig.subplots_adjust(
        wspace=0.1, hspace=0.2, right=0.98, left=0.125, top=0.975, bottom=0.05)
    
    for sp_i, (species, comp_ref) in enumerate(zip(species_to_plot, comp_refs)):
        # Grab reference data
        X_Fe_ref = chem_df["{}_{}".format(species, comp_ref)].values
        e_X_Fe_ref = chem_df["e_{}_{}".format(species, comp_ref)].values
        Fe_H_ref = chem_df["Fe_H_{}".format(comp_ref)].values

        # Grab chemodynamic data
        X_Fe_CD = chem_df["{}_SM25".format(species)].values
        e_X_Fe_CD = chem_df["e_{}_SM25".format(species)].values

        # Plot error bars with overplotted scatter points + colour bar
        axes[sp_i].errorbar(
            X_Fe_ref, 
            X_Fe_CD, 
            xerr=e_X_Fe_ref,
            yerr=e_X_Fe_CD,
            fmt=".",
            zorder=0,
            ecolor="black",
            #markersize=ms,
            elinewidth=0.2,)

        sc = axes[sp_i].scatter(
            X_Fe_ref, X_Fe_CD, c=Fe_H_ref, zorder=1,)

        cb = fig.colorbar(sc, ax=axes[sp_i])
        cb.ax.tick_params(labelsize="medium", rotation=0)
        cb.ax.set_title("[Fe/H]", fontsize="medium")
        
        # Plot 1:1 line
        all_X_Fe = np.concatenate([X_Fe_ref, X_Fe_CD])
        lim_min = np.nanmin(all_X_Fe)
        lim_max = np.nanmax(all_X_Fe)

        # Plot 1:1 line
        xx = np.arange(lim_min, lim_max, (lim_max-lim_min)/100)
        axes[sp_i].plot(xx, xx, "k--", zorder=0)

        # Other setup
        species_str = "[{}]".format(species.replace("_", "/"))
        axes[sp_i].set_xlabel("{} ({})".format(species_str, comp_ref))
        axes[sp_i].set_ylabel("{} (Chemodynamic)".format(species_str))

        # Compute residuals
        resid = X_Fe_ref - X_Fe_CD

        # Compute statistics before and after
        med = np.nanmedian(resid)
        std = np.nanstd(resid)

        # Annotate residuals
        txt = r"${:0.2f} \pm {:0.2f}\,$dex".format(med, std)
        txt = txt.replace("-0.00", "0.00")
    
        axes[sp_i].text(
            x=0.5,
            y=0.1,
            s=txt,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[sp_i].transAxes,
            bbox=dict(facecolor="grey", edgecolor="None", alpha=0.5),)


#------------------------------------------------------------------------------
# Literature samples + info
#------------------------------------------------------------------------------
# Mapping of samples to TSV files with a complete Gaia DR3 crossmatch
samples = {
    "VF05":"data/VF05.tsv",             # Const sigma for: all
    "R07":"data/R07_dr3_all.tsv",       # Individual sigmas
    "A12":"data/A12_gaia_all.tsv",      # Const sigma for: Fe
    "RA12":"data/RA12_dr3_all.tsv",     # Individual sigmas
    "M13":"data/M13_prim_dr3.tsv",      # Individual sigmas
    "G14a":"data/G14a_gaia_all.tsv",    # Individual sigmas
    "G14b":"data/G14b_gaia_all.tsv",    # Individual sigmas
    "M14":"data/M14_all.tsv",           # Individual sigmas
    "M15":"data/mann15_all_dr3.tsv",    # Individual sigmas
    "B16":"data/B16_dr3_all.tsv",       # Const sigma for: all
    "M18":"data/montes18_prim.tsv",     # Individual sigmas
    "L18":"data/luck18_all_dr3.tsv",    # Individual sigmas
    "D19":"data/D19_gaia_all.tsv",      # Individual sigmas
    "RB20":"data/RB20_dr3_all.tsv",     # Const sigma for: all
    "M20":"data/M20_gaia_all.tsv",      # Const sigma for: all
    "SM25":"data/SM25_X_Fe_chemodynamic_Ti_Ca_Na_Al_Mg.tsv",# Invidivual sigmas
}

# Mapping of chemical species within each sample
species_all = {
    "VF05":["M", "Na", "Si", "Ti", "Fe", "Ni"],
    # Note: R07 report FeI, FeII, and Fe, and O for LTE and NLTE
    "R07":["Fe", "O"],  
    "A12":["Na", "Mg", "Al", "Si", "Ca", "Ti", "Cr", "Ni", "Mn", "Fe", "Co", 
           "Sc", "Mn", "V"],
    "RA12":["M", "Fe"],
    "M13":["M", "Na", "Si", "Ti", "Fe", "Ni"],
    "M14":["Fe"],
    "G14a":["Fe"],
    "G14b":["Fe"],
    "M15":["Fe"],
    "B16":["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "V", "Cr", "Mn",
           "Fe", "Ni", "Y"],
    "M18":["Fe", "Na", "Mg", "Al", "Si", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
           "Co", "Ni"],
    # Note: ignoring Eu since it lacks an uncertainty
    "L18":["Na", "Mg", "Al", "Si", "S", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
           "Fe", "Co", "Ni", "Cu", "Zn", "Sr", "Y", "Zr", "Ba", "La", "Ce",
           "Nd", "Sm",], # "Eu",]
    "D19":["Fe", "M",],
    "RB20":["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "V", "Cr", "Mn",
            "Fe", "Ni", "Y"],
    "M20":["Fe",],# "C", "Na", "Mg", "Al", "Si", "Ca", "Sc", "Ti", "V", "Cr",
           #"Mn", "Co", "Ni", "Zn",],
    "SM25":["Na", "Mg", "Al", "Ca", "Ti"],
}

# Mapping of abundance uncertainties for those samples with constant adopted
# sigma for each [X/H].
sigmas = {
    "VF05":{"M":0.029,
            "Na":0.032,
            "Si":0.019,
            "Ti":0.046,
            "Fe":0.03,
            "Ni":0.03,},
    "A12":{"Fe":0.03,},     # From paper, pg. 2, section 2.
    "B16":{"C":0.026,
            "N":0.042,
            "O":0.036,
            "Na":0.014,
            "Mg":0.012,
            "Al":0.028,
            "Si":0.008,
            "Ca":0.014,
            "Ti":0.012,
            "V":0.034,
            "Cr":0.014,
            "Mn":0.020,
            "Fe":0.010,
            "Ni":0.012,
            "Y":0.03,},
                          # limits
    "RB20":{"C":0.05,     # −0.60–0.64
            "N":0.09,     # −0.86–0.84
            "O":0.07,     # −0.36–0.77
            "Na":0.06,    # −1.09–0.78
            "Mg":0.03,    # −0.70–0.54
            "Al":0.05,    # −0.66–0.58
            "Si":0.03,    # −0.65–0.57
            "Ca":0.03,    # −0.73–0.54
            "Ti":0.03,    # −0.71–0.52
            "V":0.04,     # −0.85–0.46
            "Cr":0.03,    # −1.07–0.52
            "Mn":0.05,    # −1.40–0.66
            "Fe":0.02,    # −0.99–0.57
            "Ni":0.03,    # −0.97–0.63
            "Y":0.07,},    # −0.87–1.35
    # From Table 3, standard dev of difference between PCA abund vs FGK primary
    "M20":{"Fe":0.04,
           "C":0.03,
           "Na":0.05,
           "Mg":0.12,
           "Al":0.05,
           "Si":0.07,
           "Ca":0.12,
           "Sc":0.05,   # ScII
           "Ti":0.06,   # TiI
           "V":0.12,
           "Cr":0.04,   # CrI
           "Mn":0.07,
           "Co":0.02,
           "Ni":0.05,
           "Zn":0.09,}
}

# If this is true, then we don't accept stars from the non-M dwarf specific
# catalogues beyond this BP-RP cut. Specifically this is to address the M 
# dwarfs in RB20 that have been run through an FGK Cannon model, but more
# generally it excludes any high-resolution studies where we assume the optical
# is no longer a suitable source of direct parameter measurement via EW or
# spectral synthesis-based analyses.
ENFORCE_K_DWARF_BP_RP_COLOUR_CUT = False
K_DWARF_BP_RP_MAX = 1.7
cool_dwarf_catalogues = \
    ["RA12", "M14", "G14a", "G14b", "M15", "D19", "M20", "SM25"]

# Set to true to drop columns related to abundances we're not correcting for
# systematics. This prevents bloat in the number of columns, and is important
# later when saving to a fits file which only support 1000 columns.
remove_unused_X_H = True

#------------------------------------------------------------------------------
# Initial Import + Unique IDs
#------------------------------------------------------------------------------
all_ids = []
dataframes = {}

for ref in samples.keys():
    df = pd.read_csv(
        samples[ref],
        delimiter="\t",
        dtype={"source_id":str, "source_id_dr3":str},)
    df.rename(columns={"source_id":"source_id_dr3"}, inplace=True)
    df.set_index("source_id_dr3", inplace=True)

    # [Optional] Exclude stars beyond BP-RP cut from non-M dwarf catalogues
    if ENFORCE_K_DWARF_BP_RP_COLOUR_CUT and ref not in cool_dwarf_catalogues:
        within_bounds = df["bp_rp"] < K_DWARF_BP_RP_MAX
        df = df[within_bounds]

    all_ids += df.index.values.tolist()
    dataframes[ref] = df

# Collect unique IDs, remove nans and '-' values
unique_ids = set(all_ids)
unique_ids.remove("-")
unique_ids.remove(np.nan)

# Create new DataFrame to be used for crossmatching to later
df_comb = pd.DataFrame(data=unique_ids, columns=["source_id_dr3"],)
df_comb.set_index("source_id_dr3", inplace=True)

#------------------------------------------------------------------------------
# Collate columns of interest
#------------------------------------------------------------------------------
# Here we are going to put all [X/H] columns in a standard format for each
# literature sample, as well as selecting BP-RP for use when correcting for
# systematics.
dataframes_cut = {}

# We assyme that Any entries with no Gaia crossmatch have one of the following
# values for source_id and are accordingly dropped before proceeding.
default_ids = ["", "-", np.nan]

#=========================================
# Valenti & Fischer 2005 - 2005ApJS..159..141V
#=========================================
VF05 = dataframes["VF05"]
has_sid = np.array([sid not in default_ids for sid in VF05.index.values])
VF05 = VF05[has_sid].copy()
n_VF05 = len(VF05)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["VF05"]]
abund_cols_new = ["{}_H_VF05".format(ss) for ss in species_all["VF05"]]

VF05.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["VF05"]:
    e_species = np.full(n_VF05, sigmas["VF05"][species])
    sigma_col_new = "e_{}_H_VF05".format(species)
    VF05[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["VF05"] = VF05[cols].copy()

#=========================================
# Ramírez+2007 - 2007A&A...465..271R
#=========================================
R07 = dataframes["R07"]
has_sid = np.array([sid not in default_ids for sid in R07.index.values])
R07 = R07[has_sid].copy()

# Ramirez+07 only reports a combined [Fe/H] abundance where Teff > 5400,
# meaning that stars cooler than this only have [FeI/H] and [FeII/H]. Their
# combined value involves a systematic correction of FeI, then a average
# weighted by the indidual line scatter. They state that the FeI abundances
# have a lower scatter and are more internally consistent, but are more
# sensitive to adopted atmospheric parameters versus FeII. Given this, to
# maximise the number of stars with parameters in the crossmatch, we'll take
# the author's [Fe/H] values where possible, and will adopt [Fe/H] = [FeII/H]
# otherwise (which will come with a higher uncertainty).
Fe_H_adopted = np.full_like(R07["[Fe/H]"].values, np.nan)
e_Fe_H_adopted = np.full_like(R07["e_[Fe/H]"].values, np.nan)
has_Fe_H_comb = ~np.isnan(R07["[Fe/H]"].values)

Fe_H_adopted[has_Fe_H_comb] = R07["[Fe/H]"].values[has_Fe_H_comb]
e_Fe_H_adopted[has_Fe_H_comb] = R07["e_[Fe/H]"].values[has_Fe_H_comb]

Fe_H_adopted[~has_Fe_H_comb] = R07["[Fe/H]2"].values[~has_Fe_H_comb]
e_Fe_H_adopted[~has_Fe_H_comb] = R07["e_[Fe/H]2"].values[~has_Fe_H_comb]

R07["Fe_H_R07"] = Fe_H_adopted
R07["e_Fe_H_R07"] = e_Fe_H_adopted

# Take the NLTE Oxygen abundances
R07.rename(columns={"[O/H]N":"O_H_R07", "e_[O/H]N":"e_O_H_R07"}, inplace=True)

abund_cols_new = ["{}_H_R07".format(ss) for ss in species_all["R07"]]
sigma_cols_new = ["e_{}_H_R07".format(ss) for ss in species_all["R07"]]

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["R07"] = R07[cols].copy()

#=========================================
# Adibekyan+2012 - 2012A&A...545A..32A
#=========================================
A12 = dataframes["A12"]
has_sid = np.array([sid not in default_ids for sid in A12.index.values])
A12 = A12[has_sid].copy()

# Some columns have been corrected for a temperature systematic (see S3.2). 
# Swap the column names such that the *corrected* columns match the standard
# naming format, and the uncorrected columns have a subscript 'uc'.
species_to_correct = ["Na", "Al", "ScI", "TiI", "V", "CrII"]

for species in species_to_correct:
    col_base = "[{}/H]".format(species)
    col_corr = "[{}/H]c".format(species)
    col_uncorr = "[{}/H]uc".format(species)
    A12.rename(columns={col_base:col_uncorr}, inplace=True)
    A12.rename(columns={col_corr:col_base}, inplace=True)

# We need to combine the ionisation states and uncertainties for some species
species_to_combine = ["Sc", "Ti", "Cr"]

for species in species_to_combine:
    n_X_I = A12["o_[{}I/H]".format(species)].values
    X_I_log10 = A12["[{}I/H]".format(species)].values
    e_X_I_log10 = A12["e_[{}I/H]".format(species)].values

    X_I = 10**X_I_log10
    e_X_I = X_I * np.log(10) * e_X_I_log10
    
    n_X_II = A12["o_[{}II/H]".format(species)].values
    X_II_log10 = A12["[{}II/H]".format(species)].values
    e_X_II_log10 = A12["e_[{}II/H]".format(species)].values

    X_II = 10**X_II_log10
    e_X_II = X_II * np.log(10) * e_X_II_log10

    # Compute combined abundance and propagate uncertainty
    #   A = ((n1 * A1) + (n2 * A2)) / (n1 + n2)
    X_H = ((n_X_I * X_I) + (n_X_II * X_II)) / (n_X_I + n_X_II)
    e_X_H = np.sqrt(
        (n_X_I / (n_X_I + n_X_II) * e_X_I)**2
        + (n_X_II / (n_X_I + n_X_II) * e_X_II)**2)
    
    X_H_log10 = np.log10(X_H)
    e_X_H_log10 = e_X_H / (X_H * np.log(10))

    # Add back to dataframe
    A12["[{}/H]".format(species)] = X_H_log10
    A12["e_[{}/H]".format(species)] = e_X_H_log10

# Add in [Fe/H] uncertainty, (sigma = 0.03, from paper, pg. 2, section 2).
A12["e_[Fe/H]"] = 0.03

# Now that we have a standard set of columns, grab and rename
abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["A12"]]
abund_cols_new = ["{}_H_A12".format(ss) for ss in species_all["A12"]]
A12.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["A12"]]
sigma_cols_new = ["e_{}_H_A12".format(ss) for ss in species_all["A12"]]
A12.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["A12"] = A12[cols].copy()

#=========================================
# Rojas-Ayala+2012 - 2012ApJ...748...93R
#=========================================
RA12 = dataframes["RA12"]
has_sid = np.array([sid not in default_ids for sid in RA12.index.values])
RA12 = RA12[has_sid].copy()

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["RA12"]]
abund_cols_new = ["{}_H_RA12".format(ss) for ss in species_all["RA12"]]
RA12.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["RA12"]]
sigma_cols_new = ["e_{}_H_RA12".format(ss) for ss in species_all["RA12"]]
RA12.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["RA12"] = RA12[cols].copy()

#=========================================
# Mann+2013 - 2013AJ....145...52M
#=========================================
M13 = dataframes["M13"]
has_sid = np.array([sid not in default_ids for sid in M13.index.values])
M13 = M13[has_sid].copy()

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["M13"]]
abund_cols_new = ["{}_H_M13".format(ss) for ss in species_all["M13"]]
M13.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["M13"]]
sigma_cols_new = ["e_{}_H_M13".format(ss) for ss in species_all["M13"]]
M13.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["M13"] = M13[cols].copy()

#=========================================
# Gaidos+2014a - 2014ApJ...791...54G
#=========================================
G14a = dataframes["G14a"]
has_sid = np.array([sid not in default_ids for sid in G14a.index.values])
G14a = G14a[has_sid].copy()
G14a.rename(columns={"BP_RP":"bp_rp"}, inplace=True)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["G14a"]]
abund_cols_new = ["{}_H_G14a".format(ss) for ss in species_all["G14a"]]
G14a.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["G14a"]]
sigma_cols_new = ["e_{}_H_G14a".format(ss) for ss in species_all["G14a"]]
G14a.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["G14a"] = G14a[cols].copy()

#=========================================
# Mann+2014 - 2014AJ....147..160M
#=========================================
M14 = dataframes["M14"]
has_sid = np.array([sid not in default_ids for sid in M14.index.values])
M14 = M14[has_sid].copy()
M14.rename(columns={"BP-RP_dr3":"bp_rp"}, inplace=True)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["M14"]]
abund_cols_new = ["{}_H_M14".format(ss) for ss in species_all["M14"]]
M14.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["M14"]]
sigma_cols_new = ["e_{}_H_M14".format(ss) for ss in species_all["M14"]]
M14.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["M14"] = M14[cols].copy()

#=========================================
# Gaidos+2014b - 2014MNRAS.443.2561G
#=========================================
G14b = dataframes["G14b"]
has_sid = np.array([sid not in default_ids for sid in G14b.index.values])
G14b = G14b[has_sid].copy()
G14b.rename(columns={"BP_RP":"bp_rp"}, inplace=True)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["G14b"]]
abund_cols_new = ["{}_H_G14b".format(ss) for ss in species_all["G14b"]]
G14b.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["G14b"]]
sigma_cols_new = ["e_{}_H_G14b".format(ss) for ss in species_all["G14b"]]
G14b.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["G14b"] = G14b[cols].copy()

#=========================================
# Mann+2015 - 2015ApJ...804...64M
#=========================================
M15 = dataframes["M15"]
has_sid = np.array([sid not in default_ids for sid in M15.index.values])
M15 = M15[has_sid].copy()
M15.rename(columns={"BP-RP_dr3":"bp_rp"}, inplace=True)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["M15"]]
abund_cols_new = ["{}_H_M15".format(ss) for ss in species_all["M15"]]
M15.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["M15"]]
sigma_cols_new = ["e_{}_H_M15".format(ss) for ss in species_all["M15"]]
M15.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["M15"] = M15[cols].copy()

#=========================================
# Terrien+2015 - 2015ApJS..220...16T
#=========================================
pass

#=========================================
# Brewer+2016 - 2016ApJS..225...32B
#=========================================
B16 = dataframes["B16"]
has_sid = np.array([sid not in default_ids for sid in B16.index.values])
B16 = B16[has_sid].copy()
n_B16 = len(B16)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["B16"]]
abund_cols_new = ["{}_H_B16".format(ss) for ss in species_all["B16"]]

B16.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["B16"]:
    e_species = np.full(n_B16, sigmas["B16"][species])
    sigma_col_new = "e_{}_H_B16".format(species)
    B16[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["B16"] = B16[cols].copy()

#=========================================
# Montes+2018 - 2018MNRAS.479.1332M
#=========================================
M18 = dataframes["M18"]
has_sid = np.array([sid not in default_ids for sid in M18.index.values])
M18 = M18[has_sid].copy()

abund_cols_old = ["{}_H".format(ss) for ss in species_all["M18"]]
abund_cols_new = ["{}_H_M18".format(ss) for ss in species_all["M18"]]
M18.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e{}_H".format(ss) for ss in species_all["M18"]]
sigma_cols_new = ["e_{}_H_M18".format(ss) for ss in species_all["M18"]]
M18.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["M18"] = M18[cols].copy()

#=========================================
# Luck 18 - 2018AJ....155..111L
#=========================================
# Before doing anything, we need to compute the solar reference values
L18_sun = pd.read_csv("data/L18_solar.tsv", delimiter="\t", index_col="ID")

x_h_solar = {}

for species in species_all["L18"]:
    abund_X_I_col = "log{}I".format(species)
    n_lines_X_I_col = "o_log{}I".format(species)

    abund_X_II_col = "log{}II".format(species)
    n_lines_X_II_col = "o_log{}II".format(species)

    # Don't have ionised state
    if abund_X_II_col not in L18_sun.columns.values:
        x_h_solar[species] = L18_sun.loc["Sun"][abund_X_I_col]

    # Don't have ground state
    elif abund_X_I_col not in L18_sun.columns.values:
        x_h_solar[species] = L18_sun.loc["Sun"][abund_X_II_col]

    # Otherwise have both ionisation states do a weighted average to for [X/H]
    else:
        X_I = L18_sun.loc["Sun"][abund_X_I_col]
        n_X_I = L18_sun.loc["Sun"][n_lines_X_I_col]

        X_II = L18_sun.loc["Sun"][abund_X_II_col]
        n_X_II = L18_sun.loc["Sun"][n_lines_X_II_col]

        X_H = ((n_X_I * X_I) + (n_X_II * X_II)) / (n_X_I + n_X_II)

        x_h_solar[species] = X_H

#=========================================
# Now we can continue to our sample proper
L18 = dataframes["L18"]
has_sid = np.array([sid not in default_ids for sid in L18.index.values])
L18 = L18[has_sid].copy()
n_L18 = len(L18)

# We need to combine the uncertainties for some species across ionisation
# states. Unlike with A12, however, we already have [X/H], just not e_[X/H].
species_to_combine = ["Si", "Ca", "Sc", "Ti", "V", "Cr", "Fe", "Y", "Zr",]

for species in species_to_combine:
    n_X_I = L18["o_log{}I".format(species)].values
    X_I_log10 = L18["log{}I".format(species)].values
    e_X_I_log10 = L18["e_log{}I".format(species)].values

    X_I = 10**X_I_log10
    e_X_I = X_I * np.log(10) * e_X_I_log10

    # Divide by solar value
    #X_I /= 10**x_h_solar[species]
    #e_X_I /= 10**x_h_solar[species]
    
    n_X_II = L18["o_log{}II".format(species)].values
    X_II_log10 = L18["log{}II".format(species)].values
    e_X_II_log10 = L18["e_log{}II".format(species)].values

    X_II = 10**X_II_log10
    e_X_II = X_II * np.log(10) * e_X_II_log10

    # Divide by solar value
    #X_II /= 10**x_h_solar[species]
    #e_X_II /= 10**x_h_solar[species]

    # Compute combined abundance and propagate uncertainty
    #   A = ((n1 * A1) + (n2 * A2)) / (n1 + n2)
    X_H = ((n_X_I * X_I) + (n_X_II * X_II)) / (n_X_I + n_X_II)
    e_X_H = np.sqrt(
        (n_X_I / (n_X_I + n_X_II) * e_X_I)**2
        + (n_X_II / (n_X_I + n_X_II) * e_X_II)**2)
    
    X_H_log10 = np.log10(X_H)
    e_X_H_log10 = e_X_H / (X_H * np.log(10))

    # Add back to dataframe
    L18["[{}/H]_calc".format(species)] = X_H_log10
    L18["e_[{}/H]".format(species)] = e_X_H_log10

# Rename columns from only a single ionisation state for consistency
single_state_species = [
    "NaI", "MgI", "AlI", "SI", "MnI", "CoI", "NiI", "CuI", "ZnI", "SrI",
    "BaII", "LaII", "CeII", "NdII", "SmII",]# "EuII"]

for species in single_state_species:
    abund_col = "[{}/H]".format(species)
    sigma_col = "e_log{}".format(species)

    abund_col_new = abund_col.replace("I", "")
    sigma_col_new = "e_{}".format(abund_col_new)

    L18.rename(
        columns={abund_col:abund_col_new, sigma_col:sigma_col_new},
        inplace=True)

# Now that we have a standard set of columns, grab and rename
abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["L18"]]
abund_cols_new = ["{}_H_L18".format(ss) for ss in species_all["L18"]]
L18.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["L18"]]
sigma_cols_new = ["e_{}_H_L18".format(ss) for ss in species_all["L18"]]
L18.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["L18"] = L18[cols].copy()

#=========================================
# Dressing+2019 - 2019AJ....158...87D
#=========================================
D19 = dataframes["D19"]
has_sid = np.array([sid not in default_ids for sid in D19.index.values])
D19 = D19[has_sid].copy()

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["D19"]]
abund_cols_new = ["{}_H_D19".format(ss) for ss in species_all["D19"]]
D19.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["D19"]]
sigma_cols_new = ["e_{}_H_D19".format(ss) for ss in species_all["D19"]]
D19.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["D19"] = D19[cols].copy()

#=========================================
# Rice & Brewer 2020 - 2020ApJ...898..119R
#=========================================
RB20 = dataframes["RB20"]
has_sid = np.array([sid not in default_ids for sid in RB20.index.values])
RB20 = RB20[has_sid].copy()
n_RB20 = len(RB20)

abund_cols_old = ["{}/H".format(ss) for ss in species_all["RB20"]]
abund_cols_new = ["{}_H_RB20".format(ss) for ss in species_all["RB20"]]

RB20.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["RB20"]:
    e_species = np.full(n_RB20, sigmas["RB20"][species])
    sigma_col_new = "e_{}_H_RB20".format(species)
    RB20[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["RB20"] = RB20[cols].copy()

#=========================================
# Maldonado+2020 - 2020A&A...644A..68M
#=========================================
M20 = dataframes["M20"]
has_sid = np.array([sid not in default_ids for sid in M20.index.values])
M20 = M20[has_sid].copy()
n_M20 = len(M20)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["M20"]]
abund_cols_new = ["{}_H_M20".format(ss) for ss in species_all["M20"]]

M20.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["M20"]:
    e_species = np.full(n_M20, sigmas["M20"][species])
    sigma_col_new = "e_{}_H_M20".format(species)
    M20[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["M20"] = M20[cols].copy()

#=========================================
# Monty GALAH+Gaia [X/Fe] predictions
#=========================================
SM25 = dataframes["SM25"]
has_sid = np.array([sid not in default_ids for sid in SM25.index.values])
SM25 = SM25[has_sid].copy()
n_SM25 = len(SM25)

abund_cols_old = ["[{}/Fe]".format(ss) for ss in species_all["SM25"]]
abund_cols_new = ["{}_Fe_SM25".format(ss) for ss in species_all["SM25"]]

SM25.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/Fe]".format(ss) for ss in species_all["SM25"]]
sigma_cols_new = ["e_{}_Fe_SM25".format(ss) for ss in species_all["SM25"]]
SM25.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
cols.append("bp_rp")
dataframes_cut["SM25"] = SM25[cols].copy()

#------------------------------------------------------------------------------
# Massaging things
#------------------------------------------------------------------------------
# If we want to use any [X/Fe] not from our reference scale, we require that
# star to have a BP-RP so it can be corrected for inter-catalogue systematics.
# However, this poses a problem for stars (especially binary hosts which we
# don't automatically reject for not having Gaia or 2MASS) as it means that we
# won't be able to use their [X/Fe] later. Thus, if we want to use one of these
# stars we need to provide it with a dummy (but reasonable) BP-RP value from a
# stellar twin. 

# Alpha Cen A (HD 128620), the binary reference to Proxima Cen, is in VF05 but
# is too bright to be in Gaia. GBS (Soubiran+24) lists its stellar parameters
# as: Teff~5804, logg~4.29, and [Fe/H]~0.20. Checking VF05 for a suitable twin,
# we select HD 24040 with Teff~5853, logg~4.36, [Fe/H]~0.21. This target has
# BP-RP = 0.8125124 in Gaia DR3, which we assign to Alpha Cen A here, which
# allows the systematics between VF05 and L18 to be corrected for.
dataframes_cut["VF05"].loc["HD 128620"]["bp_rp"] = 0.81
dataframes_cut["L18"].loc["HD 128620"]["bp_rp"] = 0.81

#------------------------------------------------------------------------------
# Join all separate tables
#------------------------------------------------------------------------------
for ref in dataframes_cut.keys():
    dataframes_cut[ref].rename(
        columns={"bp_rp":"bp_rp_{}".format(ref)}, inplace=True)
    df_comb = df_comb.join(
        dataframes_cut[ref], "source_id_dr3", rsuffix="_{}".format(ref))

# Grab only a single BP-RP column
bp_rp_cols = ["bp_rp_{}".format(ref) for ref in dataframes_cut.keys()]

# NOTE: by taking the median (rather than any single value) we are assuming
# that the only difference in BP-RP will be due to rounding issues caused by
# selecting a different number of significant figures.
bp_rp_adopt = np.nanmedian(df_comb[bp_rp_cols].values, axis=1)

df_comb.insert(loc=0, column="BP_RP", value=bp_rp_adopt,)

df_comb.drop(columns=bp_rp_cols, inplace=True)

#------------------------------------------------------------------------------
# Fitting + Correcting Residuals
#------------------------------------------------------------------------------
# Default polynomial order and outlier rejection options
POLY_ORDER = 4
OUTLIER_DEX = 0.3

# K dwarfs. Note: we need to give a secondary reference here since VF05 doesn't
# have all [X/Fe] of interest. 
species_to_correct_K = ["Fe_H", "Ti_H", "Mg_H", "Ca_H", "Na_H", "Al_H"]
comp_ref_K = "VF05"
comp_ref_secondary_K = "B16"
references_to_compare_K = np.array(["R07", "A12", "B16", "M18", "L18", "RB20"])

print("K Dwarfs\n--------")
fit_dict_K = correct_abundance_trends(
    chem_df=df_comb,
    species_to_correct=species_to_correct_K,
    comp_ref=comp_ref_K,
    comp_ref_secondary=comp_ref_secondary_K,
    references_to_compare=references_to_compare_K,
    poly_order=POLY_ORDER,
    outlier_dex=OUTLIER_DEX,)

# M dwarfs
species_to_correct_M = ["Fe_H",]
comp_ref_M = "M15"
references_to_compare_M = np.array(["RA12", "G14a", "G14b", "M20"])

print("\nM Dwarfs\n--------")
fit_dict_M = correct_abundance_trends(
    chem_df=df_comb,
    species_to_correct=species_to_correct_M,
    comp_ref=comp_ref_M,
    references_to_compare=references_to_compare_M,
    poly_order=POLY_ORDER,
    outlier_dex=OUTLIER_DEX,)

#------------------------------------------------------------------------------
# Computing [X/Fe]
#------------------------------------------------------------------------------
# Compute [X/Fe] for the species we've corrected for systematics
X_Fe_to_compute =  [ss.replace("_H", "") for ss in species_to_correct_K]

sp.compute_X_Fe(
    chem_df=df_comb,
    species=X_Fe_to_compute,
    samples=samples.keys(),)

#------------------------------------------------------------------------------
# Correcting [X/Fe] systematics for chemodynamic trends
#------------------------------------------------------------------------------
species_to_correct_CD = ["Mg_Fe", "Ca_Fe", "Ti_Fe", "Na_Fe", "Al_Fe"]
comp_ref_CD = "B16"
references_to_compare_CD = np.array(["SM25"])

fit_dict_CD = correct_abundance_trends(
    chem_df=df_comb,
    species_to_correct=species_to_correct_CD,
    comp_ref=comp_ref_CD,
    references_to_compare=references_to_compare_CD,
    poly_order=0,   # Note: single order poly only!
    outlier_dex=OUTLIER_DEX,)

#------------------------------------------------------------------------------
# [Optional] Remove unused [X/Fe] to reduce the number of columns
#------------------------------------------------------------------------------
possible_X_H = ["C", "N", "O", "Na", "Mg", "Al", "Si", "S", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Sr", "Y", "Zr", "Ba", "La",
    "Ce", "Nd", "Sm", "Eu"]

if remove_unused_X_H:
    columns = df_comb.columns.values
    drop_mask = np.full_like(columns, False).astype(bool)

    for element in possible_X_H:
        X_H = "{}_H".format(element)

        # Dron't drop elements we've corrected
        if X_H in species_to_correct_K:
            continue

        # Check all columns, update our mask accordingly
        drop_mask = np.logical_or([X_H in col for col in columns], drop_mask)

    # Remove these columns
    df_comb.drop(columns=columns[drop_mask], inplace=True)

#------------------------------------------------------------------------------
# Save to file
#------------------------------------------------------------------------------
species_fn = [stc.replace("_H", "") for stc in species_to_correct_K]

save_filename = "data/lit_chemistry_corrected_{}.tsv".format(
    "_".join(species_fn))

# Dump corrected DataFrame
df_comb.to_csv(save_filename, sep="\t")

#------------------------------------------------------------------------------
# Plotting
#------------------------------------------------------------------------------
# We produce one set of plots for M and K dwarfs, with each set having one plot
# per species corrected.
DO_LIMIT_Y_EXTENT = True

# Plot abundance trends for K dwarfs
BP_RP_LIMS_K = (0.52, 1.4)
ABUND_Y_LIMS_K = (-0.35, 0.35)
BP_RP_TICKS_K = (0.1, 0.05)

plot_abundance_trends(
    chem_df=df_comb,
    fit_dict=fit_dict_K,
    species_to_correct=species_to_correct_K,
    comp_ref=comp_ref_K,
    comp_ref_secondary=comp_ref_secondary_K,
    references_to_compare=references_to_compare_K,
    bp_rp_lims=BP_RP_LIMS_K,
    abund_Y_lims=ABUND_Y_LIMS_K,
    do_limit_y_extent=DO_LIMIT_Y_EXTENT,
    fn_label="K",
    bp_rp_ticks=BP_RP_TICKS_K,)

# Plot abundance trends for M dwarfs
BP_RP_LIMS_M = (1.6, 4.8)
ABUND_Y_LIMS_M = (-0.5, 0.5)
BP_RP_TICKS_M = (0.2, 0.1)

plot_abundance_trends(
    chem_df=df_comb,
    fit_dict=fit_dict_M,
    species_to_correct=species_to_correct_M,
    comp_ref=comp_ref_M,
    references_to_compare=references_to_compare_M,
    bp_rp_lims=BP_RP_LIMS_M,
    abund_Y_lims=ABUND_Y_LIMS_M,
    do_limit_y_extent=DO_LIMIT_Y_EXTENT,
    fn_label="M",
    bp_rp_ticks=BP_RP_TICKS_M,)

# Plot abundance trends for chemodynamic [X/Fe]
plot_abundance_trends(
    chem_df=df_comb,
    fit_dict=fit_dict_CD,
    species_to_correct=species_to_correct_CD,
    comp_ref=comp_ref_CD,
    references_to_compare=references_to_compare_CD,
    bp_rp_lims=BP_RP_LIMS_K,
    abund_Y_lims=ABUND_Y_LIMS_M,
    do_limit_y_extent=DO_LIMIT_Y_EXTENT,
    fn_label="CD",
    bp_rp_ticks=BP_RP_TICKS_K,
    figsize=(16,10))

# Plot literature recovery for chemodynamic correction
plot_chemodynamic_one_to_one_recovery(
    chem_df=df_comb,
    species_to_plot=species_to_correct_CD,
    comp_ref=comp_ref_CD,
    figsize=(16,10))