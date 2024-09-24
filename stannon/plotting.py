"""Plotting functions related to Stannon
"""
import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plumage.plotting as pplt
import matplotlib.ticker as plticker
from collections import OrderedDict
from stannon.vectorizer import PolynomialVectorizer
from scipy.interpolate import splrep, BSpline

def plot_label_recovery(
    label_values,
    e_label_values,
    label_pred,
    e_label_pred,
    teff_lims=(2800,4500),
    logg_lims=(4.4,5.4),
    feh_lims=(-1.0,0.75),
    show_offset=True,
    fn_suffix="",
    teff_ticks=(500,250,100,50),
    logg_ticks=(0.5,0.25,0.2,0.1),
    feh_ticks=(0.5,0.25,0.5,0.25),
    plot_folder="plots/",):
    """Plot 1x3 grid of Teff, logg, and [Fe/H] literature comparisons.

    Saves as paper/std_comp<fn_suffix>.<pdf/png>.

    Parameters
    ----------
    label_values: 2D numpy array
            Label array with columns [teff, logg, feh]
        
    label_pred: 2D numpy array
        Predicted label array with columns [teff, logg, feh]

    teff_lims, feh_lims: float array, default:[3000,4600],[-1.4,0.75]
        Axis limits for Teff and [Fe/H] respectively.

    show_offset: bool, default: False
        Whether to plot the median offset as text.

    fn_suffix: string, default: ''
        Suffix to append to saved figures
        
    title_text: string, default: ''
        Text for fig.suptitle.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    plt.close("all")

    # Panel label with n_labels
    panel_label = "{:0.0f} Label".format(label_pred.shape[1])

    # Make plot
    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5)

    # Temperatures
    pplt.plot_std_comp_generic(
        fig=fig,
        axis=axes[0],
        lit=label_values[:,0],
        e_lit=e_label_values[:,0],
        fit=label_pred[:,0],
        e_fit=e_label_pred[:,0],
        colour=label_values[:,2],
        fit_label=r"$T_{\rm eff}$ (K, $\it{Cannon}$)",
        lit_label=r"$T_{\rm eff}$ (K, Adopted)",
        cb_label="[Fe/H] (Adopted)",
        x_lims=teff_lims,
        y_lims=teff_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,
        panel_label=panel_label,)
    
    # Ensure we only plot logg for stars we haven't given a default value to.
    logg_mask = e_label_values[:,2] < 0.2

    # Gravity
    pplt.plot_std_comp_generic(
        fig=fig,
        axis=axes[1],
        lit=label_values[:,1][logg_mask],
        e_lit=e_label_values[:,1][logg_mask],
        fit=label_pred[:,1][logg_mask],
        e_fit=e_label_pred[:,1][logg_mask],
        colour=label_values[:,2][logg_mask],
        fit_label=r"$\log g$ ($\it{Cannon}$)",
        lit_label=r"$\log g$ (Adopted)",
        cb_label="[Fe/H] (Adopted)",
        x_lims=logg_lims,
        y_lims=logg_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=logg_ticks,
        panel_label=panel_label,
        plot_resid_y_label=False,)
    
    # Ensure we only plot [Fe/H] for stars we haven't given a default value to.
    feh_mask = e_label_values[:,2] < 0.2

    # [Fe/H]]
    pplt.plot_std_comp_generic(
        fig=fig,
        axis=axes[2],
        lit=label_values[:,2][feh_mask],
        e_lit=e_label_values[:,2][feh_mask],
        fit=label_pred[:,2][feh_mask],
        e_fit=e_label_pred[:,2][feh_mask],
        colour=label_values[:,0][feh_mask],
        fit_label=r"[Fe/H] ($\it{Cannon}$)",
        lit_label=r"[Fe/H] (Adopted)",
        cb_label=r"$T_{\rm eff}$ (K, Adopted)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        panel_label=panel_label,
        plot_resid_y_label=False,)

    fig.set_size_inches(12, 3)
    fig.tight_layout()

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_param_recovery{}".format(fn_suffix))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_label_recovery_per_source(
    label_values,
    e_label_values,
    label_pred,
    e_label_pred,
    obs_join,
    teff_lims=(2800,4500),
    feh_lims=(-1.0,0.75),
    show_offset=True,
    fn_suffix="",
    teff_ticks=(500,250,100,50),
    feh_ticks=(0.5,0.25,0.5,0.25),
    do_plot_mid_K_panel=False,
    plot_folder="plots/",):
    """Plot 1x3 grid of Teff, logg, and [Fe/H] literature comparisons.

    Saves as paper/std_comp<fn_suffix>.<pdf/png>.

    Parameters
    ----------
    label_values: 2D numpy array
            Label array with columns [teff, logg, feh]
        
    label_pred: 2D numpy array
        Predicted label array with columns [teff, logg, feh]

    teff_lims, feh_lims: float array, default:[3000,4600],[-1.4,0.75]
        Axis limits for Teff and [Fe/H] respectively.

    show_offset: bool, default: False
        Whether to plot the median offset as text.

    fn_suffix: string, default: ''
        Suffix to append to saved figures
        
    title_text: string, default: ''
        Text for fig.suptitle.

    do_plot_mid_K_panel: boolean, default: False
        Whether to plot an extra panel for mid-K dwarf benchmarks.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    plt.close("all")

    # Panel label with n_labels
    panel_label = "{:0.0f} Label".format(label_pred.shape[1])

    # Make plot
    if do_plot_mid_K_panel:
        fig, axes  = plt.subplots(1,5)
        (ax_teff_int, ax_feh_m15, ax_feh_ra12, ax_feh_cpm, ax_feh_mid_k) = axes
    else:
        fig, axes = plt.subplots(1,4)
        (ax_teff_int, ax_feh_m15, ax_feh_ra12, ax_feh_cpm) = axes

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.0)

    # Interferometric temperatures
    int_mask = ~np.isnan(obs_join["teff_int"])

    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_teff_int,
        lit=label_values[:,0][int_mask],
        e_lit=e_label_values[:,0][int_mask],
        fit=label_pred[:,0][int_mask],
        e_fit=e_label_pred[:,0][int_mask],
        colour=label_values[:,2][int_mask],
        fit_label=r"$T_{\rm eff}$ (K, $\it{Cannon}$)",
        lit_label=r"$T_{\rm eff}$ (K, Interferometry)",
        cb_label="[Fe/H] (Adopted)",
        x_lims=teff_lims,
        y_lims=teff_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,
        panel_label=panel_label,)

    # Mann+15 [Fe/H]
    feh_mask = ~np.isnan(obs_join["feh_m15"])

    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_feh_m15,
        lit=label_values[:,2][feh_mask],
        e_lit=e_label_values[:,2][feh_mask],
        fit=label_pred[:,2][feh_mask],
        e_fit=e_label_pred[:,2][feh_mask],
        colour=label_values[:,0][feh_mask],
        fit_label=r"[Fe/H] ($\it{Cannon}$)",
        lit_label=r"[Fe/H] (Mann+15)",
        cb_label=r"$T_{\rm eff}$ (K, Adopted)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        panel_label=panel_label,
        plot_cbar_label=False,
        plot_resid_y_label=False,)

    # Rojas-Ayala+12 [Fe/H]
    feh_mask = ~np.isnan(obs_join["feh_ra12"])

    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_feh_ra12,
        lit=label_values[:,2][feh_mask],
        e_lit=e_label_values[:,2][feh_mask],
        fit=label_pred[:,2][feh_mask],
        e_fit=e_label_pred[:,2][feh_mask],
        colour=label_values[:,0][feh_mask],
        fit_label=r"[Fe/H] ($\it{Cannon}$)",
        lit_label=r"[Fe/H] (Rojas-Ayala+12)",
        cb_label=r"$T_{\rm eff}$ (K, Adopted)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        panel_label=panel_label,
        plot_cbar_label=False,
        plot_y_label=True,
        plot_resid_y_label=False,)

    # CPM [Fe/H]
    feh_mask = obs_join["is_cpm"].values

    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_feh_cpm,
        lit=label_values[:,2][feh_mask],
        e_lit=e_label_values[:,2][feh_mask],
        fit=label_pred[:,2][feh_mask],
        e_fit=e_label_pred[:,2][feh_mask],
        colour=label_values[:,0][feh_mask],
        fit_label=r"[Fe/H] ($\it{Cannon}$)",
        lit_label=r"[Fe/H] (Binary Primary)",
        cb_label=r"$T_{\rm eff}$ (K, Adopted)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        panel_label=panel_label,
        plot_y_label=True,
        plot_resid_y_label=False,)
    
    # Mid-K dwarfs [Fe/H]
    if do_plot_mid_K_panel:
        feh_mask = obs_join["is_mid_k_dwarf"].values

        pplt.plot_std_comp_generic(
            fig=fig,
            axis=ax_feh_mid_k,
            lit=label_values[:,2][feh_mask],
            e_lit=e_label_values[:,2][feh_mask],
            fit=label_pred[:,2][feh_mask],
            e_fit=e_label_pred[:,2][feh_mask],
            colour=label_values[:,0][feh_mask],
            fit_label=r"[Fe/H] ($\it{Cannon}$)",
            lit_label=r"[Fe/H] (Mid-K Literature)",
            cb_label=r"$T_{\rm eff}$ (K, Adopted)",
            x_lims=feh_lims,
            y_lims=feh_lims,
            cmap="magma",
            show_offset=show_offset,
            ticks=feh_ticks,
            panel_label=panel_label,
            plot_y_label=True,
            plot_resid_y_label=False,)

    fig.set_size_inches(16, 3)
    fig.tight_layout()

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_param_recovery_ps{}.pdf".format(fn_suffix))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_label_recovery_abundances(
    label_values,
    e_label_values,
    label_pred,
    e_label_pred,
    obs_join,
    abundance_labels,
    feh_lims=(-1.0,0.75),
    show_offset=True,
    fn_suffix="",
    feh_ticks=(0.5,0.25,0.5,0.25),
    plot_folder="plots/",):
    """Plot 1x3 grid of Teff, logg, and [Fe/H] literature comparisons.

    Saves as paper/std_comp<fn_suffix>.<pdf/png>.

    Parameters
    ----------
    label_values: 2D numpy array
            Label array with columns [teff, logg, feh]
        
    label_pred: 2D numpy array
        Predicted label array with columns [teff, logg, feh]

    teff_lims, feh_lims: float array, default:[3000,4600],[-1.4,0.75]
        Axis limits for Teff and [Fe/H] respectively.

    show_offset: bool, default: False
        Whether to plot the median offset as text.

    fn_suffix: string, default: ''
        Suffix to append to saved figures
        
    title_text: string, default: ''
        Text for fig.suptitle.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    plt.close("all")

    # Panel label with n_labels
    panel_label = "{:0.0f} Label".format(label_pred.shape[1])

    n_abundances = len(abundance_labels)

    if n_abundances == 0:
        print("No abundances to plot!")
        return

    # Make plot
    fig, axes = plt.subplots(1, n_abundances)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5)

    if n_abundances == 1:
        axes = [axes]

    # Plot each abundance
    for abundance_i, abundance in enumerate(abundance_labels):
        label_i = 3 + abundance_i

        if "H" in abundance:
            abundance_label = "[{}/H]".format(abundance.split("_")[0])
        else:
            abundance_label = "[{}/Fe]".format(abundance.split("_")[0])
        
        abund_sources = obs_join["label_source_{}".format(abundance)]

        abundance_mask = np.array([src != "" for src in abund_sources])

        #~np.isnan(obs_join["label_adopt_{}".format(abundance)])

        pplt.plot_std_comp_generic(
            fig=fig,
            axis=axes[abundance_i],
            lit=label_values[:,label_i][abundance_mask],
            e_lit=e_label_values[:,label_i][abundance_mask],
            fit=label_pred[:,label_i][abundance_mask],
            e_fit=e_label_pred[:,label_i][abundance_mask],
            colour=label_values[:,0][abundance_mask],
            fit_label=r"{} ($\it{{Cannon}}$)".format(abundance_label),
            lit_label=r"{} (Adopted)".format(abundance_label),
            cb_label=r"$T_{\rm eff}\,$K (Adopted)",
            x_lims=feh_lims,
            y_lims=feh_lims,
            cmap="magma",
            show_offset=show_offset,
            ticks=feh_ticks,
            panel_label=panel_label,)
    
    fig.set_size_inches(4*n_abundances, 3)
    fig.tight_layout()
    
    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_param_recovery_abundance{}.pdf".format(fn_suffix))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_cannon_cmd(
    benchmark_colour,
    benchmark_mag,
    benchmark_feh,
    science_colour=None,
    science_mag=None,
    x_label=r"$BP-RP$",
    y_label=r"$M_{K_S}$",
    highlight_mask=None,
    highlight_mask_label="",
    highlight_mask_2=None,
    highlight_mask_label_2="",
    bp_rp_cutoff=0,
    plot_folder="plots/",):
    """Plots a colour magnitude diagram using the specified columns and saves
    the result as paper/{label}_cmd.pdf. Optionally can plot a second set of
    stars for e.g. comparison with standards.

    Parameters
    ----------
    info_cat: pandas.DataFrame
        Table of stellar literature info.

    info_cat_2: pandas.DataFrame, default: None
        Table of stellar literature info for second set of stars (e.g. 
        standards). Optional.

    plot_toi_ids: bool, default: False
        Plot the TOI IDs on top of the points for diagnostic purposes.

    colour: string, default: 'Bp-Rp'
        Column name for colour (x) axis of CMD.

    abs_mag: string, default: 'G_mag_abs'
        Column name for absolute magnitude (y) axis of CMD.

    x_label, y_label: string, default: r'$B_P-R_P$', r'$M_{\rm G}$'
        Axis labels for X and Y axis respectively.

    label: string, default: 'tess'
        Label to use in filename, e.g. {label}_cmd.pdf

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    plt.close("all")
    fig, axis = plt.subplots()

    # Plot benchmarks
    scatter = axis.scatter(
        benchmark_colour, 
        benchmark_mag, 
        zorder=1,
        c=benchmark_feh,
        label="{} ({})".format("Benchmark", len(benchmark_colour)),
        alpha=0.9,
        cmap="viridis",
    )

    cb = fig.colorbar(scatter, ax=axis)
    cb.set_label("[Fe/H]")

    # Plot science targets, making sure to not plot any science targets beyond
    # the extent of our benchmarks
    if (science_colour is not None and science_mag is not None 
        and len(science_colour) > 0 and len(science_mag) > 0):
        scatter = axis.scatter(
            science_colour[science_colour > bp_rp_cutoff],
            science_mag[science_colour > bp_rp_cutoff],
            marker="o",
            edgecolor="black",#"#ff7f0e",
            #facecolors="none",
            zorder=2,
            alpha=0.6,
            label="Science",)

    # If we've been given a highlight mask, plot for diagnostic reasons
    if highlight_mask is not None:
        label = "{} ({})".format(
            highlight_mask_label, int(np.sum(highlight_mask)))
        
        scatter = axis.scatter(
            benchmark_colour[highlight_mask],
            benchmark_mag[highlight_mask],
            marker="o",
            c=benchmark_feh[highlight_mask],
            edgecolor="k",
            linewidths=1.2,
            zorder=1,
            label=label,)
        
    # If we've been given a *second* highlight mask, also plot
    if highlight_mask_2 is not None:
        label = "{} ({})".format(
            highlight_mask_label_2, int(np.sum(highlight_mask_2)))
        scatter = axis.scatter(
            benchmark_colour[highlight_mask_2],
            benchmark_mag[highlight_mask_2],
            marker="o",
            c=benchmark_feh[highlight_mask_2],
            edgecolor="k",
            linewidths=1.2,
            linestyle= (0, (1, 1)),
            zorder=1,
            label=label,)

    plt.legend(loc="best", fontsize="large")
    
    # Flip magnitude axis
    ymin, ymax = axis.get_ylim()
    axis.set_ylim((ymax, ymin))

    axis.set_xlabel(x_label, fontsize="large")
    axis.set_ylabel(y_label, fontsize="large")

    axis.tick_params(axis='both', which='major', labelsize="large")

    axis.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

    axis.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))

    fig.tight_layout()

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_cmd")

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_kiel_diagram(
    teffs,
    e_teffs,
    loggs,
    e_loggs,
    fehs,
    max_teff=4200,
    label="",):
    """
    """
    plt.close("all")
    fig, axis = plt.subplots()

    # Mask only those stars within the bounds of our trained Cannon model
    mask = teffs < max_teff

    # Plot 
    scatter = axis.scatter(
        teffs[mask],
        loggs[mask],
        zorder=1,
        c=fehs[mask],
        cmap="viridis",
    )

    cb = fig.colorbar(scatter, ax=axis)
    cb.set_label("[Fe/H]")

    axis.errorbar(
        x=teffs[mask],
        y=loggs[mask],
        xerr=e_teffs[mask],
        yerr=e_loggs[mask],
        zorder=0,
        ecolor="black",
        elinewidth=0.4,
        fmt=".",
    )

    # Flip axes
    ymin, ymax = axis.get_ylim()
    axis.set_ylim((ymax, ymin))

    xmin, xmax = axis.get_xlim()
    axis.set_xlim((xmax, xmin))

    axis.set_xlabel(r"$T_{\rm eff}$ (K)", fontsize="large")
    axis.set_ylabel(r"$\log g$", fontsize="large")

    axis.tick_params(axis='both', which='major', labelsize="large")

    axis.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=100))

    axis.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))

    fig.tight_layout()
    plt.savefig("paper/cannon_kiel_{}.png".format(label), dpi=200)
    plt.savefig("paper/cannon_kiel_{}.pdf".format(label))


def plot_theta_coefficients(
    sm,
    teff_scale=0.3,
    x_lims=(5700,6400),
    y_spec_lims=(0,2.5),
    y_theta_linear_lims=(-0.1,0.1),
    y_theta_quadratic_lims=(-0.1,0.1),
    y_theta_cross_lims=(-0.1,0.1),
    y_s2_lims=(-0.001,0.01),
    x_ticks=(500,100),
    linewidth=0.5,
    alpha=0.9,
    fn_label="",
    fn_suffix="",
    leg_loc="upper center",
    line_list=None,
    line_list_cm="cubehelix",
    species_to_plot=[],
    species_line_width=0.75,
    species_line_lims_spec=(1.6,2.0),
    species_line_lims_scatter=(0.003,0.004),
    only_plot_first_order_coeff=True,
    sm2=None,
    sm1_label="",
    sm2_label="",
    use_logarithmic_scale_axis=False,
    plot_folder="plots/",):
    """Plot fluxes, values of first order theta coefficients for Teff, logg,
    and [Fe/H], as well as model scatter - all against wavelength.

    Parameters
    ----------
    TODO

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    plt.close("all")
    # Three axes if plotting only spectra, first order coeff, and the scatter
    if only_plot_first_order_coeff:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 6))
    
    # Five axes if we're plotting all theta coefficients
    else:
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(16, 7.5))

    fig.subplots_adjust(
        left=0.05,
        bottom=0.075,
        right=0.98,
        top=0.975,
        hspace=0.000,
        wspace=0.001)

    axes = axes.flatten()

    # We want to avoid plotting lines across the gaps we've excluded, so we're
    # going to insert nans in the breaks so that matplotlib leaves a gap. This
    # is a bit clunky, but this involves looping over each gap and inserting a
    # fake wavelength value and corresponding nan for the theta and scatter 
    # arrays.
    gap_px = np.argwhere(
        np.abs(sm.masked_wl[:-1] - sm.masked_wl[1:]) > 1.0)[:,0]
    gap_px = np.concatenate((gap_px+1, [sm.P]))

    px_min = 0

    wave = []
    theta = []
    scatter = []

    for px_max in gap_px:
        wave.append(np.concatenate(
            (sm.masked_wl[px_min:px_max], [sm.masked_wl[px_max-1]+1])))
        theta.append(np.concatenate(
            (sm.theta[px_min:px_max],
                np.atleast_2d(np.full(sm.N_COEFF, np.nan)))))
        scatter.append(np.concatenate((sm.s2[px_min:px_max], [np.nan])))

        px_min = px_max

    wave = np.concatenate(wave)
    theta = np.concatenate(theta)
    scatter = np.concatenate(scatter)

    # Adjust scale
    theta[:,1] *= teff_scale

    # Grab each set of coefficients (linear, quadratic, cross-term) and format
    # as appropriate for plotting
    vectorizer = PolynomialVectorizer(sm.label_names, 2)
    theta_lvec = vectorizer.get_human_readable_label_vector()

    # Format for plotting
    theta_lvec = theta_lvec.replace("teff", r"$T_{eff}$")
    theta_lvec = theta_lvec.replace("logg", r"$\log g$")
    theta_lvec = theta_lvec.replace("feh", "[Fe/H]")

    if sm.L > 3:
        for abundance_i, abundance in enumerate(sm.label_names[3:]):
            label_i = 4 + abundance_i
            
            abundance_label = "[{}]".format(abundance.replace("_", "/"))

            theta_lvec =  \
                theta_lvec.replace(abundance, abundance_label)
    
    theta_lvec = theta_lvec.replace("*", r"$\times$")
    theta_lvec =  theta_lvec.replace("^2", r"$^2$")
    
    theta_lvec = theta_lvec.split(" + ")

    linear_term_ii = np.arange(sm.L) + 1
    quad_term_ii = np.array(
        [i for i in range(len(theta_lvec)) if "^" in theta_lvec[i]])
    cross_term_ii = np.array(
        [i for i in range(len(theta_lvec)) if "times" in theta_lvec[i]])

    # -------------------------------------------------------------------------
    # Panel 1: Spectra
    # -------------------------------------------------------------------------
    # Initialise teff colours
    cmap = cm.get_cmap("magma")
    teff_min = np.min(sm.training_labels[:,0])
    teff_max = np.max(sm.training_labels[:,0])

    # Do bad px masking
    masked_spectra = sm.training_data.copy()
    masked_spectra[sm.bad_px_mask] = np.nan

    # First plot spectra
    for star_i, star in enumerate(masked_spectra):
        teff = sm.training_labels[star_i, 0]
        colour = cmap((teff-teff_min)/(teff_max-teff_min))

        axes[0].plot(sm.wavelengths, star, linewidth=0.2, c=colour)

    # Only show teff_scale if != 1.0
    if teff_scale == 1.0:
        teff_label =  r"$T_{\rm eff}$"
    else:
        teff_label =  r"$T_{\rm eff} \times$" + "{:0.1f}".format(teff_scale)

    # -------------------------------------------------------------------------
    # Panel 2: First Order Coefficients
    # -------------------------------------------------------------------------
    labels = [teff_label, r"$\log g$", "[Fe/H]"]

    for label_i in linear_term_ii:
        axes[1].plot(
            wave,
            theta[:,label_i],
            linewidth=linewidth,
            alpha=alpha,
            label=theta_lvec[label_i],)

    axes[1].hlines(0, 3400, 7100, linestyles="dashed", linewidth=0.1)

    # -------------------------------------------------------------------------
    # [Optional] Panel 4 + 5: Quadratic + cross term coefficients
    # -------------------------------------------------------------------------
    if not only_plot_first_order_coeff:
        # Plot quadratic coefficents
        for quad_coeff_i in quad_term_ii:
            axes[2].plot(
                wave,
                theta[:,quad_coeff_i],
                linewidth=linewidth,
                alpha=alpha,
                label=theta_lvec[quad_coeff_i],)

        # Plos cross-term coefficients
        for cross_coeff_i in cross_term_ii:
            axes[3].plot(
                wave,
                theta[:,cross_coeff_i],
                linewidth=linewidth,
                alpha=alpha,
                label=theta_lvec[cross_coeff_i],)
        
        axes[2].set_ylabel(r"$\theta_{\rm Quadratic}$")
        axes[3].set_ylabel(r"$\theta_{\rm Cross}$")

        axes[2].set_ylim(y_theta_quadratic_lims)
        axes[3].set_ylim(y_theta_cross_lims)


    # -------------------------------------------------------------------------
    # Final Panel: Scatter
    # -------------------------------------------------------------------------
    axes[-1].plot(
        wave,
        scatter,
        linewidth=linewidth,
        label=sm1_label,
        alpha=alpha,
        color="darkturquoise")

    # If we've been given a second Stannon model, plot its scatter as well
    if sm2 is not None:
        scatter_sm2 = scatter.copy()
        scatter_sm2[~np.isnan(scatter_sm2)] = sm2.s2
        axes[-1].plot(
            wave,
            scatter_sm2,
            linewidth=linewidth,
            label=sm2_label,
            alpha=alpha,
            color="firebrick")

    if use_logarithmic_scale_axis:
        axes[-1].set_yscale("log")

    # -------------------------------------------------------------------------
    # [Optional] Atomic line plot
    # -------------------------------------------------------------------------
    # Overplot line list on spectrum and scatter subplots
    if line_list is not None and len(species_to_plot) > 0:
        # Remove any species not in our list
        species_mask = np.isin(line_list["ion"].values, species_to_plot)
        line_list_adopt = line_list[species_mask].copy()

        # Count how many unique species are in the line list
        unique_species = list(set(line_list_adopt["ion"].values))
        unique_species.sort()
        n_unique_species = len(unique_species)
        colour_i = np.arange(len(unique_species))/n_unique_species
        species_mapping_dict = OrderedDict(zip(unique_species, colour_i))

        # Get the colour map for our lines
        cmap = cm.get_cmap(line_list_cm)

        # Only print those in our wavelength range
        line_mask = np.logical_and(
            line_list_adopt["wl"].values > x_lims[0],
            line_list_adopt["wl"].values < x_lims[1],)
        
        for line_i, line_data in line_list_adopt[line_mask].iterrows():
            # Change arabic numbers to roman numerals
            species_str = line_data["ion"].replace("1", "I").replace("2", "II")

            # Label lines on spectral plot
            axes[0].vlines(
                x=line_data["wl"],
                ymin=species_line_lims_spec[0],
                ymax=species_line_lims_spec[1],
                linewidth=species_line_width,
                colors=cmap(species_mapping_dict[line_data["ion"]]),
                label=species_str,
                alpha=0.6,)
            
            # Label lines on scatter plot
            axes[-1].vlines(
                x=line_data["wl"],
                ymin=species_line_lims_scatter[0],
                ymax=species_line_lims_scatter[1],
                linewidth=species_line_width,
                colors=cmap(species_mapping_dict[line_data["ion"]]),
                label=species_str,
                alpha=0.6,)

    else:
        n_unique_species = 0

    # -------------------------------------------------------------------------
    # Final Setup
    # -------------------------------------------------------------------------
    for axis in axes:
        # Mask emission and telluric regions for all panels
        pplt.shade_excluded_regions(
            wave=sm.wavelengths,
            bad_px_mask=~sm.adopted_wl_mask,
            axis=axis,
            res_ax=None,
            colour="red",
            alpha=0.25,
            hatch=None)
        
        # Legend
        handles, labels = axis.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        leg = axis.legend(
            handles=by_label.values(),
            labels=by_label.keys(),
            loc=leg_loc,
            ncol=np.max([sm.L+2, n_unique_species]),
            fontsize="small",)
        
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.5)

    axes[0].set_xlim(x_lims)
    axes[0].set_ylim(y_spec_lims)
    axes[1].set_ylim(y_theta_linear_lims)
    axes[-1].set_ylim(y_s2_lims)

    axes[0].set_ylabel(r"Flux (Norm.)")
    axes[1].set_ylabel(r"$\theta_{\rm Linear}$")
    axes[-1].set_ylabel(r"Scatter")

    axes[0].yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    axes[0].yaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

    for ax in axes[1:-1]:
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.05))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.025))

    axes[-1].xaxis.set_major_locator(plticker.MultipleLocator(base=x_ticks[0]))
    axes[-1].xaxis.set_minor_locator(plticker.MultipleLocator(base=x_ticks[1]))

    plt.xlabel(r"Wavelength (${\rm \AA}$)")
    
    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "theta_coefficients_{}{}.pdf".format(fn_label, fn_suffix))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_spectra_comparison(
    sm,
    obs_join,
    fluxes,
    bad_px_masks,
    labels_all,
    source_ids,
    y_offset=1.8,
    x_lims=(5400,7000),
    x_ticks=(200,100),
    fn_label="",
    data_label="",
    star_name_col="simbad_name",
    sort_col_name=None,
    do_reverse_sort=True,
    do_plot_eps=False,
    fig_size=(12,8),
    data_plot_label="Observed",
    data_plot_colour="k",
    bp_rp_col="BP_RP_dr3",
    fluxes_2=None,
    fluxes_2_plot_label="",
    fluxes_2_plot_colour="",
    do_plot_galah_bands=False,
    n_leg_col=2,
    plot_folder="plots/",):
    """Plot a set of observed spectra against their Cannon generated spectra
    equivalents.

    Parameters
    ----------
    TODO

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    # Intialise
    plt.close("all")

    # If plotting diagnostic plot, use custom size
    if fn_label == "d":
        n_inches = len(source_ids) * 2# / 3
        fig, ax = plt.subplots(1, 1, figsize=(12, n_inches))
    else:
        fig, ax = plt.subplots(1, 1, figsize=fig_size,)
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    # We want to sort based on a given column, but it's tricky since we have
    # multiple data stuctures. So what we'll do is to use the sort order
    # indices as the y axis offsets.
    if sort_col_name is not None and sort_col_name in obs_join.columns.values:
        # First reduce obs_join down to just the selected source_ids
        selected_mask = np.isin(obs_join.index, source_ids)

        sorted_indices = np.argsort(
            obs_join[selected_mask][sort_col_name].values)

        if do_reverse_sort:
            sorted_indices = sorted_indices[::-1]

        obs_join = obs_join[selected_mask].iloc[sorted_indices]
        fluxes = fluxes[selected_mask][sorted_indices]
        bad_px_masks = bad_px_masks[selected_mask][sorted_indices]
        labels_all = labels_all[selected_mask][sorted_indices]
        source_ids = obs_join.index.values

        if fluxes_2 is not None:
            fluxes_2 = fluxes_2[selected_mask][sorted_indices]

    # Mask out emission and telluric regions
    pplt.shade_excluded_regions(
        wave=sm.wavelengths,
        bad_px_mask=~sm.adopted_wl_mask,
        axis=ax,
        res_ax=None,
        colour="red",
        alpha=0.25,
        hatch=None)

    if do_plot_galah_bands:
        bands = {
            "blue":((4718, 4903), "dodgerblue"),
            "green":((5649,5873), "forestgreen"),
            "red":((6481, 6739), "tomato"),}
        
        for band in bands.keys():
            banded_region = np.logical_and(
                sm.wavelengths > bands[band][0][0],
                sm.wavelengths < bands[band][0][1],
            )
            pplt.shade_excluded_regions(
                wave=sm.wavelengths,
                bad_px_mask=banded_region,
                axis=ax,
                res_ax=None,
                colour=bands[band][1],
                alpha=0.2,
                hatch=None)

    # Do bad px masking
    masked_spectra = fluxes.copy()
    masked_spectra[bad_px_masks] = np.nan

    if fluxes_2 is not None:
        fluxes_2_masked = fluxes_2.copy()
        fluxes_2_masked[bad_px_masks] = np.nan

    # For every star in source_ids, plot blue and red spectra
    for star_i, source_id in enumerate(source_ids):
        # Get the index of the particular benchmark
        bm_i = int(np.argwhere(obs_join.index == source_id))

        # Generate a model spectrum (with nans for our excluded regions)
        labels = labels_all[bm_i]

        spec_gen = np.full(fluxes.shape[1], np.nan)
        spec_gen[sm.adopted_wl_mask] = sm.generate_spectra(labels)

        # Plot observed spectrum
        ax.plot(
            sm.wavelengths,
            masked_spectra[bm_i] + star_i*y_offset,
            linewidth=0.2,
            c=data_plot_colour,
            label=data_plot_label,)

        # Plot model spectrum
        ax.plot(
            sm.wavelengths,
            spec_gen + star_i*y_offset,
            linewidth=0.2,
            c="tomato",
            label=r"$\it{Cannon}$",)
        
        # [Optional] Plot third set of spectra
        if fluxes_2 is not None:
            ax.plot(
                sm.wavelengths,
                fluxes_2_masked[bm_i] + star_i*y_offset,
                linewidth=0.2,
                c=fluxes_2_plot_colour,
                label=fluxes_2_plot_label,)

        # Label spectrum
        star_txt = (
            r"{}, $T_{{\rm eff}}={:0.0f}\,$K, "
            r"[Fe/H]$ = ${:+.2f}, $(BP-RP)={:0.2f}$")
        star_txt = star_txt.format(
            obs_join.loc[source_id][star_name_col],
            labels[0],
            labels[2],
            obs_join.loc[source_id][bp_rp_col],)

        ax.text(
            x=x_lims[0]+(x_lims[1]-x_lims[0])/2,
            y=star_i*y_offset+1.6,
            s=star_txt,
            horizontalalignment="center",
        )

        # Only plot one set of legend items
        if star_i == 0:
            leg = ax.legend(loc="upper right", ncol=n_leg_col,)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(1.5)
    
    ax.set_yticks([])
    ax.set_xlim(x_lims)
    ax.set_ylim((0, star_i*y_offset+2.4))

    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=x_ticks[0]))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=x_ticks[1]))

    ax.set_xlabel(r"Wavelength (${\rm \AA}$)")
    plt.tight_layout()

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_spectra_comp_{}_{}".format(
            data_label, fn_label).replace("__", "_"))

    plt.savefig("{}.pdf".format(plot_fn))

    # Don't plot a PNG for the diagnostic plot of all spectra since the image
    # dimensions will be excessively large
    if fn_label != "d":
        plt.savefig("{}.png".format(plot_fn), dpi=300)

    if do_plot_eps:
        plt.savefig("{}.eps".format(plot_fn))



def plot_label_uncertainty_adopted_vs_true_labels(
    sm,
    n_bins=20,
    fn_label="",
    plot_folder="plots/",):
    """Function to plot histograms comparing the adopted and true label
    distributions (+the difference between them) at the conclusion of training
    a label uncertainties model. Currently works for three parameter and four
    parameter models.

    Parameters
    ----------
    sm: Stannon Model
        Trained *label uncertainties* Stannon model.

    n_bins: float, default: 20

    fn_label: string, default: ""
        Label of the filename.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    if sm.model_type != "label_uncertainties":
        raise ValueError("Stannon model must be label_uncertainties.")
    
    labels_adopt = sm.masked_labels.copy()
    labels_true = (sm.true_labels * sm.std_labels + sm.mean_labels).copy()

    delta_labels = labels_adopt - labels_true

    med_dl = np.median(delta_labels, axis=0)
    std_dl = np.std(delta_labels, axis=0)

    plt.close("all")
    if sm.L == 3:
        fig, ((ax_t, ax_g, ax_f), (ax_dt, ax_dg, ax_df)) = \
            plt.subplots(2, 3, figsize=(9, 6))
    else:
        fig, ((ax_t, ax_g, ax_f, ax_ti), (ax_dt, ax_dg, ax_df, ax_dti)) = \
            plt.subplots(2, 4, figsize=(12, 6))

    # -------------------------------------------------------------------------
    # Teff
    # -------------------------------------------------------------------------
    # Plot histogram for adopted and true labels
    _ = ax_t.hist(
        labels_adopt[:,0],
        bins=n_bins,
        alpha=0.5,
        label=r"$T_{\rm eff, adopt}$")
    
    _ = ax_t.hist(
        labels_true[:,0],
        bins=n_bins,
        alpha=0.5,
        label=r"$T_{\rm eff, true}$")
    
    ax_t.legend(loc="best")
    ax_t.set_xlabel(r"$T_{\rm eff}$")

    # Plot histogram of the *difference* between these two sets of labels
    _ = ax_dt.hist(delta_labels[:,0], bins=n_bins, alpha=0.5)
    ax_dt.set_xlabel(r"${\Delta}T_{\rm eff}$")

    # Plot text for median +/- std
    x_lims = ax_dt.get_xlim()
    y_lims = ax_dt.get_ylim()
    text = r"${:0.0f}\pm{:0.0f}\,K$".format(med_dl[0], std_dl[0])
    ax_dt.text(
        x=((x_lims[1]-x_lims[0])/2 + x_lims[0]), 
        y=0.5*(y_lims[1]-y_lims[0])+y_lims[0],
        s=text,
        horizontalalignment="center",)

    # -------------------------------------------------------------------------
    # logg
    # -------------------------------------------------------------------------
    # Plot histogram for adopted and true labels
    _ = ax_g.hist(
        labels_adopt[:,1],
        bins=n_bins,
        alpha=0.5,
        label=r"$\log g_{\rm adopt}$")
    
    _ = ax_g.hist(
        labels_true[:,1],
        bins=n_bins,
        alpha=0.5,
        label=r"$\log g_{\rm true}$")
    
    ax_g.legend(loc="best")
    ax_g.set_xlabel(r"$\log g$")

    # Plot histogram of the *difference* between these two sets of labels
    _ = ax_dg.hist(delta_labels[:,1], bins=n_bins, alpha=0.5)
    ax_dg.set_xlabel(r"$\Delta\log g$")

    # Plot text for median +/- std
    x_lims = ax_dg.get_xlim()
    y_lims = ax_dg.get_ylim()
    text = r"${:0.3f}\pm{:0.3f}\,$dex".format(med_dl[1], std_dl[1])
    ax_dg.text(
        x=((x_lims[1]-x_lims[0])/2 + x_lims[0]), 
        y=0.5*(y_lims[1]-y_lims[0])+y_lims[0],
        s=text,
        horizontalalignment="center",)

    # -------------------------------------------------------------------------
    # [Fe/H]
    # -------------------------------------------------------------------------
    # Plot histogram for adopted and true labels
    _ = ax_f.hist(
        labels_adopt[:,2],
        bins=n_bins,
        alpha=0.5,
        label=r"[Fe/H]$_{adopt}$")
    
    _ = ax_f.hist(
        labels_true[:,2],
        bins=n_bins,
        alpha=0.5,
        label=r"[Fe/H]$_{true}$")
    
    ax_f.legend(loc="best")
    ax_f.set_xlabel(r"[Fe/H]]")

    # Plot histogram of the *difference* between these two sets of labels
    _ = ax_df.hist(delta_labels[:,2], bins=n_bins, alpha=0.5)
    ax_df.set_xlabel(r"$\Delta$[Fe/H]")

    # Plot text for median +/- std
    x_lims = ax_df.get_xlim()
    y_lims = ax_df.get_ylim()
    text = r"${:0.3f}\pm{:0.3f}\,$dex".format(med_dl[2], std_dl[2])
    ax_df.text(
        x=((x_lims[1]-x_lims[0])/2 + x_lims[0]), 
        y=0.5*(y_lims[1]-y_lims[0])+y_lims[0],
        s=text,
        horizontalalignment="center",)

    # -------------------------------------------------------------------------
    # [Ti/H]
    # -------------------------------------------------------------------------
    # Plot histogram for adopted and true labels
    if sm.L == 4:
        _ = ax_ti.hist(
            labels_adopt[:,3],
            bins=n_bins,
            alpha=0.5,
            label=r"[Ti/H]$_{adopt}$")
        
        _ = ax_ti.hist(
            labels_true[:,3],
            bins=n_bins,
            alpha=0.5,
            label=r"[Ti/H]$_{true}$")
        
        ax_ti.legend(loc="best")
        ax_ti.set_xlabel(r"[Ti/H]]")

        # Plot histogram of the *difference* between these two sets of labels
        _ = ax_dti.hist(delta_labels[:,3], bins=n_bins, alpha=0.5)
        ax_dti.set_xlabel(r"$\Delta$[Ti/H]")

        # Plot text for median +/- std
        x_lims = ax_dti.get_xlim()
        y_lims = ax_dti.get_ylim()
        text = r"${:0.3f}\pm{:0.3f}\,$dex".format(med_dl[3], std_dl[3])
        ax_dti.text(
            x=((x_lims[1]-x_lims[0])/2 + x_lims[0]), 
            y=0.5*(y_lims[1]-y_lims[0])+y_lims[0],
            s=text,
            horizontalalignment="center",)

    # -------------------------------------------------------------------------
    # Tidy up and save
    # -------------------------------------------------------------------------
    plt.tight_layout()
    
    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "adopted_vs_true_label_hists{}.pdf".format(fn_label))

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_delta_cannon_vs_marcs(
    obs_join,
    sm,
    fluxes_marcs_norm,
    delta_thresholds=(0.1, 0.05, 0.02),
    fig_size=(12,3),
    plot_smooth_interpolated_fluxes=False,
    s=None,
    plot_folder="plots/",):
    """Plots a series of comparisons of Cannon vs MARCS spectra below the
    provided set of fractional flux tolerances.

    Parameters
    ----------
    obs_join: pandas DataFrame
        Dataframe of stellar properties.
    
    sm: Stannon object
        Trained Stannon model.

    fluxes_marcs_norm: 2D float array
        Normalised array of MARCS fluxes of shape [N_benchmark, N_lambda] 
        corresponding to the benchmarks used to train the Cannon model.

    delta_thresholds: float array, default: (0.1, 0.05, 0.02)
        Fractional flux thresholds to plot, where each value results in a
        subplot panel. For instance, delta_thresholds = (0.1, 0.05, 0.02) plots
        three panels showing only those wavelengths with fluxes within 10%, 5%,
        and 2%.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    # Sort our benchmarks by their BP-RP colour
    is_cannon_benchmark = obs_join["is_cannon_benchmark"].values

    bprp = obs_join[is_cannon_benchmark]["BP_RP_dr3"].values
    bprp_ii = np.argsort(bprp)

    spec_cannon_all = np.vstack(
        [sm.generate_spectra(sm.training_labels[i]) for i in range(sm.S)])

    # Make sure gaps due to telluric regions are still present
    masked_cannon_fluxes = np.full_like(sm.training_data, np.nan)
    masked_cannon_fluxes[:, sm.adopted_wl_mask] = spec_cannon_all

    # Sort
    cannon_fluxes_sorted = masked_cannon_fluxes[bprp_ii]
    marcs_fluxes_sorted = fluxes_marcs_norm[bprp_ii]

    # Compute the fractional flux difference
    delta = (cannon_fluxes_sorted - marcs_fluxes_sorted) / cannon_fluxes_sorted

    #import pdb; pdb.set_trace()

    # TODO
    if plot_smooth_interpolated_fluxes:
        n_bprp = sm.S
        bp_rp_sampled = np.linspace(np.min(bprp), np.max(bprp), n_bprp)

        delta_new = np.full_like(delta, np.nan)

        # Interpolate each spectral pixel
        for px_i in range(delta.shape[1]):
            tck_s = splrep(bprp[bprp_ii], delta[:,px_i], s=s)
            yy = BSpline(*tck_s)(bp_rp_sampled)
            delta_new[:,px_i] = yy
        
        delta = delta_new

    plt.close("all")
    fig, axes = plt.subplots(
        len(delta_thresholds),
        figsize=fig_size,
        sharex=True)

    # Plot one panel per provided flux threshold
    for dt_i, (dt, ax) in enumerate(zip(delta_thresholds, axes)):
        # Create a mask of the excluded regions to plot that is red where we
        # don't have Cannon spectra, and nan otherwise
        excl = np.isnan(cannon_fluxes_sorted)
        
        rgb_mask = np.array((
            excl*np.full_like(sm.training_data, 255),
            excl*np.full_like(sm.training_data, 0),
            excl*np.full_like(sm.training_data, 0)))
        rgb_mask = np.transpose(rgb_mask, (1,2,0))

        ax.imshow(rgb_mask,
            aspect="auto",
            extent=(3500, 7000,bprp.max(), bprp.min()),
            interpolation="none")

        # Plot data
        dm = np.abs(delta) > dt
        delta_masked = delta.copy()
        delta_masked[dm] = np.nan
        
        ss = ax.imshow(
            delta_masked*100,
            aspect="auto",
            extent=(3500, 7000,bprp.max(), bprp.min()),
            interpolation="none",
            cmap="PRGn")
        
        # Other plot formatting
        ax.set_xlim(4000,7000)
        ax.set_title(
            r"$\Delta{{\rm Flux}} < $ {:0.0f}%".format(dt*100),
            fontsize="small")

        fmt = lambda x, pos: r'${:+2.1f}$'.format(x)
        cb = fig.colorbar(
            ss, ax=ax, pad=0.01, fraction=0.05, aspect=5, format=fmt,)
        cb.ax.minorticks_on()
        cb.set_label(r"$\Delta{\rm Flux}$ (%)")
        
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=100))

        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))

        if dt_i < len(delta_thresholds) - 1:
            ax.set_xticks([])

    # Final formatting + saving
    fig.subplots_adjust(
        wspace=0.1, hspace=0.5, right=0.98, left=0.05, top=0.90, bottom=0.20)
    fig.supylabel(r"$BP-RP$")
    ax.set_xlabel("Wavelength")

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_vs_marcs_delta_flux")

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_scatter_histogram_comparison(
    sm1,
    sm2,
    sm_1_label,
    sm_2_label,
    n_bins=250,
    hist_bin_lims=(0, 0.0005),
    plot_folder="plots/",):
    """Plots a histogram comparison of the fitted model scatters for two 
    different trained Cannon models.

    Parameters
    ----------
    sm1, sm2: Stannon object
        Trained Stannon objects to compare.
    
    sm_1_label, sm_2_label: str
        Model labels to plot on legend.

    n_bins: int, default: 250
        Number of histogram bins.

    his_bin_lims: foat tuple, default (0, 0.0005)
        Lower and upper scatter limits to consider.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    bins = np.linspace(hist_bin_lims[0], hist_bin_lims[1], n_bins)
    plt.close("all")

    fig, ax = plt.subplots(1, figsize=(12, 4))
    _ = ax.hist(sm1.s2, bins=bins, label=sm_1_label, alpha=0.5)
    _ = ax.hist(sm2.s2, bins=bins, label=sm_2_label, alpha=0.5)
    ax.legend()
    ax.set_xlim(-0.000005, hist_bin_lims[1])

    plt.tight_layout()

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "cannon_model_scatter_comparison_hist",)

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)


def plot_abundance_trend_recovery(
    obs_join,
    vf05_full_file,
    vf05_sampled_file,
    vf05_teff_lim=4500,
    vf05_logg_lim=3.0,
    vf05_feh_lim=-1.0,
    X_Fe_lims=(-0.275,0.45),
    show_offset=True,
    X_Fe_ticks=(0.2,0.1,0.2,0.1),
    plot_folder="plots/",):
    """Function to compare literature [Ti/Fe] to that predicted from GALAH-Gaia
    chemodynamic relations, and overplot the adopted sample of binary benchmark
    FGK primaries. The literature sample is Valenti & Fischer 2005.

    Note: Currently only [Ti/Fe] is supported.

    Parameters
    ----------
    obs_join: pandas DataFrame
        Dataframe of stellar properties.

    vf05_full_file: str
        Filepath to Valenti & Fischer 2005 data with Gaia DR3 crossmatch.

    vf05_sampled_file: str
        Filepath to sampled/predicted parameters for VF05 stars.
    
    vf05_teff_lim: int, default: 4500
        Lower temperature limit to adopt when plotting VF05 stars.

    vf05_logg_lim: float, default: 3.0
        Lower logg limit to adopt when plotting VF05 stars.

    vf05_feh_lim: float, default: -1.0
        Lower [Fe/H] limit to adopt when plotting VF05 stars.

    X_Fe_lims: float tuple, default: (-0.275,0.45)
        Lower and upper limits to adopt for [X/Fe] when plotting.

    show_offset: bool, default: True
        Whether to print offset +/- std on plot.

    X_Fe_ticks: float tuple, default: (0.2,0.1,0.2,0.1)
        [X/Fe] major and minor ticks for each axis.

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    # Import VF05 full file
    vf05_full = pd.read_csv(vf05_full_file, sep="\t", dtype={"source_id":str})
    vf05_full.set_index("source_id", inplace=True)
    
    # Import VF05 sampled file
    vf05_sampled = pd.read_csv(
        vf05_sampled_file,
        delim_whitespace=True,
        dtype={"GaiaID":str})
    
    vf05_sampled.rename(columns={"GaiaID":"source_id"}, inplace=True)
    vf05_sampled.set_index("source_id", inplace=True)

    # Join both these together
    vf05_join = vf05_full.join(vf05_sampled, "source_id")

    # Create a mask on Teff and logg to match the valid range of the mapping
    in_range = np.logical_and(
        vf05_join["[Fe/H]"].values > vf05_feh_lim,
        np.logical_and(
            vf05_join["Teff"].values > vf05_teff_lim,
            vf05_join["log(g)"].values > vf05_logg_lim))

    n_stars = np.sum(~np.isnan(vf05_join["[Fe/H]"].values[in_range]))
    print("{} stars".format(n_stars))

    # Make plot
    plt.close("all")
    fig, axis = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5)

    # TODO: proper uncertainties?
    sigmas = np.zeros_like(vf05_join["[Ti/Fe]_true"].values[in_range])

    axis, resid_ax = pplt.plot_std_comp_generic(
        fig=fig,
        axis=axis,
        lit=vf05_join["[Ti/Fe]_true"].values[in_range],
        e_lit=sigmas,
        fit=vf05_join["[Ti/Fe]_pred"].values[in_range],
        e_fit=sigmas,
        colour=vf05_join["[Fe/H]"].values[in_range],
        fit_label=r"[Ti/Fe] (Predicted)",
        lit_label=r"[Ti/Fe] (Valenti & Fischer 2005)",
        cb_label=r"[Fe/H]",
        x_lims=X_Fe_lims,
        y_lims=X_Fe_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=X_Fe_ticks,
        return_axes=True,
        scatter_label="Valenti & Fischer 2005",)

    # Print offset
    delta_Ti_Fe_vf05 = (vf05_join["[Ti/Fe]_true"].values[in_range]
                        - vf05_join["[Ti/Fe]_pred"].values[in_range])
    
    print("Δ[Ti/Fe] = {:+0.3f} +/- {:0.3f}".format(
        np.nanmedian(delta_Ti_Fe_vf05), np.nanstd(delta_Ti_Fe_vf05)))

    # Plot the overlapping points
    feh_min = np.nanmin(vf05_join["[Fe/H]"].values[in_range])
    feh_max = np.nanmax(vf05_join["[Fe/H]"].values[in_range])
    norm = plt.Normalize(feh_min, feh_max)

    is_binary = obs_join["is_cpm"].values

    _ = axis.scatter(
        x=obs_join[is_binary]["label_adopt_Ti_Fe"].values,
        y=obs_join[is_binary]["Ti_Fe_monty"].values,
        c=obs_join[is_binary]["label_adopt_feh"].values,
        norm=norm,
        edgecolor="k",
        linewidths=1.5,
        label="Observed")
    
    delta_Ti_Fe = (obs_join[is_binary]["label_adopt_Ti_Fe"].values
                    - obs_join[is_binary]["Ti_Fe_monty"].values)
    
    _ = resid_ax.scatter(
        x=obs_join[is_binary]["label_adopt_Ti_Fe"].values,
        y=delta_Ti_Fe,
        c=obs_join[is_binary]["label_adopt_feh"].values,
        norm=norm,
        edgecolor="k",
        linewidths=1.5,)

    resid_ax.set_ylabel(r"${\rm lit}-{\rm pred}$")

    resid_ax.yaxis.set_major_formatter(
        plticker.StrMethodFormatter(r"${x:+.1f}$"))

    #leg = axis.legend(loc="best")
    plt.tight_layout()

    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(
        plot_folder,
        "vf05_Ti_Fe_vs_predicted")

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)