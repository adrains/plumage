"""Plotting functions related to Stannon
"""
from ctypes import alignment
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plumage.plotting as pplt
import matplotlib.ticker as plticker

def plot_label_recovery(
    label_values,
    e_label_values,
    label_pred,
    e_label_pred,
    obs_join,
    abundance_labels=[],
    teff_lims=(2800,4500),
    logg_lims=(4.4,5.4),
    feh_lims=(-1.0,0.75),
    elinewidth=0.4,
    show_offset=True,
    fn_suffix="",
    title_text="",
    teff_ticks=(500,250,100,50),
    logg_ticks=(0.5,0.25,0.2,0.1),
    feh_ticks=(0.5,0.25,0.5,0.25),):
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
    """
    plt.close("all")

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
        fit_label=r"$T_{\rm eff}$ (K, Cannon)",
        lit_label=r"$T_{\rm eff}$ (K, Literature)",
        cb_label="[Fe/H] (Literature)",
        x_lims=teff_lims,
        y_lims=teff_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,)
    
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
        fit_label=r"$\log g$ (Cannon)",
        lit_label=r"$\log g$ (Literature)",
        cb_label="[Fe/H] (Literature)",
        x_lims=logg_lims,
        y_lims=logg_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=logg_ticks,)
    
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
        fit_label=r"[Fe/H] (Cannon)",
        lit_label=r"[Fe/H] (Literature)",
        cb_label=r"$T_{\rm eff}\,$K (Literature)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,)

    # Save Teff plot
    fig.set_size_inches(12, 3)
    fig.tight_layout()
    fig.savefig("paper/cannon_param_recovery{}.pdf".format(fn_suffix))
    fig.savefig("paper/cannon_param_recovery{}.png".format(fn_suffix), dpi=300)


def plot_label_recovery_per_source(
    label_values,
    e_label_values,
    label_pred,
    e_label_pred,
    obs_join,
    teff_lims=(2800,4500),
    logg_lims=(4.4,5.4),
    feh_lims=(-1.0,0.75),
    elinewidth=0.4,
    show_offset=True,
    fn_suffix="",
    title_text="",
    teff_ticks=(500,250,100,50),
    logg_ticks=(0.5,0.25,0.2,0.1),
    feh_ticks=(0.5,0.25,0.5,0.25),):
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
    """
    plt.close("all")

    # Make plot
    fig, (ax_teff_int, ax_feh_m15, ax_feh_ra12, ax_feh_cpm) = plt.subplots(1,4)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5)

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
        fit_label=r"$T_{\rm eff}$ (K, Cannon)",
        lit_label=r"$T_{\rm eff}$ (K, Interferometry)",
        cb_label="[Fe/H] (Literature)",
        x_lims=teff_lims,
        y_lims=teff_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,)

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
        fit_label=r"[Fe/H] (Cannon)",
        lit_label=r"[Fe/H]] (Mann+15)",
        cb_label=r"$T_{\rm eff}\,$K (Literature)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,)

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
        fit_label=r"[Fe/H] (Cannon)",
        lit_label=r"[Fe/H] (Rojas-Ayala+12)",
        cb_label=r"$T_{\rm eff}\,$K (Literature)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,)

    # CPM [Fe/H]
    feh_mask = ~np.isnan(obs_join["feh_cpm"])

    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_feh_cpm,
        lit=label_values[:,2][feh_mask],
        e_lit=e_label_values[:,2][feh_mask],
        fit=label_pred[:,2][feh_mask],
        e_fit=e_label_pred[:,2][feh_mask],
        colour=label_values[:,0][feh_mask],
        fit_label=r"[Fe/H] (Cannon)",
        lit_label=r"[Fe/H] (CPM)",
        cb_label=r"$T_{\rm eff}\,$K (Literature)",
        x_lims=feh_lims,
        y_lims=feh_lims,
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,)

    # Save plot
    fig.set_size_inches(16, 3)
    fig.tight_layout()
    fig.savefig("paper/cannon_param_recovery_ps{}.pdf".format(fn_suffix))
    fig.savefig("paper/cannon_param_recovery_ps{}.png".format(fn_suffix), dpi=200)


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
    feh_ticks=(0.5,0.25,0.5,0.25),):
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
    """
    plt.close("all")

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

        abundance_label = "[{}/H]".format(abundance.split("_")[0])

        abundance_mask = ~np.isnan(obs_join[abundance])

        pplt.plot_std_comp_generic(
            fig=fig,
            axis=axes[abundance_i],
            lit=label_values[:,label_i][abundance_mask],
            e_lit=e_label_values[:,label_i][abundance_mask],
            fit=label_pred[:,label_i][abundance_mask],
            e_fit=e_label_pred[:,label_i][abundance_mask],
            colour=label_values[:,0][abundance_mask],
            fit_label=r"{} (Cannon)".format(abundance_label),
            lit_label=r"{} (Literature)".format(abundance_label),
            cb_label=r"$T_{\rm eff}\,$K (Literature)",
            x_lims=feh_lims,
            y_lims=feh_lims,
            cmap="magma",
            show_offset=show_offset,
            ticks=feh_ticks,)
    
    # Save plot
    fig.set_size_inches(4*n_abundances, 3)
    fig.tight_layout()
    fig.savefig(
        "paper/cannon_param_recovery_abundance{}.pdf".format(fn_suffix))
    fig.savefig(
        "paper/cannon_param_recovery_abundance{}.png".format(fn_suffix),
        dpi=300)


def plot_cannon_cmd(
    benchmark_colour,
    benchmark_mag,
    benchmark_feh,
    science_colour,
    science_mag,
    x_label=r"$BP-RP$",
    y_label=r"$M_{K_S}$",
    highlight_mask=None,
    highlight_mask_label="",
    bp_rp_cutoff=1.7,):
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
    """
    plt.close("all")
    fig, axis = plt.subplots()

    # Plot benchmarks
    scatter = axis.scatter(
        benchmark_colour, 
        benchmark_mag, 
        zorder=1,
        c=benchmark_feh,
        label="Benchmark",
        alpha=0.9,
        cmap="viridis",
    )

    cb = fig.colorbar(scatter, ax=axis)
    cb.set_label("[Fe/H]")

    # Plot science targets, makingg sure to not plot any science targets beyond
    # the extent of our benchmarks
    scatter = axis.scatter(
        science_colour[science_colour > bp_rp_cutoff],
        science_mag[science_colour > bp_rp_cutoff],
        marker="o",
        edgecolor="black",#"#ff7f0e",
        facecolors="none",
        zorder=2,
        alpha=0.6,
        label="Science",)

    # If we've been given a highlight mask, plot for diagnostic reasons
    if highlight_mask is not None:
        scatter = axis.scatter(
            benchmark_colour[highlight_mask][science_colour > bp_rp_cutoff],
            benchmark_mag[highlight_mask][science_colour > bp_rp_cutoff],
            marker="o",
            edgecolor="red",#"#ff7f0e",
            facecolors="none",
            zorder=4,
            alpha=0.8,
            label=highlight_mask_label,)

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
    plt.savefig("paper/cannon_cmd.png", dpi=200)
    plt.savefig("paper/cannon_cmd.pdf")


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
    y_theta_lims=(-0.1,0.1),
    y_s2_lims=(-0.001,0.01),
    x_ticks=(500,100),
    linewidth=0.5,
    alpha=0.9,
    label="",
    fn_suffix="",
    leg_loc="best",
    line_list=None,):
    """Plot fluxes, values of first order theta coefficients for Teff, logg,
    and [Fe/H], as well as model scatter - all against wavelength.

    TODO
    """
    plt.close("all")
    # Initialise axes
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    axes = axes.flatten()

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

    # Plot first order coefficients
    axes[1].plot(
        wave,
        theta[:,1]*teff_scale,
        linewidth=linewidth,
        alpha=alpha,
        label=teff_label,)

    axes[1].plot(
        wave,
        theta[:,2],
        linewidth=linewidth,
        alpha=alpha,
        label=r"$\log g$")

    axes[1].plot(
        wave,
        theta[:,3],
        linewidth=linewidth,
        label="[Fe/H]")

    # And first order abundance coefficients if we have it
    if sm.L > 3:
        for abundance_i, abundance in enumerate(sm.label_names[3:]):
            label_i = 4 + abundance_i
            
            abundance_label = "[{}/H]".format(abundance.split("_")[0])

            axes[1].plot(
                wave,
                theta[:,label_i],
                linewidth=linewidth,
                alpha=alpha,
                label=abundance_label,)

    axes[1].hlines(0, 3400, 7100, linestyles="dashed", linewidth=0.1)

    # Plot scatter
    axes[2].plot(wave, scatter, linewidth=linewidth,)

    if line_list is not None:
        point_height_spec = 1.5
        text_height_spec = 2

        point_height_scatter = 0.003
        text_height_scatter = 0.004

        line_mask = np.logical_and(
            line_list["wl"] > x_lims[0],
            line_list["wl"] < x_lims[1],)

        for line_i, line_data in line_list[line_mask].iterrows():

            if line_i % 2 == 0:
                offset_spec = (text_height_spec - point_height_spec)/4
                offset_scatter = (text_height_scatter - point_height_scatter)/4
            else:
                offset_spec = 0
                offset_scatter = 0

            # Plot text and arrow for each line
            axes[0].annotate(
                s=line_data["ion"],
                xy=(line_data["wl"], point_height_spec),
                xytext=(line_data["wl"], text_height_spec + offset_spec),
                horizontalalignment="center",
                fontsize=5,
                arrowprops=dict(arrowstyle='->',lw=0.2),)

            axes[2].annotate(
                s=line_data["ion"],
                xy=(line_data["wl"], point_height_scatter),
                xytext=(line_data["wl"], text_height_scatter + offset_scatter),
                horizontalalignment="center",
                fontsize=5,
                arrowprops=dict(arrowstyle='->',lw=0.2),)


    # Now mask out emission and telluric regions
    pplt.shade_excluded_regions(
        wave=sm.wavelengths,
        bad_px_mask=~sm.adopted_wl_mask,
        axis=axes[0],
        res_ax=None,
        colour="red",
        alpha=0.25,
        hatch=None)

    pplt.shade_excluded_regions(
        wave=sm.wavelengths,
        bad_px_mask=~sm.adopted_wl_mask,
        axis=axes[1],
        res_ax=None,
        colour="red",
        alpha=0.25,
        hatch=None)

    pplt.shade_excluded_regions(
        wave=sm.wavelengths,
        bad_px_mask=~sm.adopted_wl_mask,
        axis=axes[2],
        res_ax=None,
        colour="red",
        alpha=0.25,
        hatch=None)

    axes[0].set_xlim(x_lims)
    axes[0].set_ylim(y_spec_lims)
    axes[1].set_ylim(y_theta_lims)
    axes[2].set_ylim(y_s2_lims)

    leg = axes[1].legend(ncol=sm.L, loc=leg_loc,)

    # Update width of legend objects
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    axes[0].set_ylabel(r"Flux")
    axes[1].set_ylabel(r"$\theta_{{1-{:0.0f}}}$".format(sm.L))
    axes[2].set_ylabel(r"Scatter")

    axes[2].xaxis.set_major_locator(plticker.MultipleLocator(base=x_ticks[0]))
    axes[2].xaxis.set_minor_locator(plticker.MultipleLocator(base=x_ticks[1]))

    plt.xlabel("Wavelength (A)")
    
    plt.tight_layout()
    plt.savefig("paper/theta_coefficients_{}{}.pdf".format(label, fn_suffix))
    plt.savefig("paper/theta_coefficients_{}{}.png".format(label, fn_suffix),
        dpi=200)


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
    fig_size=(12,8),):
    """Plot a set of observed spectra against their Cannon generated spectra
    equivalents.
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

    # Mask out emission and telluric regions
    pplt.shade_excluded_regions(
        wave=sm.wavelengths,
        bad_px_mask=~sm.adopted_wl_mask,
        axis=ax,
        res_ax=None,
        colour="red",
        alpha=0.25,
        hatch=None)

    # Do bad px masking
    masked_spectra = fluxes.copy()
    masked_spectra[bad_px_masks] = np.nan

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
            c="k",
            label="Observed",)

        # Plot model spectrum
        ax.plot(
            sm.wavelengths,
            spec_gen + star_i*y_offset,
            linewidth=0.2,
            c="r",
            label="Cannon",)

        # Label spectrum
        star_txt = (
            r"{}, $T_{{\rm eff}}={:0.0f}\,$K, "
            r"[Fe/H]$ = ${:+.2f}, $(BP-RP)={:0.2f}$")
        star_txt = star_txt.format(
            obs_join.loc[source_id][star_name_col],
            labels[0],
            labels[2],
            obs_join.loc[source_id]["Bp-Rp"],)

        ax.text(
            x=x_lims[0]+(x_lims[1]-x_lims[0])/2,
            y=star_i*y_offset+1.6,
            s=star_txt,
            horizontalalignment="center",
        )

        # Only plot one set of legend items
        if star_i == 0:
            leg = ax.legend(loc="upper right", ncol=2,)

            for legobj in leg.legendHandles:
                legobj.set_linewidth(1.5)
    
    ax.set_yticks([])
    ax.set_xlim(x_lims)
    ax.set_ylim((0, star_i*y_offset+2.4))

    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=x_ticks[0]))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=x_ticks[1]))

    ax.set_xlabel(r"Wavelength (${\rm \AA}$)")
    plt.tight_layout()

    fn = "paper/cannon_spectra_comp_{}_{}".format(
        data_label, fn_label).replace("__", "_")

    plt.savefig("{}.pdf".format(fn))
    plt.savefig("{}.png".format(fn), dpi=300)

    if do_plot_eps:
        plt.savefig("{}.eps".format(fn))