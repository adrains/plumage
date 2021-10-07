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
    fig.savefig("paper/cannon_param_recovery{}.png".format(fn_suffix), dpi=200)


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
        dpi=200)


def plot_cannon_cmd(
    benchmark_colour,
    benchmark_mag,
    benchmark_feh,
    science_colour,
    science_mag,
    x_label=r"$B_P-R_P$",
    y_label=r"$M_{K_S}$",
    highlight_mask=None,
    highlight_mask_label="",):
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

    # Plot science targets
    scatter = axis.scatter(
        science_colour, 
        science_mag, 
        marker="o",
        edgecolor="black",#"#ff7f0e",
        facecolors="none",
        zorder=2,
        alpha=0.6,
        label="Science",)

    # If we've been given a highlight mask, plot for diagnostic reasons
    if highlight_mask is not None:
        scatter = axis.scatter(
            benchmark_colour[highlight_mask],
            benchmark_mag[highlight_mask],
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
    label="",
    fn_suffix="",):
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

    # Plot first order coefficients
    axes[1].plot(sm.masked_wl, sm.theta[:,1]*teff_scale, linewidth=0.5, 
        label=r"$T_{\rm eff} \times$" + "{:0.1f}".format(teff_scale))
    axes[1].plot(sm.masked_wl, sm.theta[:,2], linewidth=0.5, label=r"$\log g$")
    axes[1].plot(sm.masked_wl, sm.theta[:,3], linewidth=0.5, label="[Fe/H]")

    # And first order abundance coefficients if we have it
    if sm.L > 3:
        for abundance_i, abundance in enumerate(sm.label_names[3:]):
            label_i = 4 + abundance_i
            
            abundance_label = "[{}/H]".format(abundance.split("_")[0])

            axes[1].plot(
                sm.masked_wl,
                sm.theta[:,label_i],
                linewidth=0.5,
                label=abundance_label,)

    axes[1].hlines(0, 3400, 7100, linestyles="dashed", linewidth=0.1)

    # Plot scatter
    axes[2].plot(sm.masked_wl, sm.s2, linewidth=0.25,)

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

    leg = axes[1].legend()

    # Update width of legend objects
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    axes[0].set_ylabel(r"Flux")
    axes[1].set_ylabel(r"$\theta_{1-3}$")
    axes[2].set_ylabel(r"Scatter")

    axes[2].xaxis.set_major_locator(plticker.MultipleLocator(base=x_ticks[0]))
    axes[2].xaxis.set_minor_locator(plticker.MultipleLocator(base=x_ticks[1]))

    plt.xlabel("Wavelength (A)")
    
    plt.tight_layout()
    plt.savefig("paper/theta_coefficients_{}_{}.pdf".format(label, fn_suffix))
    plt.savefig("paper/theta_coefficients_{}_{}.png".format(label, fn_suffix),
        dpi=200)


def plot_spectra_comparison(
    sm,
    obs_join,
    source_ids,
    y_offset=1.8,
    x_lims=(5400,7000),
    x_ticks=(200,100),
    fn_label="",):
    """Plot a set of observed spectra against their Cannon generated spectra
    equivalents.
    """
    # Intialise
    plt.close("all")

    # If plotting diagnostic plot, use custom size
    if fn_label == "d":
        n_inches = len(source_ids) * 2 / 3
        fig, ax = plt.subplots(1, 1, figsize=(12, n_inches))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

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
    masked_spectra = sm.training_data.copy()
    masked_spectra[sm.bad_px_mask] = np.nan

    # For every star in source_ids, plot blue and red spectra
    for star_i, source_id in enumerate(source_ids):
        # Get the index of the particular benchmark
        bm_i = int(np.argwhere(obs_join.index == source_id))

        # Generate a model spectrum (with nans for our excluded regions)
        labels = sm.training_labels[bm_i]

        spec_gen = np.full(sm.training_data.shape[1], np.nan)
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
            label="Model",)

        # Label spectrum
        star_txt = r"{}, $T_{{\rm eff}}={:0.0f}\,$K, [Fe/H]$ = ${:+.2f}"
        star_txt = star_txt.format(
            obs_join.loc[source_id]["simbad_name"],
            labels[0],
            labels[2])

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

    ax.set_xlabel("Wavelength (A)")
    plt.tight_layout()

    plt.savefig("paper/cannon_spectra_comp_{}.pdf".format(fn_label))
    plt.savefig("paper/cannon_spectra_comp_{}.png".format(fn_label), dpi=200)