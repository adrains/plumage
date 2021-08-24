"""Plotting functions related to Stannon
"""
import numpy as np
import matplotlib.pyplot as plt
import plumage.plotting as pplt
import matplotlib.ticker as plticker

def plot_label_recovery(
    label_values,
    e_label_values,
    label_pred,
    e_label_pred,
    teff_lims=(2800,4500),
    logg_lims=(4.4,5.4),
    feh_lims=(-0.6,0.75),
    teff_axis_step=200,
    logg_axis_step=0.2,
    feh_axis_step=0.25,
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
    fig, (ax_teff, ax_logg, ax_feh) = plt.subplots(1,3)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5)

    # Temperatures
    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_teff,
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
    
    # Gravity
    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_logg,
        lit=label_values[:,1],
        e_lit=e_label_values[:,1],
        fit=label_pred[:,1],
        e_fit=e_label_pred[:,1],
        colour=label_values[:,2],
        fit_label=r"$\log g$ (Cannon)",
        lit_label=r"$\log g$ (Literature)",
        cb_label="[Fe/H] (Literature)",
        x_lims=logg_lims,
        y_lims=logg_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=logg_ticks,)
    
    # [Fe/H]]
    pplt.plot_std_comp_generic(
        fig=fig,
        axis=ax_feh,
        lit=label_values[:,2],
        e_lit=e_label_values[:,2],
        fit=label_pred[:,2],
        e_fit=e_label_pred[:,2],
        colour=label_values[:,0],
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
    fig.savefig("paper/cannon_param_recovery{}.png".format(fn_suffix))


def plot_cannon_cmd(
    benchmark_colour,
    benchmark_mag,
    benchmark_feh,
    science_colour,
    science_mag,
    x_label=r"$B_P-R_P$",
    y_label=r"$M_{K_S}$",):
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
    plt.savefig("paper/cannon_cmd.png")
    plt.savefig("paper/cannon_cmd.pdf")