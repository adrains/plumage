"""Plotting functions related to Stannon
"""
import numpy as np
import matplotlib.pyplot as plt
import plumage.plotting as pplt

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