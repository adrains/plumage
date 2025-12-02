"""Plotting functions for working with MIKE spectra.
"""
import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import astropy.constants as const
from numpy.polynomial.polynomial import Polynomial

def plot_flux_calibration(
    fit_dict,
    plot_folder,
    plot_label,
    fig_size=(18,8),
    clip_edge_px=True,
    edge_px_to_clip=10,):
    """Plots an overview diagnostic  of the flux calibration for a single
    target. The plot has six panels:
        1) Stellar and telluric (H2O, O2) transmission.
        2) Observatory extinction.
        3) Low-resolution flux calibrated reference spectrum.
        4) Raw MIKE spectrum.
        5) Fitted transfer functions for each spectral order.
        6) Fluxed MIKE spectrum.

    PDF of figure is saved as <plot_folder>/flux_cal_result_{plot_label>.pdf.

    Parameters
    ----------
    fit_dict: dictionary
        Dictionary as output from from scipy.optimize.least_squares, with added
        keys: ['scale_H2O', 'scale_O2', 'poly_coef', 'wave_obs_2D',
        'spec_obs_2D', 'spec_synth_2D', 'spec_fluxed_2D', 'extinction_2D',
        'tau_H2O_2D', 'tau_O2_2D'].

    plot_folder: str
        File path to save our diagnostic plot to.

    plot_label: str
        Label for the plot, e.g. night of observation and target name.

    fig_size: float tuple, default: (18,8)
        Size of the figure in inches.

    clip_edge_px: boolean, default: True
        Whether to clip the edges of obseved MIKE spectra to not plot bad px.

    edge_px_to_clip: int, default: 10
        Number of edge pixels to not plot, per clip_edge_px.
    """
    # Unpack dict
    scale_H2O = fit_dict["scale_H2O"]
    scale_O2 = fit_dict["scale_O2"] 
    poly_coef = fit_dict["poly_coef"]
    wave_means = fit_dict["wave_means"]
    wave_obs_2D = fit_dict["wave_obs_2D"]
    spec_obs_2D = fit_dict["spec_obs_2D"]
    sigma_obs_2D = fit_dict["sigma_obs_2D"]
    wave_obs_2D_broad = fit_dict["wave_obs_2D_broad"]
    spec_obs_2D_broad = fit_dict["spec_obs_2D_broad"]
    sigma_obs_2D_broad = fit_dict["sigma_obs_2D_broad"]
    spec_synth_2D = fit_dict["spec_synth_2D"]
    spec_fluxed_2D = fit_dict["spec_fluxed_2D"]
    extinction_2D = fit_dict["extinction_2D"]
    tau_H2O_2D = fit_dict["tau_H2O_2D"]
    tau_O2_2D = fit_dict["tau_O2_2D"]
    scale_H2O = fit_dict["scale_H2O"]
    scale_O2 = fit_dict["scale_O2"]
    telluric_corr_spec_2D = fit_dict["telluric_corr_spec_2D"]

    (n_order, n_px) = wave_obs_2D.shape

    # [Optional] Clip edges of MIKE spectra when plotting
    plot_mask = np.full((n_px), True)
    
    if clip_edge_px:
        plot_mask[:edge_px_to_clip] = False
        plot_mask[-edge_px_to_clip:] = False

    # Calculate per-order SNR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) 
        med_snr = np.nanmedian(spec_obs_2D / sigma_obs_2D, axis=1)

    # -------------------------------------------------------------------------
    # Diagnostic plotting
    # -------------------------------------------------------------------------
    plt.close( "all")
    fig, (ax_temp, ax_ext, ax_flux, ax_obs, ax_comp, ax_tf, ax_corr) = \
        plt.subplots(nrows=7, sharex=True, figsize=fig_size)
    
    fig.subplots_adjust(
        left=0.04,
        bottom=0.075,
        right=0.995,
        top=0.95,
        #wspace=0.05,
        hspace=0.075)

    for order_i in range(n_order):
        wave_ith = wave_obs_2D[order_i]
        spec_ith = spec_obs_2D[order_i]

        wave_broad_ith = wave_obs_2D_broad[order_i]
        spec_broad_ith = spec_obs_2D_broad[order_i]

        synth_ith = spec_synth_2D[order_i]
        flux_ith = spec_fluxed_2D[order_i]
        extinction_ith = extinction_2D[order_i]
        
        trans_H20 = np.exp(-scale_H2O * tau_H2O_2D[order_i])
        trans_O2 = np.exp(-scale_O2 * tau_O2_2D[order_i])

        tf_poly = Polynomial(poly_coef[order_i])
        smooth_tf = tf_poly(wave_ith-wave_means[order_i])
        
        # --------
        # Panel #1: cont norm synthetic stellar and telluric (O2, and H2) spec
        ax_temp.plot(
            wave_broad_ith,
            synth_ith,
            linewidth=0.5,
            c="k",
            alpha=0.8,
            label="Star" if order_i == 0 else None,)
        
        ax_temp.plot(
            wave_broad_ith,
            trans_H20,
            linewidth=0.5,
            c="maroon",
            alpha=0.8,
            label="H2O ({:0.2f})".format(scale_H2O) if order_i == 0 else None,)
        
        ax_temp.plot(
            wave_broad_ith,
            trans_O2,
            linewidth=0.5,
            c="b",
            alpha=0.8,
            label="O2 ({:0.2f})".format(scale_O2) if order_i == 0 else None,)

        # --------
        # Panel #2: atmospheric extinction
        ax_ext.plot(
            wave_broad_ith,
            extinction_ith,
            linewidth=0.5,
            c="k",
            alpha=1.0,
            label="Observatory Atmospheric Extinction" if order_i == 0 else None,)

        # --------
        # Panel #3: fluxed spectrum
        ax_flux.plot(
            wave_broad_ith,
            flux_ith,
            linewidth=0.5,
            c="g",
            label="Flux Reference" if order_i == 0 else None,)
        
        # --------
        # Panel #4: observed spectrum (real + smoothed)
        ax_obs.plot(
            wave_ith[plot_mask],
            spec_ith[plot_mask],
            linewidth=0.5,
            c="k",
            alpha=0.8,
            label="MIKE Spectrum (raw)" if order_i == 0 else None,)
        
        ax_obs.plot(
            wave_broad_ith,
            spec_broad_ith,
            linewidth=0.5,
            c="r",
            alpha=0.8,
            label="MIKE Spectrum (raw, smoothed)" if order_i == 0 else None,)
        
        # SNR annotations
        if not np.isnan(med_snr[order_i]):
            x_mean = np.nanmean(wave_ith[plot_mask])
            y_max = np.nanmax(spec_ith[plot_mask])

            if ~np.isnan(x_mean) and ~np.isnan(y_max):
                ax_obs.text(
                    x=x_mean,
                    y=y_max*1.05,
                    s="{:0.0f}".format(med_snr[order_i]),
                    horizontalalignment="center",
                    fontsize="x-small",)
        
        # --------
        # Panel #5: comparison
        ax_comp.plot(
            wave_broad_ith,
            flux_ith,
            linewidth=0.5,
            c="g",
            label="Flux Reference" if order_i == 0 else None,)

        ax_comp.plot(
            wave_obs_2D_broad[order_i],
            telluric_corr_spec_2D[order_i],
            linewidth=0.5,
            c="b",
            label="'corrected' broadened spectrum" if order_i == 0 else None,)

        # --------
        # Panel #5: fitted polynomials overplotted on 'corrected' spectra
        ax_tf.plot(
            wave_ith,
            1/smooth_tf,
            linewidth=0.5,
            c="r",
            label="Fitted Transfer Function" if order_i == 0 else None,)
        
        # 'Corrected' spectrum
        tf_poly = Polynomial(poly_coef[order_i])
        tf = tf_poly(wave_obs_2D_broad[order_i] - wave_means[order_i])

        ax_tf.plot(
            wave_obs_2D_broad[order_i],
            telluric_corr_spec_2D[order_i] / tf,
            linewidth=0.5,
            c="b",
            label="'corrected' broadened spectrum" if order_i == 0 else None,)
        
        # --------
        # Panel #6: flux calibrated spectrum
        ax_corr.plot(
            wave_ith[plot_mask],
            spec_ith[plot_mask]*smooth_tf[plot_mask],
            linewidth=0.5,
            label="MIKE Spectrum (Fluxed)" if order_i == 0 else None,)
    
    # Legends
    loc = "lower center"
    fontsize = "small"
    ncol = 3
    ax_temp.legend(loc=loc, fontsize=fontsize, ncol=ncol,)
    ax_ext.legend(loc=loc, fontsize=fontsize, ncol=ncol,)
    ax_flux.legend(loc=loc, fontsize=fontsize, ncol=ncol,)
    ax_obs.legend(loc=loc, fontsize=fontsize, ncol=ncol,)
    ax_comp.legend(loc=loc, fontsize=fontsize, ncol=ncol,)
    ax_tf.legend(loc=loc, fontsize=fontsize, ncol=ncol,)
    ax_corr.legend(loc=loc, fontsize=fontsize, ncol=ncol,)

    # Axis Labels
    ax_temp.set_ylabel(r"Transmission", fontsize=fontsize)
    ax_ext.set_ylabel(r"$k(\lambda)$", fontsize=fontsize)
    ax_flux.set_ylabel(
        r"Flux (erg$\cdot$s$^{-1}\cdot$cm$^{-1}$Å$^{-1}$)", fontsize=fontsize)
    ax_obs.set_ylabel(r"Counts", fontsize=fontsize)
    ax_comp.set_ylabel(
        r"Flux (erg$\cdot$s$^{-1}\cdot$cm$^{-1}$Å$^{-1}$)", fontsize=fontsize)
    ax_tf.set_ylabel(r"$\times$TF", fontsize=fontsize)
    ax_corr.set_ylabel(
        r"Flux (erg$\cdot$s$^{-1}\cdot$cm$^{-1}$Å$^{-1}$)", fontsize=fontsize)

    ax_corr.set_xlim(
        0.995*np.nanmin(wave_obs_2D), 1.005*np.nanmax(wave_obs_2D))
    ax_corr.set_xlabel("Wavelength (Å)")

    # Title
    fig.suptitle(plot_label.replace("_", ", "))

    plot_fn = os.path.join(
        plot_folder, "flux_cal_result_{}.pdf".format(plot_label))

    plt.savefig(plot_fn)


def plot_cc_rv_diagnostic(
    rv_fit_dict,
    obj_name,
    figsize=(16,4),
    fig_save_path="plots/rv_diagnostics",):
    """Diagnostic plotting function for cross-correlation RV fits with one
    panel displaying the CCF, and another the overlapping spectra.

    Parameters
    ----------
    rv_fit_dict: dict
        Dictionary with keys ['rv', 'bcor', 'rv_steps', 'cross_corrs', 
        'wave_template', 'spec_template', 'calc_template_flux', 
        'wave_telluric', 'trans_telluric', 'wave_2D', 'spec_2D', 'orders', 
        'order_excluded'] as output from spectra_mike.fit_rv().

    obj_name: str
        Name of the stars to use as the title of the plot and when saving.

    figsize: float tuple, default: (16,4)
        Figure size of diagnostic plot.

    fig_save_path: str, default: "plots/rv_diagnostics"
        Path to save diagnostic figure to.
    """
    # -------------------------------------------------------------------------
    # Unpacking dict
    # -------------------------------------------------------------------------
    rv = rv_fit_dict["rv"]
    bcor = rv_fit_dict["bcor"]
    rv_steps = rv_fit_dict["rv_steps"]
    cross_corrs = rv_fit_dict["cross_corrs"]
    calc_template_flux = rv_fit_dict["calc_template_flux"]

    wave_template = rv_fit_dict["wave_template"]
    spec_template = rv_fit_dict["spec_template"]
    
    wave_telluric = rv_fit_dict["wave_telluric"]
    trans_telluric = rv_fit_dict["trans_telluric"]

    wave_2D = rv_fit_dict["wave_2D"]
    spec_2D = rv_fit_dict["spec_2D"]
    orders = rv_fit_dict["orders"]

    order_excluded = rv_fit_dict["order_excluded"]

    # Grab dimensions for convenience
    (n_order, n_px) = wave_2D.shape

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    plt.close("all")
    fig, (cc_axis, spec_axis) = plt.subplots(2,1, figsize=figsize)
    plt.subplots_adjust(hspace=0.4,)

    # Plot cross correlation fit
    cc_axis.plot(rv_steps, cross_corrs, linewidth=0.2)
    cc_axis.set_title(obj_name)
    cc_axis.set_xlabel("RV (km/s)")
    cc_axis.set_ylabel("Cross Correlation")

    # Compute best fit template and plot spectral fit
    template_flux = calc_template_flux(
        wave_template * (1-(rv-bcor)/(const.c.si.value/1000)))

    # Plot per-order science spectrum
    for order_i in range(n_order):
        # Skip missing orders
        if np.nansum(wave_2D[order_i]) == 0:
            continue
        
        # For M-dwarfs with continuum suppression, we want to scale the
        # science flux so it fits over the template for readability.
        wl_mask = np.logical_and(
            wave_template > np.nanmin(wave_2D[order_i]),
            wave_template < np.nanmax(wave_2D[order_i]),)
        continuum_scale = np.mean(spec_template[wl_mask])

        spec_axis.plot(
            wave_2D[order_i],
            spec_2D[order_i] * continuum_scale,
            linewidth=0.5,
            alpha=0.5,
            c="r",
            label="science" if order_i == 0 else None,)

        spec_axis.text(
            np.nanmean(wave_2D[order_i]),
            1.5,
            s="{}".format(orders[order_i]),
            color="r" if order_excluded[order_i] else "k",
            fontsize="x-small",
            horizontalalignment="center",)

    # Template flux
    spec_axis.plot(
        wave_template,
        template_flux,
        linewidth=0.5,
        alpha=0.5,
        c="b",
        label="template")
    
    # Telluric transmission
    spec_axis.plot(
        wave_telluric,
        trans_telluric,
        linewidth=0.5,
        alpha=0.5,
        c="k",
        label="telluric")

    leg = spec_axis.legend(ncol=3)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)
    
    spec_axis.set_xlabel("Wavelength (nm)")
    spec_axis.set_ylabel("Flux (cont norm)")

    spec_axis.set_xlim(np.nanmin(wave_2D)*0.99, np.nanmax(wave_2D)*1.01)
    spec_axis.set_ylim(0,2)
    plt.tight_layout()

    fig_name = os.path.join(fig_save_path, "rv_diagnostic")
    plt.savefig("{}_{}.pdf".format(fig_name, obj_name))
    plt.savefig("{}_{}.png".format(fig_name, obj_name), dpi=200)


def plot_all_cc_rv_diagnostics(
    all_rv_fit_dicts,
    obj_names,
    figsize,
    fig_save_path,):
    """Calls plot_cc_rv_diagnostic() in a loop for many RV fit diagnostics.

    Parameters
    ----------
    all_rv_fit_dicts: list of dicts
        List of dicts with keys ['rv', 'bcor', 'rv_steps', 'cross_corrs', 
        'wave_template', 'spec_template', 'calc_template_flux', 
        'wave_telluric', 'trans_telluric', 'wave_2D', 'spec_2D', 'orders', 
        'order_excluded'] as output from spectra_mike.fit_rv(), of shape 
        [n_star].

    obj_names: 1D str list
        Names of the stars to use as the title of the plot and when saving, of
        shape [n_star].

    figsize: float tuple, default: (16,4)
        Figure size of diagnostic plot.

    fig_save_path: str, default: "plots/rv_diagnostics"
        Path to save diagnostic figure to.
    """
    n_stars = len(all_rv_fit_dicts)

    desc = "Plotting RV diagnostics"

    for star_i in tqdm(range(n_stars), desc=desc, leave=False):
        plot_cc_rv_diagnostic(
            rv_fit_dict=all_rv_fit_dicts[star_i],
            obj_name=obj_names[star_i],
            figsize=figsize,
            fig_save_path=fig_save_path,)