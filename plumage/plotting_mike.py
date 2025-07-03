"""Plotting functions for working with MIKE spectra.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def plot_flux_calibration(fit_dict, plot_folder, plot_label, fig_size=(18,8),):
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
    """
    # Unpack dict
    scale_H2O = fit_dict["scale_H2O"]
    scale_O2 = fit_dict["scale_O2"] 
    poly_coef = fit_dict["poly_coef"]
    wave_obs_2D = fit_dict["wave_obs_2D"]
    spec_obs_2D = fit_dict["spec_obs_2D"]
    spec_synth_2D = fit_dict["spec_synth_2D"]
    spec_fluxed_2D = fit_dict["spec_fluxed_2D"]
    extinction_2D = fit_dict["extinction_2D"]
    tau_H2O_2D = fit_dict["tau_H2O_2D"]
    tau_O2_2D = fit_dict["tau_O2_2D"]

    (n_order, n_px) = wave_obs_2D.shape

    # -------------------------------------------------------------------------
    # Diagnostic plotting
    # -------------------------------------------------------------------------
    plt.close( "all")
    fig, (ax_temp, ax_ext, ax_flux, ax_obs, ax_tf, ax_corr) = plt.subplots(
        nrows=6, sharex=True, figsize=fig_size)

    for order_i in range(n_order):
        wave_ith = wave_obs_2D[order_i]
        spec_ith = spec_obs_2D[order_i]
        synth_ith = spec_synth_2D[order_i]
        flux_ith = spec_fluxed_2D[order_i]
        extinction_ith = extinction_2D[order_i]
        
        trans_H20 = np.exp(-scale_H2O * tau_H2O_2D[order_i])
        trans_O2 = np.exp(-scale_O2 * tau_O2_2D[order_i])

        tf_poly = Polynomial(poly_coef[order_i])
        smooth_tf = tf_poly(wave_ith)
        
        # --------
        # Panel #1: cont norm synthetic stellar and telluric (O2, and H2) spec
        ax_temp.plot(
            wave_ith,
            synth_ith,
            linewidth=0.5,
            c="k",
            alpha=0.8,
            label="Star" if order_i == 0 else None,)
        
        ax_temp.plot(
            wave_ith,
            trans_H20,
            linewidth=0.5,
            c="maroon",
            alpha=0.8,
            label="H2O" if order_i == 0 else None,)
        
        ax_temp.plot(
            wave_ith,
            trans_O2,
            linewidth=0.5,
            c="b",
            alpha=0.8,
            label="O2" if order_i == 0 else None,)

        # --------
        # Panel #2: atmospheric extinction
        ax_ext.plot(
            wave_ith,
            extinction_ith,
            linewidth=0.5,
            c="k",
            alpha=1.0,
            label="Observatory Atmospheric Extinction" if order_i == 0 else None,)

        # --------
        # Panel #3: fluxed spectrum
        ax_flux.plot(
            wave_ith,
            flux_ith,
            linewidth=0.5,
            c="g",
            label="Flux Reference" if order_i == 0 else None,)
        
        # --------
        # Panel #4: fluxed spectrum
        ax_obs.plot(
            wave_ith,
            spec_ith,
            linewidth=0.5,
            c="k",
            alpha=0.8,
            label="MIKE Spectrum (Raw)" if order_i == 0 else None,)
        
        # --------
        # Panel #5: fitted polynomials
        ax_tf.plot(
            wave_ith,
            smooth_tf,
            linewidth=0.5,
            c="r",
            label="Fitted Transfer Function" if order_i == 0 else None,)
        
        # --------
        # Panel #6: flux calibrated spectrum
        ax_corr.plot(
            wave_ith,
            spec_ith*smooth_tf,
            linewidth=0.5,
            label="MIKE Spectrum (Fluxed)" if order_i == 0 else None,)
    
    # Legends
    loc = "lower left"
    ax_temp.legend(loc=loc)
    ax_ext.legend(loc=loc)
    ax_flux.legend(loc=loc)
    ax_obs.legend(loc=loc)
    ax_tf.legend(loc=loc)
    ax_corr.legend(loc=loc)

    # Axis Labels
    ax_temp.set_ylabel(r"Transmission")
    ax_ext.set_ylabel(r"$k(\lambda)$")
    ax_flux.set_ylabel(r"Flux (erg$\cdot$s$^{-1}\cdot$cm$^{-1}$Å$^{-1}$)")
    ax_obs.set_ylabel(r"Counts")
    ax_tf.set_ylabel(r"$\times$TF")
    ax_corr.set_ylabel(r"Flux (erg$\cdot$s$^{-1}\cdot$cm$^{-1}$Å$^{-1}$)")

    ax_corr.set_xlabel("Wavelength (Å)")

    # Scale
    ax_tf.set_yscale("log")

    # Title
    fig.suptitle(plot_label.replace("_", ", "))
    plt.tight_layout()

    plot_fn = os.path.join(
        plot_folder, "flux_cal_result_{}.pdf".format(plot_label))

    plt.savefig(plot_fn)