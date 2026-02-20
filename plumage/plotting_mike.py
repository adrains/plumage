"""Plotting functions for working with MIKE spectra.
"""
import os
import warnings
import numpy as np
from tqdm import tqdm
import plumage.spectra_mike as sm
import matplotlib.pyplot as plt
import astropy.constants as const
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
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
    wave_mins = fit_dict["wave_mins"]
    wave_maxes = fit_dict["wave_maxes"]
    wave_deltas = wave_maxes - wave_mins
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

        # Compute windowed wavelength scale
        wave_ith_windowed = ((wave_ith - wave_mins[order_i]) 
                             / wave_deltas[order_i] * 2 - 1)

        tf_poly = Polynomial(poly_coef[order_i])
        smooth_tf = tf_poly(wave_ith_windowed)
        
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
        wave_ith_broad_windowed = \
            ((wave_obs_2D_broad[order_i] - wave_mins[order_i])
             / wave_deltas[order_i] * 2 - 1)

        tf_poly = Polynomial(poly_coef[order_i])
        tf = tf_poly(wave_ith_broad_windowed)

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
            c="r" if order_i % 2 == 0 else "k",
            alpha=0.8,
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

    # Scale
    ax_obs.set_yscale("log")
    ax_tf.set_yscale("log")

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
    n_coef = poly_coef.shape[1]
    fig.suptitle(plot_label.replace("_", ", ") + "[order: {}]".format(n_coef))

    plot_fn = os.path.join(
        plot_folder, "flux_cal_result_{}_o{}.pdf".format(plot_label, n_coef))

    plt.savefig(plot_fn)


def plot_cc_rv_diagnostic(
    rv_fit_dict,
    obj_name,
    figsize=(16,4),
    fig_save_path="plots/rv_diagnostics",
    run_in_wavelength_scale_debug_mode=False,):
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

    run_in_wavelength_scale_debug_mode: boolean, default: False
        If True, instead of plotting the offset of the observed spectrum from a
        template stellar spectrum, we plot the offset of the observed spectrum
        from a template *telluric* spectrum and zoom in on a region with H2O
        region.
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

    # ------------------------------------------------
    # Panel #1
    # ------------------------------------------------
    # Plot cross correlation fit
    cc_axis.plot(rv_steps, cross_corrs, linewidth=0.5,)
    cc_axis.set_title(obj_name)
    cc_axis.set_xlabel("RV (km/s)")
    cc_axis.set_ylabel("Cross Correlation")

    # Annotate the RV itself
    rv_txt = "RV = {:0.1f} km/s".format(rv)
    (y_min, y_max) = cc_axis.get_ylim()
    cc_axis.vlines(
        x=rv,
        ymin=y_min,
        ymax=y_max,
        colors="r",
        linestyles="dashed",
        alpha=0.75,
        label=rv_txt)
    
    cc_axis.legend(loc="lower right",)

    # ------------------------------------------------
    # Panel #2
    # ------------------------------------------------
    # Compute best fit template and plot spectral fit
    if run_in_wavelength_scale_debug_mode:
        template_flux = calc_template_flux(wave_template)
    else:
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

        if not run_in_wavelength_scale_debug_mode:
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
    
    leg = spec_axis.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fancybox=True,)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)
    
    spec_axis.set_xlabel("Wavelength (nm)")
    spec_axis.set_ylabel("Flux (cont norm)")

    spec_axis.set_xlim(np.nanmin(wave_2D)*0.99, np.nanmax(wave_2D)*1.01)
    spec_axis.set_ylim(0,2)

    if run_in_wavelength_scale_debug_mode:
        spec_axis.set_xlim(8270, 8295)

    plt.tight_layout()

    # Save plot
    if not os.path.isdir(fig_save_path):
        os.mkdir(fig_save_path)

    fig_name = os.path.join(fig_save_path, "rv_diagnostic")
    plt.savefig("{}_{}.pdf".format(fig_name, obj_name))
    plt.savefig("{}_{}.png".format(fig_name, obj_name), dpi=200)


def plot_all_cc_rv_diagnostics(
    all_rv_fit_dicts,
    obj_names,
    figsize,
    fig_save_path,
    run_in_wavelength_scale_debug_mode=False,):
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

    run_in_wavelength_scale_debug_mode: boolean, default: False
        If True, instead of plotting the offset of the observed spectrum from a
        template stellar spectrum, we plot the offset of the observed spectrum
        from a template *telluric* spectrum and zoom in on a region with H2O
        region.
    """
    n_stars = len(all_rv_fit_dicts)

    desc = "Plotting RV diagnostics"

    for star_i in tqdm(range(n_stars), desc=desc, leave=False):
        plot_cc_rv_diagnostic(
            rv_fit_dict=all_rv_fit_dicts[star_i],
            obj_name=obj_names[star_i],
            figsize=figsize,
            fig_save_path=fig_save_path,
            run_in_wavelength_scale_debug_mode=\
                run_in_wavelength_scale_debug_mode,)
        

def plot_all_flux_calibrated_spectra(
    wave_3D,
    spec_3D,
    sigma_3D,
    object_ids,
    is_spphot_1D,
    figsize,
    plot_folder,
    plot_label,):
    """Plots all flux normalised spectra in the 3D datacube, and normalises by
    the sigma clipped maximum of each. We annotate flux standards with a blue
    star symbol.

    Plot is saved as <plot_folder>/all_flux_cal_<plot_label>.pdf.

    Parameters
    ----------
    wave_3D, spec_3D, sigma_3D: 3D float array
        Wavelengths scale, spectra, and uncertainties of shape 
        [n_spec, n_order, n_px].

    object_ids: 1D str list
        List of object names of length [n_star] for labelling the plots.

    is_spphot_1D: 1D bool list
        Whether or not the given spectrum belongs to a flux standard.

    figsize: float tuple
        Two element tuple for the matplotlib figure size, e.g. (16,4).

    plot_folder: str
        Folder to save the plot to.

    plot_label: str
        Unique label to include in the plot filename.
    """
    # Grab dimensions for convenience
    (n_star, n_order, n_px) = wave_3D.shape

    # Normalise by (5 sigma) maximum of entire spectrum
    spec_3D_norm = spec_3D.copy()

    spec_maxes = np.nanmax(
        sigma_clip(spec_3D_norm, sigma=5, axis=(1,2)), axis=(1,2)).data
    spec_3D_norm /= np.broadcast_to(spec_maxes[:,None,None], spec_3D.shape)

    # HACK: exclude the edges
    spec_3D_norm[:,:,:5] = np.nan
    spec_3D_norm[:,:,-5:] = np.nan
    
    wave_min = np.nanmin(wave_3D)
    wave_max = np.nanmax(wave_3D)
    wave_mid = (wave_max - wave_min)/2 + wave_min

    plt.close("all")
    fig, axes = plt.subplots(figsize=figsize,)

    for star_i in range(n_star):
        for order_i in range(n_order):
            offset = 1 * star_i
            axes.plot(
                wave_3D[star_i, order_i],
                spec_3D_norm[star_i, order_i]+offset,
                c="r" if order_i % 2 == 0 else "k",
                alpha=0.75,
                linewidth=0.2)
        
        snr = np.nanmedian(spec_3D[star_i] / sigma_3D[star_i])
        txt = "{} [SNR ~ {:0.0f}]".format(object_ids[star_i], snr)

        axes.text(
            x=wave_mid,
            y=offset+0.75,
            s=txt,
            horizontalalignment="center",)
        
        if is_spphot_1D[star_i]:

            txt = r"$\star$ {} $\star$".format(" " * len(txt))

            axes.text(
                x=wave_mid,
                y=offset+0.75,
                s=txt,
                horizontalalignment="center",
                fontsize="xx-large",
                c="dodgerblue",)
            
    axes.set_ylim(0, offset+1)
    axes.set_xlabel("Wavelength (Å)")
    plt.tight_layout()
    plt.savefig("{}/all_flux_cal_{}.pdf".format(plot_folder, plot_label))


def plot_telluric_scale_terms(
    wave_2D,
    spec_2D,
    wave_tt,
    trans_tt,
    test_tt_scale,
    plot_label,
    species,
    wave_min,
    wave_max,):
    """Function to plot telluric 'correction' using different optical depth
    scaling terms for H2O and O2 to inform which to use when flux calibrating.
    Plots are saved as:

    plots/telluric_scale_diagnostics/telluric_scale_<plot_label>_<species>.png

    Parameters
    ----------
    wave_2D, spec_2D, sigma_2D: 2D float array
        Wavelengths scale, spectra, and uncertainties of shape [n_order, n_px].

    wave_tt, trans_tt: 1D float array
        Telluric wavelength scale and transmission vectors, this will either
        apply to H2O or O2.

    test_tt_scale: 1D float array
        Array of test values for optical depth scaling.

    plot_label: string
        Unique label for the plot filename and plot super title.

    species: string
        Species label, either H2O or O2.

    wave_min, wave_max: float
        Minimum and maximum wavelengths to show on the plot.
    """
    # Grab dimensions for convenience
    (n_order, n_px) = wave_2D.shape

    # Create interpolator for telluric transmission
    interp_trans_tt = interp1d(
        x=wave_tt,
        y=trans_tt,
        bounds_error=False,
        fill_value=1.0,
        kind="cubic",)

    # Interpolate telluric transmission onto observed wavelength scale as tau
    tau_tt_2D = np.full_like(wave_2D, np.nan)

    for order_i in range(n_order):
        wave_ith = wave_2D[order_i]
        tau_tt_2D[order_i] = -np.log(interp_trans_tt(wave_ith))

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=len(test_tt_scale), figsize=(16,12), sharex=True, sharey=True)

    fig.subplots_adjust(
        left=0.05,
        bottom=0.03,
        right=0.995,
        top=0.95,
        hspace=0.01)

    # Loop over all orders
    for order_i in range(n_order):
        wave_1D = wave_2D[order_i]

        # Don't plot any orders outside the wavelength range
        if (np.nansum(wave_1D > wave_min) == 0
            or np.nansum(wave_1D < wave_max) == 0):
            continue
        
        # For all test optical depth scaling terms, 'correct' the telluric
        # transmission and plot.
        for od_scale_i, od_scale in enumerate(test_tt_scale):
            # Compute telluric transmission
            trans_tt_2D = np.exp(-od_scale * tau_tt_2D[order_i])
            
            # 'Correct' telluric absorption
            spec_corr = spec_2D[order_i] / trans_tt_2D

            lbl = "{:0.2f}".format(od_scale)

            # Plot before and after
            axes[od_scale_i].plot(
                wave_1D, spec_2D[order_i], c="k", linewidth=0.5)
            axes[od_scale_i].plot(wave_1D, spec_corr, c="r", linewidth=0.5,)

            axes[od_scale_i].set_yscale("log")

            axes[od_scale_i].text(
                x=0.5,
                y=0.15,
                s=lbl,
                c="b",
                fontsize="small",
                horizontalalignment="center",
                transform=axes[od_scale_i].transAxes,
                bbox=dict(facecolor="b", alpha=0.1),)

    axes[od_scale_i].set_xlim(wave_min, wave_max)

    fig.suptitle("{} -- {}".format(plot_label, species))

    fn = "telluric_scale_{}_{}.png".format(plot_label, species)

    plt.savefig("plots/telluric_scale_diagnostics/{}".format(fn), dpi=300)


def plot_pseudocontinuum_normalisation_diagnostics(
    obs_info,
    wave_1D,
    spec_2D,
    spec_2D_norm,
    sigma_2D_norm,
    continua_2D,
    telluric_template="data/viper_stdAtmos_vis.fits",
    trans_mask_threshold=0.95,
    resolving_power_science=46000,
    figsize=(16,6),
    arm="r",
    plot_folder="plots",):
    """Creates two panel diagnostic plots of 1) science spectra, masked 
    spectral regions, and fitted pseudocontinuum; and 2) pseudocontinuum
    normalised spectra with uncertainties and overplotted telluric transmission
    for each science observation. Figures saved as PDFs to:

        <plot_folder>/cont_norm_<date>_<arm_><source_id>.pdf

    Parameters
    ----------
    obs_info: pandas dataframe
        Dataframe containing information about each observation.

    wave_1D: 1D float array
        1D wavelength scale for spec_2D and sigma_2D, of shape [n_px].
    
    spec_2D: 2D float array
        2D unnormalised spectra of shape [n_obs, n_px].

    spec_2D_norm, sigma_2D_norm, continua_2D: 2D float arrays
        Pseudeocontinuum normalised science spectra and uncertainties, as well
        as the pseudocontinuum itself, of shape [n_obs, n_px].

    telluric_template: str, default: 'data/viper_stdAtmos_vis.fits'
        Location of VIPER fits telluric transmission templates.

    trans_mask_threshold: float, default: 0.95
        Transmission threshold below which we consider a pixel too telluric
        contaminated to be included in the smoothing, i.e. 0.95 means that
        telluric lines of depth up to 5% are acceptable.

    resolving_power_science: float, default: 46000
        Resolving power of the science spectrum.

    figsize: float tuple
        Size of the figure to pass to plt.subplots().

    arm: str
        Spectrograph arm, either 'b' or 'r', used for selecting the appropriate
        SNR value from obs_info and for the figure filenames.

    plot_folder: str
        Filepath to save figures to.
    """
    # Grab dimensions for convenience
    (n_obs, n_px) = spec_2D_norm.shape

    # Import telluric spectrum, smoothed to the resolving power and wavelength
    # sampling of the science spectrum
    _, _, _, trans_telluric = sm.read_and_broaden_telluric_transmission(
        telluric_template=telluric_template,
        resolving_power=resolving_power_science,
        do_convert_vac_to_air=True,
        wave_scale_new=wave_1D,)
    
    is_telluric_affected = trans_telluric < trans_mask_threshold
    
    # -------------------------------------------------------------------------
    # Diagnostic plotting
    # -------------------------------------------------------------------------
    desc = "Plotting pseudocontinuum normalisation diagnostics"

    for obs_i in tqdm(range(n_obs), leave=False, desc=desc):
        plt.close("all")
        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=figsize,)

        # ----------------------------
        # Panel 1: pseudocontinuum fit
        # ----------------------------
        # Plot science spectrum (masking out telluric affected px)
        spec_1D_masked = spec_2D[obs_i].copy()
        spec_1D_masked[is_telluric_affected] = np.nan

        axes[0].plot(
            wave_1D,
            spec_1D_masked,
            linewidth=0.5,
            c="k",
            label="Science",)
        
        # Plot masked pixels
        spec_1D_ta = spec_2D[obs_i].copy()
        spec_1D_ta[~is_telluric_affected] = np.nan

        axes[0].plot(
            wave_1D,
            spec_1D_ta,
            linewidth=0.5,
            c="r",
            label="Science (Masked)",)
        
        # Plot the pseudocontinuum
        axes[0].plot(
            wave_1D,
            continua_2D[obs_i],
            linewidth=1.0,
            c="g",
            label="Pseudocontinuum",)

        leg_1 = axes[0].legend(loc="upper center", ncol=3, fontsize="small")

        for legobj in leg_1.legendHandles:
            legobj.set_linewidth(1.5)

        # -------------------------------------------
        # Panel 2: pseudocontinuum normalised spectra
        # -------------------------------------------
        # Plot the uncertainties
        axes[1].fill_between(
            wave_1D,
            spec_2D_norm[obs_i]+sigma_2D_norm[obs_i]/2, 
            spec_2D_norm[obs_i]-sigma_2D_norm[obs_i]/2, 
            alpha=0.5,
            color="r")
        
        # Plot the pseudocontinuum normalised science spectrum
        axes[1].plot(
            wave_1D,
            spec_2D_norm[obs_i],
            linewidth=0.5,
            c="k",
            label="Science (Normalised)")

        # Overplot the telluric transmission
        axes[1].plot(
            wave_1D,
            trans_telluric,
            linewidth=0.5,
            c="b",
            alpha=0.5,
            label="Telluric Transmission",)
        
        leg_2 = axes[1].legend(loc="upper center", ncol=2, fontsize="small")

        for legobj in leg_2.legendHandles:
            legobj.set_linewidth(1.5)

        # -------------------------------------------
        # Final plot setup + saving
        # -------------------------------------------
        axes[1].set_xlabel(r"Wavelength (${\rm \AA}$)")

        title = "{} ({}, SNR~{:0.0f})".format(
            obs_info.iloc[obs_i]["source_id"],
            obs_info.iloc[obs_i]["ut_date"],
            obs_info.iloc[obs_i]["snr_{}".format(arm)],)

        fig.suptitle(title)

        axes[1].set_xlim(wave_1D[0]*0.995, wave_1D[-1]*1.005)

        plt.tight_layout()

        # Save plot
        if not os.path.isdir(plot_folder):
            os.mkdir(plot_folder)

        fn = "cont_norm_{}_{}_{}.pdf".format(
            obs_info.iloc[obs_i]["ut_date"],
            arm,
            obs_info.iloc[obs_i]["source_id"])

        plot_fn = os.path.join(plot_folder, fn)

        plt.savefig("{}.pdf".format(plot_fn))