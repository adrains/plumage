"""Various plotting functions"
"""
from __future__ import print_function, division
import os
import numpy as np
import glob
import batman as bm
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.colors as mplc
import plumage.spectra as spec
import plumage.synthetic as synth
import plumage.transits as transit
import plumage.utils as utils
from tqdm import tqdm
import matplotlib.transforms as transforms
import matplotlib.ticker as plticker
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.integrate import simps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# Ensure the plotting folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "plots"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

def plot_nightly_spectra(root, night, plot_step_id="10", 
                         snr_step_id="08", plot_only=None, 
                         plot_output_path=None):
    """Plots red and blue band spectra for each night, stacked and offset, 
    saved to plumage/plots.

    Parameters
    ----------
    root: str
        Base filepath where reductions are stored.

    night: str
        Night observations were taken, and name of folder they're stored in.
        Of format YYYYMMDD.

    plot_step_id: str
        The data reduction step ID of pyWiFeS to use for plotting
        (i.e. typically after flux and telluric correction)

    snr_step_id: str
        The data reduction step ID of pyWiFeS to use for calculation of SNR 
        (i.e. before flux and telluric correction)

    plot_only: str or None
        If not none, provide either "r" or "b" to plot only that arm of the 
        spectrograph. Useful if issues with other arm.

    plot_output_path: str or None
        Directory to save the output plot to. If None, defaults to 
        plumage/plots/.

    """
    # Import 
    path_plt = os.path.join(root, night, "ascii", "*_%s_*" % plot_step_id)
    path_snr = os.path.join(root, night, "ascii", "*_%s_*" % snr_step_id)
    files_plt = glob.glob(path_plt)
    files_snr = glob.glob(path_snr)

    files_plt.sort()
    files_snr.sort()

    # If plotting both arms, make sure we have an equal amount of spectra
    if plot_only is None:
        assert len(files_plt) == len(files_snr)
        bands = ["b", "r"]
        n_ax = 2

    # if plotting only one arm, this doesn't matter
    elif plot_only == "r":
        bands = ["r"]
        n_ax = 1

    elif plot_only == "b":
        bands = ["b"]
        n_ax = 1

    else:
        raise Exception("Unknown arm")

    # Abort if still no files
    if len(files_plt) == 0:
        print("No fits files, aborting.")
        return
    
    plt.close("all")
    fig, axes = plt.subplots(1, n_ax, sharey=True)
    #bands = ["b", "r"]

    if plot_only is not None:
        axes = [axes]

    snr_sort_i = None

    # Plot bands on separate axes
    for band, axis in zip(bands, axes):
        fplt_band = np.array([fp for fp in files_plt 
                              if "_%s.dat" % band in fp])
        fsnr_band = np.array([fs for fs in files_snr 
                              if "_%s.dat" % band in fs])

        # Sort by SNR
        snrs = []
        for fsnr in fsnr_band:
            sp_snr = np.loadtxt(fsnr)
            snrs.append(np.nanmedian(sp_snr[:,1])/np.sqrt(np.nanmedian(sp_snr[:,1])))

        snrs = np.array(snrs)

        # Sort only once (with order from a single band)
        if snr_sort_i is None:
            snr_sort_i = np.argsort(snrs)
            print("Sorting")

        fplt_band = fplt_band[snr_sort_i]
        fsnr_band = fsnr_band[snr_sort_i]
        snrs = snrs[snr_sort_i]
    
        # Now plot
        for sp_i, (fplt, fsnr, snr) in enumerate(zip(fplt_band, fsnr_band, 
                                                     snrs)):
            sp_plt = np.loadtxt(fplt)
            sp_snr = np.loadtxt(fsnr)

            axis.plot(sp_plt[:,0], sp_plt[:,1]/np.median(sp_plt[:,1])+2*sp_i, 
                    linewidth=0.1)

            # Plot label
            if np.isnan(snr):
                label = "%s [%i]" % (fplt.split("/")[-1][9:-4], 0)
            else:
                label = "%s [%i]" % (fplt.split("/")[-1][9:-4], snr)
            
            print(sp_i, label)
            axis.text(sp_plt[:,0].mean(), 2*sp_i, label, fontsize="x-small", 
                      ha="center")
    
    axis.set_ylim([-1,sp_i*2+4])
    fig.suptitle(night)
    fig.tight_layout()
    fig.text(0.5, 0.04, "Wavelength (A)", ha='center')
    fig.text(0.04, 0.5, "Flux (scaled)", va='center', rotation='vertical')
    plt.gcf().set_size_inches(9, 16)

    # Save plot
    if plot_output_path is None:
        plot_output_path = "plots"
    
    plt.savefig(os.path.join(plot_output_path, "spectra_%s.pdf" % night))

def merge_spectra_pdfs(path, new_fn):
    """Merge diagnostic pdfs together for easy checking.
    
    Code from:
    https://stackoverflow.com/questions/3444645/merge-pdf-files
    """
    from PyPDF2 import PdfFileMerger

    pdfs = glob.glob(path)
    pdfs.sort()
    
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    with open(new_fn, 'wb') as fout:
        merger.write(fout)

def plot_teff_sorted_spectra(spectra, observations, arm="r",
                             mask=None, suffix="", normalise=False,
                             show_telluric_box=True):
    """Plot all spectra, their IDs, RVs, and Teffs sorted by Teff.
    """
    if mask is None:
        mask = np.ones(len(spectra)).astype(bool)

    plt.close("all")
    teff_order = np.argsort(observations[mask]["teff_fit_rv"].values)
    sorted_spec = spectra[mask][teff_order]
    ids = observations[mask]["id"].values[teff_order]
    uids = observations[mask]["uid"].values[teff_order]
    teffs = observations[mask]["teff_fit_rv"].values[teff_order]
    loggs = observations[mask]["logg_fit_rv"].values[teff_order]
    fehs = observations[mask]["feh_fit_rv"].values[teff_order]
    rvs = observations[mask]["rv"].values[teff_order]
    e_rvs = observations[mask]["e_rv"].values[teff_order]
    programs = observations[mask]["program"].values[teff_order]
    subsets = observations[mask]["subset"].values[teff_order]

    if arm == "b":
        snrs = observations[mask]["snr_b"].values[teff_order]
    elif arm == "r":
        snrs = observations[mask]["snr_r"].values[teff_order]
    else:
        raise ValueError("Invalid Arm: must be either b or r.")

    for sp_i, (spec, id, teff, logg, feh, rv, e_rv, snr, pg, sub) in enumerate(
        zip(sorted_spec, ids, teffs, loggs, fehs, rvs, e_rvs, snrs,
            programs, subsets)): 
        # Rescale if normalising
        if normalise:
            spec_scale = np.nanmedian(spec[1,:])
        else:
            spec_scale = 1

        plt.plot(spec[0,:], sp_i+spec[1,:]/spec_scale, linewidth=0.1) 
        #label = "%s [%i K, %0.2f km/s]" % (id, teff, rv)

        # Prepare the label
        label = (r"%s [%s, %s, %i K, %0.1f, %0.2f, %0.2f$\pm$%0.2f km/s]"
                % (id, pg, sub, teff, logg, feh, rv, e_rv))
        label += r" SNR ~ %i" % snr 

        plt.text(spec[0,:].mean(), sp_i+0.5, label, fontsize=4, 
                        ha="center")

    if arm == "r" and show_telluric_box:
        h2o = [6270.0, 6290.0]
        o2 = [6856.0, 6956.0]

        plt.axvspan(h2o[0], h2o[1], ymin=0, ymax=sp_i+1, alpha=0.1, color="grey")
        plt.axvspan(o2[0], o2[1], ymin=0, ymax=sp_i+1, alpha=0.1, color="grey")

    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux (Normalised, offset)")
    plt.ylim([0,sp_i+2])
    plt.gcf().set_size_inches(9, 80)
    plt.tight_layout()

    if suffix != "":
        plt.savefig("plots/teff_sorted_spectra_%s_%s.pdf" % (arm, suffix))
    else:
        plt.savefig("plots/teff_sorted_spectra_%s.pdf" % arm)


def plot_normalised_spectra(spectra, observations, band="r", snr=0, mask=None,
                            plot_balmer=False, do_med_norm=False):
    """
    """
    if mask is None:
        mask = np.ones(len(spectra)).astype(bool)

    plt.close("all")

    stars_plotted = 0

    for i in range(0,len(spectra[mask])): 
        if do_med_norm:
            scale = np.nanmedian(spectra[mask][i,1,:])
        else:
            scale = 1

        if int(observations[mask].iloc[i]["snr_%s" % band]) > snr: 
            plt.plot(spectra[mask][i,0,:], spectra[mask][i,1,:]/scale,linewidth=0.1,)
                     #alpha=0.5, color="black") 
            stars_plotted += 1
    
    print("%i stars plotted" % (stars_plotted))

    if band == "b":
        plt.xlim([3600,5500]) 
    elif band == "r":
        plt.xlim([5400,7000]) 

    if plot_balmer:
        plot_balmer_series()

    plt.ylim([0,3])
    plt.xlabel("Wavelength (A)") 
    plt.ylabel("Flux (Normalised)")
    
    
def plot_balmer_series():
    """
    """
    balmer = {r"H$\alpha$":6564.5,
                r"H$\beta$":4861.4,
                r"H$\gamma$":4340.5,
                r"H$\delta$":4101.7,
                r"H$\epsilon$":3970.1,
                r"H$\zeta$":3889.01,
                r"H$\eta$":3835.4,}

    for line in balmer.keys():
        plt.vlines(balmer[line], 0.2, 10, color="grey", linestyles="dashed", linewidth=1)
        plt.text(balmer[line], 0.1, line, fontsize="x-small", horizontalalignment='center')


def plot_templates(ref_spec_norm, ref_params):
    """
    """
    plt.close("all")
    for sp_i, (spec, params) in enumerate(zip(ref_spec_norm[::-1], ref_params[::-1])):
        plt.plot(spec[0,:], 2*sp_i+spec[1,:],linewidth=0.1)

        label = (r"T$_{\rm eff}=%i\,$K, $\log g=%0.1f$, [Fe/H]$=%0.1f$"
                 % (params[0], params[1], params[2]))
        plt.text(spec[0,:].mean(), 2*sp_i, label, fontsize=4, 
                        ha="center")

    plt.xlabel("Wavelength (A)") 
    plt.ylabel("Flux (Normalised)")
    plt.gcf().set_size_inches(9, 20)
    plt.tight_layout()
    plt.savefig("plots/teff_sorted_ref_spectra.pdf") 
    plt.savefig("plots/teff_sorted_ref_spectra.png") 
    

def plot_standards(spectra, observations, catalogue):
    """
    """
    # Get the standards
    is_standard = catalogue["program"] == "standard"

    # Get only IDs that have been matched
    #id_match_mask = [sid is not "" for sid in observations["uid"]]

    # Now get the standards that have been observed
    is_observed = [sid in observations["uid"].values 
                            for sid in catalogue["source_id"].values]

    # Final mask exludes those with empty souce ID
    mask = is_standard * is_observed

    plt.close("all")
    plt.scatter(catalogue[mask]["teff_lit"], catalogue[mask]["logg_lit"],)
                #catalogue[mask]["feh_lit"])

    plt.xlim([6700, 3000])
    plt.ylim([5.1, 1])


# -----------------------------------------------------------------------------
# Synthetic fit diagnostics
# ----------------------------------------------------------------------------- 
def plot_synthetic_fit(
    wave, 
    spec_sci, 
    e_spec_sci, 
    spec_synth, 
    bad_px_mask,
    teff,
    e_teff,
    logg,
    e_logg,
    feh,
    e_feh,
    rv,
    e_rv,
    snr,
    date,
    airmass,
    rchi2=None,
    date_id=None, 
    save_path=None, 
    fig=None, 
    axis=None, 
    save_fig=False, 
    arm="r", 
    bad_synth_px_mask=None,
    spec_synth_lit=None):
    """TODO: Sort out proper sharing of axes

    Parameters
    ----------
    wave: float array
        Wavelength scale for the spectrum.

    spec_sci: float array
        Spectrum flux vector corresponding to wave.

    e_spec_sci: float array
        Flux uncertainties corresponding to wave.

    spec_synth: float array
        Synthetic spectrum flux corresponding to wave.

    bad_px_mask: boolean array
        Array of bad pixels (i.e. bad pixels are True) corresponding to wave.

    obs_info: pandas dataframe row
        Information on observations of target in question.

    param_cols: string array
        Column names for the synthetic fit params. Generally:
        ["teff_synth", "logg_synth", "feh_synth"]

    date_id: string, default: None
        String used to uniquely label each plot if saving. 

    save_path: string, default: None
        Where to save the resulting plot if saving.

    fig: matplotlib.figure.Figure object, default: None
        Existing figure to plot with, otherwise generate new one.
    
    axis: matplotlib.axes._subplots.AxesSubplot, default: None
        Existing axis to plot on, otherwise make new one.

    save_fig: bool, default: False
        Whether to save the plot or not.

    arm: string, default: 'r'
        Arm of spectrograph, either 'b' or 'r'
    
    spec_synth_lit: float array, default: None
        Synthetic spectra generated at literature values (for standard stars).
        Of shape same shape as wave if provided, otherwise defaults to
        None and not plotted.
    """
    # Initialise empty mask for bad pixels if none is provided
    if bad_px_mask is None:
        bad_px_mask = np.full(len(wave), False)

    # Use fig/axis handles we've been given, otherwise make new ones
    if fig is None and axis is None:
        plt.close("all")
        fig, axis = plt.subplots()

    # Setup lower panel for residuals
    plt.setp(axis.get_xticklabels(), visible=False)
    divider = make_axes_locatable(axis)
    res_ax = divider.append_axes("bottom", size="30%", pad=0)
    axis.figure.add_axes(res_ax, sharex=axis)

    axis.errorbar(wave, spec_sci, yerr=e_spec_sci, label="sci", linewidth=0.2,
                  elinewidth=0.2, barsabove=True, capsize=0.3, capthick=0.1)
    axis.plot(wave, spec_synth, "--", label="synth", linewidth=0.2)

    # Plot the literature synthetic spectrum if it has been provided
    if spec_synth_lit is not None:
        axis.plot(wave, spec_synth_lit, "-.", label="synth (lit)", 
                  linewidth=0.2)

    # Plot residuals
    res_ax.hlines(0, wave[0]-100, wave[-1]+100, linestyles="dotted", 
                  linewidth=0.2)
    res_ax.plot(wave, spec_sci-spec_synth, linewidth=0.2, color="red")
    axis.set_xlim(wave[0]-10, wave[-1]+10)
    axis.set_ylim(0.0, 2.0)
    res_ax.set_xlim(wave[0]-10, wave[-1]+10)
    res_ax.set_ylim(-0.3, 0.3)

    # Shade the bad observed pixels in red
    shade_excluded_regions(wave, bad_px_mask, axis, res_ax, colour="red",
                           alpha=0.1)

    # Shade the bad synthetic wavelengths in black
    if bad_synth_px_mask is not None:
        shade_excluded_regions(wave, bad_synth_px_mask, axis, res_ax, 
                               colour="black", alpha=0.05, hatch="//")

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Label synthetic fit parameters
    param_label = (r"$T_{{\rm eff}} = {:0.0f}\pm {:0.0f}\,$K, "
                   r"$\log g = {:0.2f}\pm {:0.2f}$, "
                   r"[Fe/H]$ = {:0.2f}\pm {:0.2f}$")
    param_label = param_label.format(teff, e_teff, logg, e_logg, feh, e_feh)
    axis.text(np.nanmean(wave), 1.6, param_label, horizontalalignment="center")

    if rchi2 is not None:
        rchi2_label = r"red. $\chi^2 = {:0.0f}$".format(rchi2)
        axis.text(np.nanmean(wave), 1.525, rchi2_label, 
                  horizontalalignment="center")

    # Label RVs
    rv_label = r"RV$ = {:0.2f}\pm{:0.2f}\,$km$\,$s$^{{-1}}$".format(rv, e_rv)
    axis.text(np.nanmean(wave), 1.45, rv_label, horizontalalignment="center")

    # Label SNR
    snr_label = r"SNR ({}) $\sim {:0.0f}$".format(arm, snr)
    axis.text(np.nanmean(wave), 1.375, snr_label, horizontalalignment="center")

    # Label date
    date_label = r"Date = {}".format(date)
    axis.text(np.nanmean(wave), 1.30, date_label, horizontalalignment="center")

    # Label airmass
    airmass_label = r"Airmass = {:0.2f}".format(airmass)
    axis.text(np.nanmean(wave), 1.225, airmass_label, 
              horizontalalignment="center")

    # Plot estimated parameters if we've been given them
    if False:
        kp_label = r""

        if "teff" in known_params:
            kp_label += r"$T_{\rm eff} = {:4.0f} \pm {:4.0f} ({})".format(
                known_params["teff"]
            )
        
        axis.text(np.nanmean(wave), 0.3, kp_label, horizontalalignment="left")

    # Do final plot setup
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.legend(loc="upper left")
    axis.set_ylabel("Flux (Normalised)")
    res_ax.set_ylabel("Residuals")
    res_ax.set_xlabel("Wavelength (A)")

    if save_fig:
        plt.suptitle(date_id)
        plt.savefig(save_path)


def plot_cmd_diagnostic(colours, abs_mags, ruwe_mask, target_colour, target_abs_mag, 
             target_ruwe, ax, xlabel, y_label, text_min=None):
    """Plots a colour magnitude diagram, outlining those stars with high
    values for Gaia DR2 RUWE (>1.4) and highlighting the current target star.

    Parameters
    ----------
    colours: float array
        CMD colour for the x-axis.
    
    abs_mags: float array
        CMD absolute magnitudes for the y-axis

    ruwe_mask: boolean array
        Boolean mask that is true where RUWE > 1.4 (i.e. flagging bad stars)

    target_colour: float
        CMD colour for the current target

    target_abs_mag: float array
        CMD absolute magnitude for the current target

    target_ruwe: float
        RUWE value for the target

    ax: matplotlib.axes._subplots.AxesSubplot
        Axis to plot on
    
    xlabel, ylabel: string
        Labels for the x and y axis (can include LaTeX formatting)

    text_min: float
        Y value to be included in limits
    """
    face_colours = ["blue"]*len(colours)
    edge_colours = ["red" if ruwe else "blue" for ruwe in ruwe_mask]

    # First plot all the points
    ax.scatter(colours, abs_mags, c=face_colours, edgecolors=edge_colours)

    # Now just the target
    if target_ruwe < 1.4:
        ax.scatter(target_colour, target_abs_mag, c="gold")
    else:
        ax.scatter(target_colour, target_abs_mag, c="gold", edgecolors="red")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)

    # Flip magnitude axis
    ymin, ymax = ax.get_ylim()

    # Consider text when setting limits
    if text_min is None:
        text_min = ymin

    if ymin > text_min - 0.5:
        ymin = text_min - 0.5

    ax.set_ylim((ymax, ymin))


def plot_synth_fit_diagnostic(
    wave_r,
    spec_r,
    e_spec_r,
    spec_synth_r,
    bad_px_mask_r,
    obs_info,
    info_cat,
    plot_path,
    wave_b=None,
    spec_b=None,
    e_spec_b=None,
    spec_synth_b=None,
    bad_px_mask_b=None,
    is_tess=False,
    use_2mass_id=False,
    text_min=4.0,
    spec_synth_lit_b=None,
    spec_synth_lit_r=None,
    logg_from_info_cat=True,):
    """Plots synthetic fit diagnostic plot, with four sections. Leftmost has
    Gaia Bp-Rp, absolute G, and 2MASS J-K - absolute K CMD of stars in info_cat
    highlighting the star in question. On the top right is the blue spectra,
    and the bottom right the red spectra, both with observed, synthetic, and 
    residuals.  Diagnostic information/flags are plotted as text.

    Note that if a target is not in info_cat, it is not plotted.

    Parameters
    ----------
    wave_r: float array
        Wavelength scale for the red spectrum.

    spec_r: float array
        The red spectrum corresponding to wave_r.

    spec_r: float array
        The red spectrum corresponding to wave_r.

    spec_synth_r: float array
        The red synthetic spectrum corresponding to wave_r.

    bad_px_mask_r: boolean array
        Array of bad pixels (i.e. bad pixels are True) for red arm
        corresponding to wave_r.

    obs_info: pandas dataframe row
        Information on observations of target in question.

    info_cat: pandas dataframe
        Dataframe containing photometric/literature information on each target

    plot_path: string
        Where to save the resulting plot.

    wave_b: float array
        Wavelength scale for the blue spectrum.

    spec_b: float array
        The blue spectrum corresponding to wave_b.

    spec_b: float array
        The blue spectrum corresponding to wave_b.

    spec_synth_b: float array
        The blue synthetic spectrum corresponding to wave_b.

    bad_px_mask_b: boolean array
        Array of bad pixels (i.e. bad pixels are True) for blue arm
        corresponding to wave_b.

    is_tess: bool
        Indicates whether we the targets have literature parameters available,
        or whether we should plot with TOI IDs.

    spec_synth_lit_b, spec_synth_lit_r: float array, default: None
        Synthetic spectra generated at literature values (for standard stars).
        Of shape same shape as wave_b/wave_r if provided, otherwise defaults to
        None and not plotted.
    """
    # Initialise
    plt.close("all")
    fig = plt.figure()#constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=5)
    ax_cmd_gaia = fig.add_subplot(gs[0, 0])
    ax_cmd_2mass = fig.add_subplot(gs[1, 0])
    ax_spec_b = fig.add_subplot(gs[0, 1:])
    ax_spec_r = fig.add_subplot(gs[1, 1:])

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Colour Magnitude Diagram
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if "pc" in info_cat:
        mask = np.logical_and(info_cat["observed"], info_cat["pc"])
    else:
        mask = info_cat["observed"]

    # Get target info
    sid = obs_info.name#["source_id"]
    is_obs = info_cat["observed"].values
    star_info = info_cat[is_obs][info_cat[is_obs].index==sid].iloc[0]

    # Get the ID and info about the target
    if is_tess:
        toi = star_info["TOI"]
        tic = star_info.name

        plt.suptitle("TOI {}, TIC {}, Gaia DR2 {}".format(
            toi, tic, sid))
    
    else:
        id_prefix = "Gaia DR2 "
        id_col = "ID"

        plt.suptitle("{}, Gaia DR2 {}".format(star_info[id_col], sid))

    # Plot Gaia CMD
    plot_cmd_diagnostic(
        info_cat[mask]["Bp-Rp"],
        info_cat[mask]["G_mag_abs"],
        info_cat[mask]["ruwe"] > 1.4,
        star_info["Bp-Rp"],
        star_info["G_mag_abs"],
        star_info["ruwe"],
        ax_cmd_gaia,
        r"$B_P-R_P$",
        r"$G_{\rm abs}$",
        text_min=text_min)
    
    # Plot 2MASS CMD
    plot_cmd_diagnostic(
        info_cat[mask]["J-K"],
        info_cat[mask]["K_mag_abs"],
        info_cat[mask]["ruwe"] > 1.4,
        star_info["J-K"], 
        star_info["K_mag_abs"],
        star_info["ruwe"],
        ax_cmd_2mass, 
        r"$J-K$",
        r"$K_{\rm abs}$")

    centre_bprp = np.mean(ax_cmd_gaia.get_xlim())   
    centre_jk = np.mean(ax_cmd_2mass.get_xlim())   

    # Plot diagnostic info for top CMD
    ax_cmd_gaia.text(centre_bprp, text_min, 
        "RUWE = {:.2f}".format(float(star_info["ruwe"])),
        horizontalalignment="center")
    ax_cmd_gaia.text(centre_bprp, 4.5, 
        r"$T_{{{{\rm eff}}, (B_p-R_p)}}  = {:.0f} \pm {:.0f}\,K$".format(
            float(star_info["teff_m15_bprp"]), 
            float(star_info["e_teff_m15_bprp"])),
        horizontalalignment="center")
    ax_cmd_gaia.text(centre_bprp, 5.0, 
        r"$T_{{{{\rm eff}}, (B_p-R_p, J-H)}} = {:.0f} \pm {:.0f}\,K$".format(
            float(star_info["teff_m15_bprp_jh"]), 
            float(star_info["e_teff_m15_bprp_jh"])),
        horizontalalignment="center")
    ax_cmd_gaia.text(centre_bprp, 5.5, 
        r"$M_{{K_s}} = {:.2f} \pm {:.2f}\,M_{{\odot}}$".format(
            float(star_info["mass_m19"]), 
            float(star_info["e_mass_m19"])),
        horizontalalignment="center")
    ax_cmd_gaia.text(centre_bprp, 6.0, 
        "# Obs = {}".format(int(star_info["wife_obs"])),
        horizontalalignment="center")
    ax_cmd_gaia.text(centre_bprp, 6.5, 
        "Dist. = {:.2f} pc".format(float(star_info["dist"])),
        horizontalalignment="center")

    # And bottom CMD
    ax_cmd_2mass.text(centre_jk, 3.5, 
        "Blended 2MASS = {}".format(bool(int(star_info["blended_2mass"]))),
        horizontalalignment="center")

    # Sort out logg
    if logg_from_info_cat:
        logg = star_info["logg_m19"]
        e_logg = star_info["e_logg_m19"]
    else:
        logg = np.nan
        e_logg = np.nan

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Blue spectra
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param_cols = ["teff_synth", "e_teff_synth", "logg_synth", "e_logg_synth", 
                  "feh_synth", "e_feh_synth"]

    spec_b, e_spec_b = spec.norm_spec_by_wl_region(wave_b, spec_b, "b", 
                                                   e_spec_b)

    # Setup temperature dependent wavelength masks for regions where the 
    # synthetic spectra are bad (e.g. missing opacities) at cool teffs
    bad_synth_px_mask_b = synth.make_synth_mask_for_bad_wl_regions(
        wave_b, 
        obs_info["rv"], 
        obs_info["bcor"], 
        obs_info["teff_fit_rv"])

    plot_synthetic_fit(
        wave_b, 
        spec_b, 
        e_spec_b, 
        spec_synth_b,
        bad_px_mask_b,
        teff=obs_info["teff_synth"],
        e_teff=obs_info["e_teff_synth"],
        logg=logg,
        e_logg=e_logg,
        feh=obs_info["feh_synth"],
        e_feh=obs_info["e_feh_synth"],
        rv=obs_info["rv"],
        e_rv=obs_info["e_rv"],
        snr=obs_info["snr_b"],
        date=obs_info["date"].split("T")[0],
        airmass=obs_info["airmass"],
        fig=fig, 
        axis=ax_spec_b,
        arm="b",
        bad_synth_px_mask=bad_synth_px_mask_b,
        spec_synth_lit=spec_synth_lit_b)

    # Intialise columns to use for two sets of reference stars
    mann15_cols = ["teff_m15", "e_teff_m15", "logg_m19", 
                   "feh_m15", "e_feh_m15"]
    ra12_cols = ["teff_ra12", "e_teff_ra12", "logg_m19", 
                 "feh_ra12", "e_feh_ra12"]
    int_cols = ["teff_int", "e_teff_int", "logg_m19", 
                "feh_m15", "e_feh_m15"] # last three will be blank

    lit_params = np.full(6, np.nan)
    kind = ""
    source = ""

    # If lit params are known, display
    if (not is_tess and spec_synth_lit_b is not None 
        and spec_synth_lit_r is not None):
        # First check which we're using - Mann+15
        if np.isfinite(np.sum(star_info[mann15_cols].values)):
            lit_params = star_info[mann15_cols].values
            kind = "NIR"
            source = "Mann+15"
        
        # Rojas-Ayala+12
        elif np.isfinite(np.sum(star_info[ra12_cols].values)):
            lit_params = star_info[ra12_cols].values
            kind = "NIR"
            source = "Rojas-Ayala+12"

        # Interferometry
        elif np.isfinite(np.sum(star_info[["teff_int", "e_teff_int"]].values)):
            lit_params = star_info[int_cols].values
            kind = "interferometry"
            source = str(star_info["source"])

        # Empty
        else:
            lit_params = np.full(5, np.nan)
            kind = ""
            source = ""

        lit_param_label = (r"Lit Params ({}, {}): $T_{{\rm eff}} = {:0.0f} "
                           r"\pm {:0.0f}\,$K, $\log g = {:0.2f}$, [Fe/H]$ = "
                           r"{:0.2f} \pm {:0.2f}$")
        lit_param_label = lit_param_label.format(
            kind,
            source, 
            lit_params[0], 
            lit_params[1], 
            lit_params[2], 
            lit_params[3], 
            lit_params[4])
        ax_spec_b.text(np.nanmean(wave_b), 0.25, lit_param_label, 
                       horizontalalignment="center")

    ax_spec_b.text(np.nanmean(wave_b), 0.175, 
    "both arms used in fit = {}".format(obs_info["both_arm_synth_fit"]), 
    horizontalalignment="center")

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Red spectra
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    spec_r, e_spec_r = spec.norm_spec_by_wl_region(wave_r, spec_r, "r", 
                                                   e_spec_r)

    # Setup temperature dependent wavelength masks for regions where the 
    # synthetic spectra are bad (e.g. missing opacities) at cool teffs
    bad_synth_px_mask_r = synth.make_synth_mask_for_bad_wl_regions(
        wave_r, 
        obs_info["rv"], 
        obs_info["bcor"], 
        obs_info["teff_fit_rv"])

    plot_synthetic_fit(
        wave_r, 
        spec_r, 
        e_spec_r, 
        spec_synth_r,
        bad_px_mask_r,
        teff=obs_info["teff_synth"],
        e_teff=obs_info["e_teff_synth"],
        logg=logg,
        e_logg=e_logg,
        feh=obs_info["feh_synth"],
        e_feh=obs_info["e_feh_synth"],
        rv=obs_info["rv"],
        e_rv=obs_info["e_rv"],
        snr=obs_info["snr_r"],
        date=obs_info["date"].split("T")[0],
        airmass=obs_info["airmass"],
        fig=fig, 
        axis=ax_spec_r,
        arm="r",
        bad_synth_px_mask=bad_synth_px_mask_r,
        spec_synth_lit=spec_synth_lit_r)

    plt.gcf().set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_save_loc = os.path.join(
        plot_path, 
        "{}_{}.pdf".format(int(obs_info["teff_synth"]), sid))
    plt.savefig(plt_save_loc)


def plot_all_synthetic_fits(
    spectra_r, 
    synth_spec_r, 
    bad_px_masks_r,
    observations,
    label,
    info_cat,
    spectra_b=None, 
    synth_spec_b=None, 
    bad_px_masks_b=None,
    is_tess=False,
    use_2mass_id=False,
    spec_synth_lit_b=None,
    spec_synth_lit_r=None,):
    """Plots diagnostic plots of all synthetic fits and position on Gaia/2MASS, 
    colour magnitude diagram, saving pdfs to plots/synth_diagnostics_{label}. 
    Plots are merged into single multi-page pdf document when done.

    Parameters
    ----------
    spectra_r: float array
        3D numpy array of red spectra of form [N_ob, wl/spec/sigma, flux].

    synth_spec_r: float array
        3D numpy array of red synthetic spectra of form [n_ob, flux]

    bad_px_masks_r: boolean array
        3D numpy boolean bad pixel mask of form [n_ob, px]

    observations: pandas dataframe
        Dataframe containing information about each observation.

    label: string
        Unique label for set of stars (e.g. tess, std) for custom plot save 
        folder.

    info_cat: pandas dataframe
        Dataframe containing photometric/literature information on each target

    spectra_b: float array
        3D numpy array of blue spectra of form [N_ob, wl/spec/sigma, flux].

    synth_spec_b: float array
        3D numpy array of blue synthetic spectra of form [n_ob, flux]

    bad_px_masks_b: boolean array
        3D numpy boolean blue bad pixel mask of form [n_ob, px]

    is_tess: bool
        Indicates whether we the targets have literature parameters available,
        or whether we should plot with TOI IDs.
    
    spec_synth_lit_b, spec_synth_lit_r: float array, default: None
        Synthetic spectra generated at literature values (for standard stars).
        Of shape [n_ob, px] if provided, otherwise defaults to None and not
        plotted.
    """
    # Make plot path if it doesn't already exist
    plot_path = os.path.join("plots", "synth_diagnostics_{}".format(label))

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    
    # If it does, clear out pdfs
    else:
        pdfs = glob.glob(os.path.join(plot_path, "*.pdf"))
        for pdf in pdfs:
            os.remove(pdf)

    # Plot diagnostics pdf for every spectrum
    for i in tqdm(range(len(observations)), desc="Plotting diagnostics"):
        # If we haven't been given literature synthetic spectra, pass through
        # None
        if spec_synth_lit_b is None or spec_synth_lit_r is None:
            spec_synth_lit_b_i = None
            spec_synth_lit_r_i = None
        else:
            spec_synth_lit_b_i = spec_synth_lit_b[i]
            spec_synth_lit_r_i = spec_synth_lit_r[i]

        # Only bother plotting stars in the paper
        if observations.iloc[i].name not in info_cat.index:
            continue

        # Do plotting
        plot_synth_fit_diagnostic(
            spectra_r[i,0], 
            spectra_r[i,1], 
            spectra_r[i,2], 
            synth_spec_r[i], 
            bad_px_masks_r[i], 
            observations.iloc[i], 
            info_cat,
            plot_path,
            spectra_b[i,0], 
            spectra_b[i,1], 
            spectra_b[i,2], 
            synth_spec_b[i], 
            bad_px_masks_b[i],
            is_tess=is_tess,
            use_2mass_id=use_2mass_id,
            spec_synth_lit_b=spec_synth_lit_b_i,
            spec_synth_lit_r=spec_synth_lit_r_i,
            )

    # Merge plots
    glob_path = os.path.join(plot_path, "*.pdf")
    merge_spectra_pdfs(glob_path, "plots/synthetic_fits_{}.pdf".format(label))


def shade_excluded_regions(wave, bad_px_mask, axis, res_ax, colour, alpha, 
                           hatch=None):
    """Function to optimally block/shade the a given a bad pixel mask using the
    minumum number of calls to the shading function.

    Parameters
    ----------
    wave: float array
        The wavelength scale for the plot (i.e. x axis)
    
    bad_px_mask: bool array
        Bad pixel mask of same size as wave, with bad pixels True
    
    axis: matplotlib.axes._subplots.AxesSubplot
        Axis object to plot on

    res_ax: matplotlib.axes._subplots.AxesSubplot
        Residual axis to plot on

    colour: string
        Colour of the shaded region

    alpha: float
        Alpha value of the shaded region

    hatch: string, default: None
        Hatching style to use (if applicable)
    """
    wl_delta = (wave[1] - wave[0]) / 2

    previous_true = False

    for xi in range(len(wave)):
        # We've gotten to the end of the shaded region, draw
        if (not bad_px_mask[xi] and previous_true
            or xi+1 == len(bad_px_mask) and previous_true):
            axis.axvspan(wave_min, wave_max, ymin=0, 
                        ymax=1.7, alpha=alpha, color=colour, hatch=hatch)
            res_ax.axvspan(wave_min, wave_max, ymin=-1, 
                        ymax=1, alpha=alpha, color=colour, hatch=hatch)

            if hatch != None:
                axis.axvspan(wave_min, wave_max, ymin=0, 
                        ymax=1.7, alpha=alpha, color=None, hatch=hatch)
                res_ax.axvspan(wave_min, wave_max, ymin=-1, 
                        ymax=1, alpha=alpha, color=None, hatch=hatch)

            previous_true = False

        # Start of shaded region, but hold off drawing
        elif bad_px_mask[xi] and not previous_true:
            wave_min = wave[xi] - wl_delta
            wave_max = wave[xi] + wl_delta
            previous_true = True
        
        # Continue to hold off drawing, but update max
        elif bad_px_mask[xi] and previous_true:
            # Set max
            wave_max = wave[xi] + wl_delta
            continue

        # reset
        else:
            previous_true = False


def plot_chi2_map(
    teff_actual,
    e_teff_actual,
    feh_actual,
    e_feh_actual,
    chi2_map_dict,
    phot_bands,
    n_levels=10,
    point_size=100,
    star_id="",
    source_id="",
    save_path="plots/",
    used_phot=False,
    phot_scale_fac=1,):
    """Visualise chi^2 space as follows:
        1) Teff vs [Fe/H] scatter plot with chi^2 colours
        2) Teff vs [Fe/H] with chi^2 contours
        3) chi^2 values along the valley of minima

    Where 1) and 2) plot the literature value of the star, and track points 
    along the valley.
    """
    # Ensure plot path exists
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Initialise
    plt.close("all")
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=5, ncols=9)
    gs.update(wspace=0.3, hspace=0.7)

    plt.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.05,)

    axes = {}

    axes["cont_ax_b"] = fig.add_subplot(gs[0, 0])
    axes["val_ax_b"] = fig.add_subplot(gs[0, 1])

    axes["cont_ax_r"] = fig.add_subplot(gs[1, 0])
    axes["val_ax_r"] = fig.add_subplot(gs[1, 1])

    axes["cont_ax_p"] = fig.add_subplot(gs[2, 0])
    axes["val_ax_p"] = fig.add_subplot(gs[2, 1])

    axes["cont_ax_all"] = fig.add_subplot(gs[3, 0])
    axes["val_ax_all"] = fig.add_subplot(gs[3, 1])

    # Spectra axes
    black_body_ax = fig.add_subplot(gs[0,2:])
    optical_ax = fig.add_subplot(gs[1,2:])
    nir_ax = fig.add_subplot(gs[2,2:])
    ir_ax = fig.add_subplot(gs[3,2:])

    kinds = ["b", "r", "p", "all"]
    titles = ["Blue", "Red", "Photometry", "Combined"]

    for kind, title in zip(kinds, titles):
        if chi2_map_dict["grid_resid_{}".format(kind)].shape[1] == 0:
            axes["cont_ax_{}".format(kind)].set_visible(False)
            axes["val_ax_{}".format(kind)].set_visible(False)
            continue

        # Extract axes
        #sc_ax = axes["sc_ax_{}".format(kind)]
        cont_ax = axes["cont_ax_{}".format(kind)]
        val_ax = axes["val_ax_{}".format(kind)]

        # Whether to use N levels in logspace instead
        levels = np.logspace(
            np.log10(chi2_map_dict["grid_rchi2_{}".format(kind)].min()), 
            np.log10(chi2_map_dict["grid_rchi2_{}".format(kind)].max()), 
            n_levels)

        # 2) Plot a contour plot with the same data
        cont_col = cont_ax.tricontourf(
            chi2_map_dict["grid_teffs"], 
            chi2_map_dict["grid_fehs"], 
            chi2_map_dict["grid_rchi2_{}".format(kind)], 
            levels)

        # Colour bar
        cb = fig.colorbar(cont_col, ax=cont_ax)
        cb.ax.tick_params(labelsize="xx-small")
        #cb.set_label(cb_label)

        cont_ax.errorbar(
            teff_actual, 
            feh_actual, 
            xerr=e_teff_actual, 
            yerr=e_feh_actual, 
            fmt="ro",
            ecolor="red")

        if title == "Colour":
            cont_ax.set_title("{} (x{})".format(title, phot_scale_fac))
        else:
            cont_ax.set_title(title, fontsize="small")

        #cont_ax.set_xlabel(r"$T_{\rm eff}$", fontsize="small")
        cont_ax.set_ylabel("[Fe/H]]", fontsize="small")
        cont_ax.set_aspect(1./cont_ax.get_data_ratio())
        cont_ax.plot(
            chi2_map_dict["valley_teffs_{}".format(kind)], 
            chi2_map_dict["valley_fehs"], 
            "x--", 
            color="orange")

        cont_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
        cont_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
        cont_ax.tick_params(axis='x', which='major', labelsize="xx-small", rotation=45)
        cont_ax.tick_params(axis='y', which='major', labelsize="xx-small")

        # 3) Plot the valley as its own plot
        val_ax.plot(
            chi2_map_dict["valley_rchi2_{}".format(kind)], 
            chi2_map_dict["valley_fehs"], 
            "x--", 
            color="orange")

        val_ax.set_xlabel("rchi^2", fontsize="small")
        val_ax.set_aspect(1./val_ax.get_data_ratio())
        val_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
        val_ax.tick_params(axis='both', which='major', labelsize="xx-small")

    # For each colour band, show map
    for filt_i, filt in enumerate(phot_bands):
        phot_ax = fig.add_subplot(gs[4, filt_i])

        filt_chi2 = chi2_map_dict["grid_resid_p"][:,filt_i]**2

        levels = np.logspace(
            np.log10(filt_chi2.min()), 
            np.log10(filt_chi2.max()), 
            n_levels)

        cont_col = phot_ax.tricontourf(
            chi2_map_dict["grid_teffs"], 
            chi2_map_dict["grid_fehs"], 
            filt_chi2, 
            levels)

        # Colour bar
        cb = fig.colorbar(cont_col, ax=phot_ax)
        cb.ax.tick_params(labelsize="xx-small")

        phot_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
        phot_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
        phot_ax.tick_params(axis='y', which='major', labelsize="xx-small")
        phot_ax.set_aspect(1./phot_ax.get_data_ratio())

        phot_ax.set_title(filt)

        # Only title leftmost y axis
        if filt_i == 0:
            phot_ax.set_ylabel("[Fe/H]]")

        phot_ax.set_xlabel(r"$T_{\rm eff}$")
        plt.setp(phot_ax.get_xticklabels(), fontsize="xx-small", rotation=45)

    # Black body spectra
    wave_bb = chi2_map_dict["wave_black_body"]
    plot_black_body_axis(black_body_ax, chi2_map_dict, wl_min=0, wl_max=1E6,)

    # Plot optical spectra
    plot_black_body_axis(optical_ax, chi2_map_dict, wl_min=3E3, wl_max=1.1E4,)
    plot_passband(optical_ax, "G", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "BP", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "RP", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "u", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "v", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "g", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "r", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "i", wave_bb, 3E3, 1.1E4)
    plot_passband(optical_ax, "z", wave_bb, 3E3, 1.1E4)

    # Plot NIR spectra
    plot_black_body_axis(nir_ax, chi2_map_dict, wl_min=1E4, wl_max=2.4E4,)
    plot_passband(nir_ax, "J", wave_bb, 1E4, 2.4E4)
    plot_passband(nir_ax, "H", wave_bb, 1E4, 2.4E4)
    plot_passband(nir_ax, "K", wave_bb, 1E4, 2.4E4)

    # Plot IR spectra
    plot_black_body_axis(ir_ax, chi2_map_dict, wl_min=2.5E4, wl_max=6E4,)
    plot_passband(ir_ax, "W1", wave_bb, 2.5E4, 6E4)
    plot_passband(ir_ax, "W2", wave_bb, 2.5E4, 6E4)

    ir_ax.set_xlabel("Wavelength (A)")

    # Title and save the plot
    title = (r"{} ({}), $T_{{\rm eff}}$={:0.0f} K, [Fe/H]={:0.2f}")
    plt.suptitle(title.format(
        star_id, source_id, teff_actual, feh_actual,), y=1.0, fontsize="x-small")
    plt.gcf().set_size_inches(16, 8)
    #plt.tight_layout()
    plt.savefig(save_path + "/chi2_map_{:0.0f}_{}.pdf".format(
        teff_actual, source_id))


def plot_black_body_axis(
    axis, 
    chi2_map_dict,
    wl_min, 
    wl_max,):
    """
    """
    wl_mask = np.logical_and(
        chi2_map_dict["wave_black_body"] > wl_min, 
        chi2_map_dict["wave_black_body"] < wl_max)

    wave_bb = chi2_map_dict["wave_black_body"][wl_mask]

    spec_bb_low_feh = chi2_map_dict["spec_bb_low_feh"][wl_mask]
    spec_bb_low_feh /= np.max(spec_bb_low_feh)

    spec_bb_high_feh = chi2_map_dict["spec_bb_high_feh"][wl_mask]
    spec_bb_high_feh /= np.max(spec_bb_high_feh)

    spec_bb_lit = chi2_map_dict["spec_bb_lit"][wl_mask]
    spec_bb_lit /= np.max(spec_bb_lit)

    axis.plot(wave_bb, spec_bb_low_feh, linewidth=0.2, alpha=0.8, label="low [Fe/H]")
    axis.plot(wave_bb, spec_bb_high_feh, linewidth=0.2, alpha=0.8, label="high [Fe/H]")
    axis.plot(wave_bb, spec_bb_lit, linewidth=0.2, alpha=0.8, label="literature")

    axis.legend(loc="best", fontsize="x-small")


def plot_passband(axis, filt, wave, wl_min, wl_max,):
    """
    """
    PROFILE_COLOURS = {
        "G":"green", 
        "BP":"blue", 
        "RP":"red",
        "u":"mediumpurple",
        "v":"darkviolet",
        "g":"seagreen",
        "r":"tomato",
        "i":"firebrick",
        "z":"maroon",
        "J":"purple",
        "H":"crimson",
        "K":"darkorange",
        "W1":"forestgreen",
        "W2":"darkgreen",
        "W3":"darkcyan",
        "W4":"darkslategrey",}

    # Initialise mask
    wl_mask = np.logical_and(wave > wl_min, wave < wl_max)

    # Load and interpolate onto our wavelength scale
    wl_filt, profile_filt = synth.load_filter_profile(
        filt, min_wl=wave[0], max_wl=wave[1], do_zero_pad=False)

    calc_filt_profile = interp1d(wl_filt, profile_filt, bounds_error=False,
        fill_value=np.nan,)

    profile_filt_interp = calc_filt_profile(wave)

    # Plot
    axis.plot(wave[wl_mask], profile_filt_interp[wl_mask], label=filt, 
        color=PROFILE_COLOURS[filt], linewidth=0.6,)


# -----------------------------------------------------------------------------
# Comparisons & other paper plots
# ----------------------------------------------------------------------------- 
def plot_std_comp_generic(fig, axis, fit, e_fit, lit, e_lit, colour, fit_label, 
    lit_label, cb_label, x_lims, y_lims, cmap, show_offset, ticks, 
    resid_y_lims=None, plot_scatter=True, ms=2, text_labels=None, 
    print_labels=False, elinewidth=0.5, offset_sig_fig=2,):
    """
    Parameters
    ----------
    fit, axis: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        Figure and axis to plot on.
    
    fit, e_fit: float array
        Fitted parameter (i.e. Teff or [Fe/H]) and uncertainty.

    lit, e_lit: float array
        Literature parameter (i.e. Teff or [Fe/H]) and uncertainty.

    colour: float array
        Parameter to plot on colour bar (i.e. Teff or [Fe/H]).

    x_label, y_label, cb_label: string
        Labels for axes and colour bar respectively.

    lims: float array
        Axis limits, of form [low, high].

    cmap: string
        Matplotlib colourmap to use for the colour bar.

    show_offset: bool, default: False
        Whether to plot the median uncertainty as text.
    """
    # Plot error bars with overplotted scatter points + colour bar
    axis.errorbar(
        lit, 
        fit, 
        xerr=e_lit,
        yerr=e_fit,
        fmt=".",
        zorder=0,
        ecolor="black",
        markersize=ms,
        elinewidth=elinewidth,)

    if plot_scatter:
        sc = axis.scatter(lit, fit, c=colour, zorder=1, cmap=cmap)

        cb = fig.colorbar(sc, ax=axis)
        cb.ax.tick_params(labelsize="large")

        if cb_label != "":
            cb.set_label(cb_label, fontsize="x-large")

    # Split lims if we've been given different x and y limits
    lim_min = np.min([x_lims[0], y_lims[0]])
    lim_max = np.min([x_lims[1], y_lims[1]])

    # Plot 1:1 line
    xx = np.arange(lim_min, lim_max, (lim_max-lim_min)/100)
    axis.plot(xx, xx, "k--", zorder=0)

    # Plot residuals
    divider = make_axes_locatable(axis)
    resid_ax = divider.append_axes("bottom", size="30%", pad=0)
    axis.figure.add_axes(resid_ax, sharex=axis)

    resid = lit - fit

    # Treat asymmetric errorbars if we have them
    e_lit = np.asarray(e_lit)
    
    if len(e_lit.shape) > 1:
        e_resid = np.array([
            np.sqrt(e_lit[0]**2 + e_fit**2),
            np.sqrt(e_lit[1]**2 + e_fit**2),
        ])
    else:
        e_resid = np.sqrt(e_lit**2 + e_fit**2)

    if plot_scatter:
        resid_ax.scatter(
            lit,
            resid,
            c=colour,
            cmap=cmap,
            zorder=1,)

    resid_ax.errorbar(
        lit,
        resid,
        xerr=e_lit,
        yerr=e_resid,
        fmt=".",
        zorder=0,
        ecolor="black",
        markersize=ms,
        elinewidth=elinewidth,)

    # Plot 0 line
    plt.setp(axis.get_xticklabels(), visible=False)
    resid_ax.hlines(0, lim_min, lim_max, linestyles="--", zorder=0)
    
    resid_ax.set_xlabel(lit_label, fontsize="x-large")

    if fit_label != "":
        axis.set_ylabel(fit_label, fontsize="x-large")
        resid_ax.set_ylabel("resid", fontsize="x-large")

    axis.set_xlim(x_lims)
    resid_ax.set_xlim(x_lims)
    axis.set_ylim(y_lims)

    #axis.set_aspect("equal")

    # Print labels if we've been given them (mostly diagnostic)
    if print_labels and text_labels is not None:
        for obj_i in range(len(text_labels)):
            if np.isnan(np.sum([lit[obj_i], fit[obj_i]])):
                continue
                
            axis.text(
                lit[obj_i],
                fit[obj_i], 
                text_labels[obj_i],
                fontsize=5,
                horizontalalignment="center",)
            
            resid_ax.text(
                lit[obj_i],
                resid[obj_i], 
                text_labels[obj_i],
                fontsize=5,
                horizontalalignment="center",)


    # Ticks
    resid_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=ticks[1]))
    resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=ticks[0]))

    resid_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=ticks[3]))
    resid_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=ticks[2]))

    axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=ticks[1]))
    axis.yaxis.set_major_locator(plticker.MultipleLocator(base=ticks[0]))
    
    axis.tick_params(axis='both', which='major', labelsize="large")
    resid_ax.tick_params(axis='x', which='major', labelsize="large")

    # Set limits on y residuals if given
    if resid_y_lims is not None:
        resid_ax.set_ylim(resid_y_lims)

    # Show mean and std
    if show_offset:
        mean_offset = np.nanmedian(fit - lit)
        std = np.nanstd(fit - lit)

        offset_lbl = (r"${:0." + str(int(offset_sig_fig)) + r"f}\pm {:0." 
                      + str(int(offset_sig_fig)) + r"f}$")

        axis.text(
            x=((x_lims[1]-x_lims[0])/2 + x_lims[0]), 
            y=0.05*(y_lims[1]-y_lims[0])+y_lims[0], 
            s=offset_lbl.format(mean_offset, std),
            horizontalalignment="center",
            fontsize="x-large")


def plot_std_comp(
    observations, 
    std_info,
    show_offset=False,
    fn_suffix="",
    title_text="",
    teff_ticks=(500,250,150,75),
    feh_ticks=(0.5,0.25,0.6,0.3),
    feh_resid_y_lims=(-0.8,1.5),
    feh_cb_label="[Fe/H] (phot)",
    teff_syst=-30,
    undo_teff_syst=False,):
    """Plot 2x3 grid of Teff and [Fe/H] literature comparisons.
        1 - Mann+15 Teffs
        2 - Rojas-Ayala+12 Teffs
        3 - Interferometric Teffs
        4 - Mann+15 [Fe/H]
        5 - Rojas-Ayala+12 [Fe/H]
        6 - CPM [Fe/H]

    Where Mann+15:
     - https://ui.adsabs.harvard.edu/abs/2015ApJ...804...64M/abstract

    And Rojas-Ayala+12:
     - https://ui.adsabs.harvard.edu/abs/2012ApJ...748...93R/abstract

    Saves as paper/std_comp<fn_suffix>.<pdf/png>.

    Parameters
    ----------
    observations: pandas.DataFrame
        Table of observations and fit results.

    std_cat: pandas.DataFrame
        Corresponding table of stellar literature info.

    teff_lims, feh_lims: float array, default:[3000,4600],[-1.4,0.75]
        Axis limits for Teff and [Fe/H] respectively.

    show_offset: bool, default: False
        Whether to plot the median offset as text.

    fn_suffix: string, default: ''
        Suffix to append to saved figures
        
    title_text: string, default: ''
        Text for fig.suptitle.
    """
    # Table join
    #observations.rename(columns={"uid":"source_id"}, inplace=True)
    obs_join = observations.join(std_info, "source_id", rsuffix="_info")

    if undo_teff_syst:
        obs_join["teff_synth"] -= teff_syst

    plt.close("all")

    # Make one plot for Teff comparison
    fig_teff, axes_teff = plt.subplots(1,4)
    (ax_teff_m15, ax_teff_ra12, ax_teff_int, ax_teff_other) = axes_teff
    fig_teff.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, 
                             wspace=0.5)

    # And another for [Fe/H]
    fig_feh, axes_feh = plt.subplots(1,4)
    (ax_feh_m15, ax_feh_ra12, ax_feh_cpm, ax_feh_other) = axes_feh
    fig_feh.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, 
                             wspace=0.5)

    # Mann+15 temperatures
    plot_std_comp_generic(
        fig_teff,
        ax_teff_m15, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_m15"],
        obs_join["e_teff_m15"],
        obs_join["feh_synth"],
        r"$T_{\rm eff}$ (K, fit)",
        r"$T_{\rm eff}$ (K, Mann+15)",
        "", #"[Fe/H] (phot)",
        x_lims=(2800,4300),
        y_lims=(2800,4300),
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,)
    
    # Rojas-Ayala+12 temperatures
    plot_std_comp_generic(
        fig_teff,
        ax_teff_ra12, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_ra12"],
        obs_join["e_teff_ra12"],
        obs_join["feh_synth"],
        "", #r"$T_{\rm eff}$ (K, fit)",
        r"$T_{\rm eff}$ (K, Rojas-Ayala+12)",
        "", #feh_cb_label,
        x_lims=(2800,4300),
        y_lims=(2800,4300),
        cmap="viridis",
        show_offset=show_offset,
        ticks=(500,250,200,100),)
    
    # Interferometric temperatures
    plot_std_comp_generic(
        fig_teff,
        ax_teff_int, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_int"],
        obs_join["e_teff_int"],
        obs_join["feh_synth"],
        "", #r"$T_{\rm eff}$ (K, fit)",
        r"$T_{\rm eff}$ (K, interferometric)",
        "", #feh_cb_label,
        x_lims=(2800,5100),
        y_lims=(2800,5100),
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,)

    # Other temperatures
    plot_std_comp_generic(
        fig_teff,
        ax_teff_other, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_other"],
        obs_join["e_teff_other"],
        obs_join["feh_synth"],
        "", #r"$T_{\rm eff}$ (K, fit)",
        r"$T_{\rm eff}$ (K, other)",
        feh_cb_label,
        x_lims=(3400,5100),
        y_lims=(3400,5100),
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,)

    # Mann+15 [Fe/H]
    plot_std_comp_generic(
        fig_feh,
        ax_feh_m15, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_m15"],
        obs_join["e_feh_m15"],
        obs_join["teff_synth"],
        r"[Fe/H] (fit)",
        r"[Fe/H] (Mann+15)",
        "",#r"$T_{\rm eff}\,$K (fit)",
        x_lims=(-0.65,0.6),
        y_lims=(-0.95,0.6),
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        resid_y_lims=(-0.8,1.5),)
    
    # Rojas-Ayala+12 [Fe/H]
    plot_std_comp_generic(
        fig_feh,
        ax_feh_ra12, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_ra12"],
        obs_join["e_feh_ra12"],
        obs_join["teff_synth"],
        "",#r"[Fe/H] (fit)",
        r"[Fe/H] (Rojas-Ayala+12)",
        "",#r"$T_{\rm eff}\,$K (fit)",
        x_lims=(-0.8,0.6),
        y_lims=(-1.2,0.6),
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        resid_y_lims=(-0.8,1.2),)

    # CPM [Fe/H]
    plot_std_comp_generic(
        fig_feh,
        ax_feh_cpm, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_cpm"],
        obs_join["e_feh_cpm"],
        obs_join["teff_synth"],
        "", #r"[Fe/H] (fit)",
        r"[Fe/H] (binary primary)",
        "",#r"$T_{\rm eff}\,$K (fit)",
        x_lims=(-1.2,0.6),
        y_lims=(-1.2,0.6),
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        resid_y_lims=(-0.8,0.7),)

    # Other [Fe/H]
    plot_std_comp_generic(
        fig_feh,
        ax_feh_other, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_other"],
        obs_join["e_feh_other"],
        obs_join["teff_synth"],
        "", #r"[Fe/H] (fit)",
        r"[Fe/H] (other)",
        r"$T_{\rm eff}\,$K (fit)",
        x_lims=(-1.1,0.6),
        y_lims=(-0.95,0.6),
        cmap="magma",
        show_offset=show_offset,
        ticks=feh_ticks,
        resid_y_lims=(-1.0,1.05),)

    #fig.suptitle(title_text)

    # Save Teff plot
    fig_teff.set_size_inches(16, 3)
    fig_teff.tight_layout()
    fig_teff.savefig("paper/std_comp_teff{}.pdf".format(fn_suffix))
    fig_teff.savefig("paper/std_comp_teff{}.png".format(fn_suffix))

    # Save [Fe/H] plot
    fig_feh.set_size_inches(16, 3)
    fig_feh.tight_layout()
    fig_feh.savefig("paper/std_comp_feh{}.pdf".format(fn_suffix))
    fig_feh.savefig("paper/std_comp_feh{}.png".format(fn_suffix))


def plot_teff_comp(
    observations, 
    std_info,
    x_col="teff_synth",
    phot_teff_col="teff_m15_bprp",
    cb_col="feh_synth",
    teff_lims=[3000,4600],
    feh_lims=[-1.4,0.75],
    mask_outside_lims=True,
    show_median_offset=True,
    fn_suffix="",
    title_text="",):
    """Plots a comparison between Teff from colour relations and the fitted
    spectroscopic Teff.

    Saves as paper/teff_comp<fn_suffix>.<pdf/png>.

    Parameters
    ----------
    observations: pandas.DataFrame
        Table of observations and fit results.

    std_cat: pandas.DataFrame
        Corresponding table of stellar literature info.

    phot_teff_col: string, default: 'teff_m15_bprp'
        Photometric relation Teff column to use.

    teff_lims: float array, default:[3000,4600]
        Axis limits for Teff.

    mask_outside_lims: bool, default: True
        Whether to only plot and consider stars inside the limits.

    show_median_offset: bool, default: False
        Whether to plot the median offset as text.

    fn_suffix: string, default: ''
        Suffix to append to saved figures
        
    title_text: string, default: ''
        Text for fig.suptitle.
    """
    # Table join
    #observations.rename(columns={"uid":"source_id"}, inplace=True)
    obs_join = observations.join(
        std_info, 
        "source_id", 
        rsuffix="_info", 
        how="inner")

    # Mask stars outside the Teff and [Fe/H] limits
    if mask_outside_lims:
        mask_teff = np.logical_and(
            obs_join["teff_synth"] > teff_lims[0],
            obs_join["teff_synth"] < teff_lims[1],)
        mask_feh = np.logical_and(
            obs_join["feh_synth"] > feh_lims[0],
            obs_join["feh_synth"] < feh_lims[1],)
        obs_join = obs_join[np.logical_and(mask_teff, mask_feh)]

    plt.close("all")
    fig, axis = plt.subplots()
    
    # Plot error bars, and overplot scatter points with a colour bar
    axis.errorbar(
        obs_join[x_col],
        obs_join[phot_teff_col],
        xerr=obs_join["e_{}".format(x_col)],
        yerr=obs_join["e_{}".format(phot_teff_col)],
        fmt=".", 
        zorder=0,)

    sc = axis.scatter(
        obs_join[x_col], 
        obs_join[phot_teff_col], 
        c=obs_join[cb_col], 
        zorder=1, 
        cmap="viridis")

    cb = fig.colorbar(sc, ax=axis)
    cb.set_label("[Fe/H]")

    fig.suptitle(title_text)
    axis.set_xlabel(r"$T_{\rm eff}\,$K (fit)")
    axis.set_ylabel(r"$T_{\rm eff}\,$K (rel)")

    # Plot 1:1 line
    xx = np.arange(teff_lims[0], teff_lims[1], (teff_lims[1]-teff_lims[0])/100)
    axis.plot(xx, xx, "--")

    axis.set_xlim(teff_lims)
    axis.set_ylim(teff_lims)

    axis.set_aspect("equal")

    # Plot median offset from photometric temperatures
    if show_median_offset:
        med_offset = np.nanmedian(
            np.abs(obs_join["teff_synth"] - obs_join[phot_teff_col]))
        axis.text(
            x=np.mean(teff_lims), 
            y=0.95*teff_lims[1], 
            s=r"$\pm {:0.2f}\,$K".format(med_offset),
            horizontalalignment="center")

    plt.savefig("paper/teff_comp{}.pdf".format(fn_suffix))
    plt.savefig("paper/teff_comp{}.png".format(fn_suffix))


def plot_tic_stellar_param_comp(
    show_offset=True,
    teff_ticks=(400,200,250,125),
    rad_ticks=(0.1,0.05,0.1,0.05),
    ms=5,):
    """
    """
    # Import
    # Load in literature info for our stars
    tic_info = utils.load_info_cat(
        remove_fp=True,
        only_observed=True,
        use_mann_code_for_masses=False,
        do_extinction_correction=False,).reset_index()
    tic_info.set_index("TIC", inplace=True)

    # Load NASA ExoFOP info on TOIs
    toi_info = utils.load_exofop_toi_cat(do_ctoi_merge=True)

    # Load in info on observations and fitting results
    observations = utils.load_fits_table("OBS_TAB", "tess", path="spectra",)

    # Temporary join to get combined info in single datastructure
    info = toi_info.join(tic_info, on="TIC", how="inner", lsuffix="", rsuffix="_2")
    comb_info = info.join(
        observations, 
        on="source_id", 
        lsuffix="", 
        rsuffix="_2", 
        how="inner")

    # Remove those with radii set to solar
    comb_info = comb_info[comb_info["Stellar Radius (R_Sun)"] != 1.0]

    plt.close("all")

    fig, (ax_teff, ax_rad) = plt.subplots(1,2)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.5)

    # Temperatures
    plot_std_comp_generic(
        fig,
        ax_teff, 
        comb_info["teff_synth"],
        comb_info["e_teff_synth"],
        comb_info["Stellar Teff (K)"],
        comb_info["Stellar Teff error"],
        None,
        r"$T_{\rm eff}$ (K, fit)",
        r"$T_{\rm eff}$ (K, TIC)",
        "", #"[Fe/H] (phot)",
        x_lims=(2900,4800),
        y_lims=(2900,4800),
        cmap="viridis",
        show_offset=show_offset,
        ticks=teff_ticks,
        plot_scatter=False,
        ms=ms,)

    # Radii
    plot_std_comp_generic(
        fig,
        ax_rad, 
        comb_info["radius"],
        comb_info["e_radius"],
        comb_info["Stellar Radius (R_Sun)"],
        comb_info["Stellar Radius error"],
        None,
        r"$R_{\star}$ ($R_{\odot}$, fit)",
        r"$R_{\star}$ ($R_{\odot}$, TIC)",
        "",
        x_lims=(0.15,0.85),
        y_lims=(0.15,0.85),
        cmap="viridis",
        show_offset=show_offset,
        ticks=rad_ticks,
        plot_scatter=False,
        ms=ms,)

    # Save Teff plot
    #fig.set_size_inches(8, 3)
    fig.tight_layout()
    fig.savefig("paper/tic_comp.pdf")
    fig.savefig("paper/tic_comp.png")


def plot_cmd(
    info_cat, 
    info_cat_2=None,
    plot_toi_ids=False,
    colour="Bp-Rp",
    abs_mag="G_mag_abs",
    x_label=r"$B_P-R_P$",
    y_label=r"$M_{\rm G}$",
    label="tess",
    feh_col="feh_m15",
    plot_feh_cb=False,):
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

    if plot_feh_cb:
        colours = info_cat[feh_col]
    else:
        colours = None

    # Plot our first set of stars
    scatter = axis.scatter(
        info_cat[colour], 
        info_cat[abs_mag], 
        zorder=1,
        c=colours,
        label="Science",
        alpha=0.9,
    )

    if plot_feh_cb:
        cb = fig.colorbar(scatter, ax=axis)
        cb.set_label("[Fe/H]")

    # Plot a second set of stars behind (e.g. standards)
    if info_cat_2 is not None:
        scatter = axis.scatter(
            info_cat_2[colour], 
            info_cat_2[abs_mag], 
            marker="o",
            edgecolor="#ff7f0e",
            facecolors="none",
            zorder=0,
            label="Standard"
        )

        plt.legend(loc="best", fontsize="large")

    # Optional, but for diagnostic purposes plot the target TOI IDs
    if plot_toi_ids:
        for star_i, star in info_cat.iterrows():
            axis.text(
                star[colour], 
                star[abs_mag]-0.1,
                star["TOI"],
                horizontalalignment="center",
                fontsize="xx-small"
            )

    # Flip magnitude axis
    ymin, ymax = axis.get_ylim()
    axis.set_ylim((ymax, ymin))

    axis.set_xlabel(x_label, fontsize="large")
    axis.set_ylabel(y_label, fontsize="large")

    axis.tick_params(axis='both', which='major', labelsize="large")

    axis.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

    fig.tight_layout()
    plt.savefig("paper/{}_cmd.png".format(label))
    plt.savefig("paper/{}_cmd.pdf".format(label))


def plot_hr_diagram(
    observations, 
    info_cat, 
    logg_col="logg_m19",
    teff_lims=[3000,4600],
    logg_lims=[4.45,5.2],
    plot_ids=False,
    id_col="TIC",):
    """Plot a Teff vs logg HR diagram for targets
    
    Parameters
    ----------
    observations: pandas.DataFrame
        Table of observations and fit results.

    info_cat: pandas.DataFrame
        Corresponding table of stellar literature info.

    logg_col: string, default: 'logg_m19'
        Logg column to use.

    teff_lims, logg_lims: float array, default:[3000,4600],[4.45,5.2]
        Axis limits for Teff and Logg respectively.
    
    plot_ids: bool, default: False
        Whether to plot IDs, useful for diagnostics.

    id_col: string, default: 'TIC'
        ID to plot.
    """
    # Table join
    obs_join = observations.join(info_cat, "source_id", rsuffix="_info")

    # Plot errors, then coloured [Fe/H] scatter points on top of them
    plt.close("all")
    plt.errorbar(
        obs_join["teff_synth"],
        obs_join[logg_col],
        xerr=obs_join["e_teff_synth"],
        yerr=obs_join["e_{}".format(logg_col)],
        fmt=".",
        zorder=0,)

    sc = plt.scatter(
        obs_join["teff_synth"],
        obs_join[logg_col],
        c=obs_join["feh_synth"],
        zorder=1,
    )

    # Plot IDs for diagnostic purposes
    if plot_ids:
        for star_i, star in obs_join.iterrows():
            plt.text(
                star["teff_synth"],
                star[logg_col],
                star[id_col])

    cb = plt.colorbar(sc)
    cb.set_label("[Fe/H]")

    plt.xlabel(r"$T_{\rm eff}$ (K)")
    plt.ylabel(r"$\logg$")
    plt.xlim(teff_lims[::-1])
    plt.ylim(logg_lims[::-1])

    plt.savefig("paper/tess_hr.pdf")
    plt.savefig("paper/tess_hr.png")


def plot_radius_comp(
    observations,
    info_cat,
    use_interferometric_radii=True,
    uncertainty_pc_limit=0.05,
    lims=(0.14,0.81),
    ms=10):
    """Plot a comparison between our radii determined here vs their Mann+15 
    equivalents. Has colourbar for [Fe/H].

    Parameters
    ----------
    observations: pandas.DataFrame
        Table of observations and fit results.

    info_cat: pandas.DataFrame
        Corresponding table of stellar literature info.
    
    use_interferometric_radii: boolean
    """
    # Table join
    obs_join = observations.join(info_cat, "source_id", rsuffix="_info")

    # 
    if use_interferometric_radii:
        label = "Interferometric"
        lit_radii_col = "radius_int"
        e_lit_radii_col = "e_radius_int"
    else:
        label = "Mann+15"
        lit_radii_col = "radii_m19"
        e_lit_radii_col = "e_lit_radii_col"

    # Only take those stars with uncertainties below our threshold
    e_rad_pc_lit = obs_join[e_lit_radii_col] / obs_join[lit_radii_col]

    obs_join = obs_join[e_rad_pc_lit < uncertainty_pc_limit]

    # Plot errors, then coloured [Fe/H] scatter points on top of them
    plt.close("all")
    fig, axis = plt.subplots()

    #ms = 0.2

    axis.errorbar(
        obs_join[lit_radii_col],
        obs_join["radius"],
        xerr=obs_join[e_lit_radii_col],
        yerr=obs_join["e_radius"],
        fmt=".",
        zorder=0,
        markersize=ms,
        ecolor="black",)

    sc = axis.scatter(
        obs_join[lit_radii_col],
        obs_join["radius"],
        c=obs_join["K_mag"],
        zorder=1,
        cmap="plasma",
    )

    # Setup lower panel for residuals
    plt.setp(axis.get_xticklabels(), visible=False)
    divider = make_axes_locatable(axis)
    res_ax = divider.append_axes("bottom", size="30%", pad=0)
    axis.figure.add_axes(res_ax, sharex=axis)

    resid = obs_join[lit_radii_col] - obs_join["radius"]
    e_resid = np.sqrt(obs_join["e_radius"]**2 + obs_join[e_lit_radii_col]**2)

    res_ax.errorbar(
        obs_join[lit_radii_col],
        resid,
        xerr=obs_join[e_lit_radii_col],
        yerr=e_resid,
        fmt=".",
        zorder=0,
        markersize=ms,
        ecolor="black",)

    res_ax.scatter(
        obs_join[lit_radii_col],
        resid,
        c=obs_join["K_mag"],
        zorder=1,
        cmap="plasma",
    )

    # Plot horizonal line
    res_ax.hlines(
        0,
        obs_join[lit_radii_col].min(),
        obs_join[lit_radii_col].max(),
        linestyles="--",
        color="black",
        zorder=0,)

    cb = plt.colorbar(sc, ax=axis)
    cb.set_label(r"$K_S$", fontsize="x-large")
    cb.ax.tick_params(labelsize="x-large")

    # Plot 1:1 line
    axis.plot(
        np.arange(0.1,0.9,0.1),
        np.arange(0.1,0.9,0.1),
        "--",
        color="black",
        zorder=0,)

    axis.set_ylabel(r"Radius ($R_\odot$, fit)", fontsize="x-large")
    res_ax.set_ylabel(r"resid", fontsize="x-large")
    res_ax.set_xlabel(r"Radius ($R_\odot$, {})".format(label), 
        fontsize="x-large")

    axis.tick_params(axis='both', which='major', labelsize="large")
    res_ax.tick_params(axis='both', which='major', labelsize="large")

    axis.set_xlim(lims)
    axis.set_ylim(lims)
    res_ax.set_xlim(lims)

    # Ticks
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))
    axis.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))
    axis.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

    res_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))
    res_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    res_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
    res_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.04))

    plt.tight_layout()

    fig.savefig("paper/radius_comp_{}.pdf".format(label))
    fig.savefig("paper/radius_comp_{}.png".format(label))


def plot_fbol_comp(
    observations,
    info_cat,
    fbols=["Bp", "Rp", "J", "H", "K", "avg"] , 
    ncols=10,
    id_col="TIC"):
    """Plot a comparison of the sampled values of fbol from each filter to 
    check whether they are consistent or not.
    """
    nrows = int(np.ceil(len(observations)/ncols))
    plt.close("all")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.25, 
        top=0.975, 
        bottom=0.025,
        left=0.075,
        right=0.95,)
    
    # Do temporary crossmatch
    obs_cm = observations.join(info_cat, "source_id", rsuffix="_2")

    # Sort dataframe by Teff - cool stars first, hot stars last
    sorted_obs = obs_cm.sort_values("teff_synth")

    # Define bands to reference, construct new headers
    band_lbl = [r"$f_{{\rm bol, {}}}$".format(fbol) for fbol in fbols]
    
    f_bol_cols= ["f_bol_{}".format(fbol) for fbol in fbols]
    e_f_bol_cols = ["e_f_bol_{}".format(fbol) for fbol in fbols]
    
    colours = ["green", "blue", "orange", "deepskyblue", "red", "black"]
    
    ids = sorted_obs[id_col].values
    
    fbols = sorted_obs[f_bol_cols].values
    e_fbols = sorted_obs[e_f_bol_cols].values

    # Only plot axes with data
    n_unused_ax = len(fbols) - len(axes)

    for ax_i in np.arange(-1, n_unused_ax-1, -1):
        axes[ax_i].axis("off")

    # Plot a subplot with the fluxes for each star
    for ax_i, (ax, fbol, e_fbol) in enumerate(zip(axes, fbols, e_fbols)):

        ax.errorbar(band_lbl, fbol, yerr=e_fbol, elinewidth=0.3,
                    fmt=".", zorder=1, ecolor="black",capsize=1,
                    capthick=0.3, 
                    markersize=0.1)
        ax.scatter(band_lbl, fbol, s=1**4, c=colours, zorder=2,) 
        
        #ax.yaxis.get_major_formatter().set_powerlimits((0,1))
        ax.set_title(ids[ax_i], fontsize="xx-small")
        ax.tick_params(axis='y', which='minor', labelsize="xx-small")
        ax.tick_params(axis='y', which='major', labelsize="xx-small")
        
        plt.setp(ax.get_xticklabels(), fontsize="xx-small", rotation="vertical")
        ax.yaxis.offsetText.set_fontsize("xx-small")
        
        # Only display x ticks if on bottom
        if ax_i < (len(fbols) - ncols):
            ax.get_xaxis().set_ticklabels([])
    
    #plt.tight_layout()
    #fig.text(0.5, 0.025, r"BC Filter Band", ha='center')
    fig.text(
        0.025, 
        0.5, 
        r"$f_{\rm bol}$ (erg$\,$s$^{-1}\,$cm$^{-2}$)", 
        va='center', 
        rotation='vertical')

    plt.gcf().set_size_inches(9, 12)
    plt.savefig("paper/fbol_comp.pdf")


def plot_representative_spectral_model_limitations(
    source_id,
    label="std",
    spec_path="spectra",
    btsettl_path="phoenix",
    btsettl_grid_point=(3200,5.0),
    blue_conv_res=0.77*10,
    red_conv_res=0.44*10,
    plot_size=(9,2.5),
    plot_suffix="",
    lw=0.3,):
    """Plots the observed, MARCS literature, and closest BT-Settl spectra with
    overplotted filters to compare. Some sample stars:

        ~3200 K --> Gl 447 (Gaia DR2 3796072592206250624)
        ~3500 K --> Gl 876 (Gaia DR2 2603090003484152064 )
        ~4000 K --> PM I12507-0046 (Gaia DR2 3689602277083844480)
        ~4500 K --> HD 131977 (Gaia DR2 6232511606838403968)
    """
    # Import observed spectra and merge
    spec_b_all, spec_r_all, obs_std = utils.load_fits(label, path=spec_path)

    # Get index
    star_i = int(np.argwhere(obs_std.index==source_id))

    wl_br, spec_br = spec.merge_wifes_arms(
        spec_b_all[star_i,0],
        spec_b_all[star_i,1],
        spec_r_all[star_i,0],
        spec_r_all[star_i,1],)

    # Import literature MARCS spectra and merge
    synth_lit_b_all = utils.load_fits_image_hdu("synth_lit", label, arm="b")
    synth_lit_r_all = utils.load_fits_image_hdu("synth_lit", label, arm="r")

    _, spec_lit = spec.merge_wifes_arms(
        spec_b_all[star_i,0],
        synth_lit_b_all[star_i],
        spec_r_all[star_i,0],
        synth_lit_r_all[star_i],)

    # Import BT-Settl spectra and merge
    wl_bts_b, spec_bts_b, wl_bts_r, spec_bts_r = synth.import_btsettl_spectra(
        blue_conv_res=blue_conv_res,
        red_conv_res=red_conv_res)
    wl_bts, spec_bts = spec.merge_wifes_arms(
        wl_bts_b, spec_bts_b, wl_bts_r, spec_bts_r)

    # Plot
    plt.close("all")
    plt.plot(wl_br, spec_br, linewidth=lw, label="Observed", alpha=0.7, zorder=2)
    plt.plot(wl_br, spec_lit, "--", linewidth=lw,  label="MARCS", alpha=0.7, zorder=1)
    plt.plot(wl_bts, spec_bts, ":", linewidth=lw, label="BT-Settl", alpha=0.7, zorder=0)

    # Filter profiles
    filters = ["v", "g", "r", "BP", "RP"]
    filter_labels = ["v", "g", "r", "B_P", "R_P"]

    for filt_i, (filt, lbl) in enumerate(zip(filters, filter_labels)):
        wl_f, fp = synth.load_filter_profile(filt, 3000, 7000, do_zero_pad=True)
        plt.plot(wl_f, fp*2, linewidth=1.0, label=r"${}$".format(lbl), zorder=3)

    plt.xlabel(r"Wavelength ($\mathrm{\AA}$)")
    plt.ylabel(r"$f_\lambda$ (Normalised)")
    plt.xlim(3200, 7000)
    leg = plt.legend(loc="upper center", ncol=8)

    # Update width of legend objects
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    plt.tight_layout()
    plt.savefig("paper/model_spectra_limitations{}.pdf".format(plot_suffix))
    plt.savefig(
        "paper/model_spectra_limitations{}.png".format(plot_suffix),
        dpi=500,)


def plot_label_comparison(
    observations, 
    std_info,
    teff_lims=(2800,6500),
    logg_lims=(4.0,6.0),
    feh_lims=(-2,1.0),
    teff_axis_step=200,
    logg_axis_step=0.5,
    feh_axis_step=0.5):
    """Plot comparison between actual labels and predicted labels for 
    Teff, logg, and [Fe/H].

    Parameters
    ----------
    label_values: 2D numpy array
        Label array with columns [teff, logg, feh]
    
    labels_pred: 2D numpy array
        Predicted label array with columns [teff, logg, feh]
    """
    plt.close("all")
    fig, (ax_teff, ax_logg, ax_feh) = plt.subplots(1, 3, figsize=(12, 4)) 
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.95, 
                        wspace=0.5)

    lit_source = [ls for ls in list(set(std_info["source"].values)) if type(ls) == str]
    
    # Only plot standards in std_info that are marked as observed
    obs_stds = std_info[std_info["observed"]]

    cms_mask = np.isin(observations["uid"], obs_stds["source_id"])
    cms_obs = observations[cms_mask]

    # Now for every observed star in cms_obs, plot params vs lit equivalents
    for star_i, star in cms_obs.iterrows(): 
        star_lit_info = obs_stds[obs_stds["source_id"]==star["uid"]].iloc[0]

        ax_teff.errorbar(
            star["teff_synth"], 
            star_lit_info["teff"], 
            yerr=star_lit_info["e_teff"],
            fmt="o")

        ax_logg.errorbar(
            star["logg_synth"], 
            star_lit_info["logg"], 
            yerr=star_lit_info["e_logg"],
            fmt="o")

        ax_feh.errorbar(
            star["feh_synth"], 
            star_lit_info["feh"], 
            yerr=star_lit_info["e_feh"],
            fmt="o")

    xy = np.arange(teff_lims[0],teff_lims[1])
    ax_teff.plot(xy, xy, "-", color="black")
    ax_teff.set_xlabel(r"T$_{\rm eff}$ (Synth)")
    ax_teff.set_ylabel(r"T$_{\rm eff}$ (Lit)")

    xy = np.arange(logg_lims[0],logg_lims[1], 0.1)
    ax_logg.plot(xy, xy, "-", color="black")
    ax_logg.set_xlabel(r"$\log g$ (Synth)")
    ax_logg.set_ylabel(r"$\log g$ (Lit)")

    xy = np.arange(feh_lims[0],feh_lims[1], 0.1)
    ax_feh.plot(xy, xy, "-", color="black")
    ax_feh.set_xlabel(r"[Fe/H] (Synth)")
    ax_feh.set_ylabel(r"[Fe/H] (Lit)") 

    plt.savefig("plots/std_label_comp.pdf")
    # Plot every source separately
    #for source in lit_source:
        # Sort the array
        #mm = np.isin(observations["uid"], std_info[std_info["source"]==source]["source_id"])  
        #obs = observations[mm]

# -----------------------------------------------------------------------------
# Planet plots
# ----------------------------------------------------------------------------- 
def plot_double_planet_radii_hist(lc_results, bin_width, min_rp=0.1,
    plot_poisson_errors=True, plot_smooth_hist=False, x_lims=(0,14), 
    logrhk_threshold=-5,):
    """
    """
    plt.close("all")
    fig, (axis_all, axis_active) = plt.subplots(1,2)

    # Import and crossmatch with activity TODO: do this properly
    observations = utils.load_fits_table("OBS_TAB", "tess", path="spectra")
    observations = utils.merge_activity_table_with_obs(
        observations, "tess", fix_missing_source_id=True)

    # TIC info
    tic_info = utils.load_info_cat(
        remove_fp=True,
        only_observed=True,
        use_mann_code_for_masses=False,).reset_index()
    tic_info.set_index("TIC", inplace=True)

    # Combine
    info = lc_results.join(
        tic_info, on="TIC", how="inner", lsuffix="", rsuffix="_2") 
    comb_info = info.join(
        observations, on="source_id", lsuffix="", rsuffix="_2", how="inner")

    active_mask = comb_info["logR'HK"] > logrhk_threshold
    active_tic_results = comb_info[active_mask]

    # Plot standard hist
    plot_planet_radii_hist(lc_results, bin_width, min_rp, plot_poisson_errors, 
        plot_smooth_hist, x_lims,axis=axis_all)

    # And histogram for active planets only
    plot_planet_radii_hist(active_tic_results, bin_width, min_rp, 
        plot_poisson_errors, plot_smooth_hist, x_lims,axis=axis_active)



def plot_planet_radii_hist(lc_results, bin_width=0.4, min_rp=0.4,
    plot_poisson_errors=True, plot_smooth_hist=False, x_lims=(0,14),
    plot_activity_hist=False, axis=None,):
    """Plots a histogram of stellar radii.

    Parameters
    ----------
    lc_results: pandas.core.frame.DataFrame
        Results DataFrame from light curve fitting

    bin_width: float, default: 0.4
        The width of the histogram bins in R_E
    """
    if axis is None:
        fig, axis = plt.subplots()

    min_rp = np.min(lc_results["rp_fit"][lc_results["rp_fit"] > min_rp])
    max_rp = np.max(lc_results["rp_fit"])

    # If plotting smooth histogram (i.e. sum of Gaussians)
    if plot_smooth_hist:
        rr = np.linspace(-20, 40, 100000)
        density = []

        for toi, res in lc_results.iterrows():
            if np.isnan(res["rp_fit"]) or np.isnan(res["e_rp_fit"]):
                continue

            density.append(
                norm.pdf(
                    rr, 
                    loc=res["rp_fit"], 
                    scale=res["e_rp_fit"]))
        density = np.sum(density, axis=0) 
        density /= np.sum([~np.isnan(lc_results["rp_fit"])])

        axis.fill_between(rr, density, )#alpha=0.5)
        
        axis.set_xlim(x_lims)
        axis.set_ylim(0, np.max(density)*1.01)
        axis.set_ylabel("PDF")

    # Otherwise plot normal histogram
    else:
        bin_edges = np.arange(min_rp, max_rp+bin_width, bin_width)

        nn, bins, _ = axis.hist(
            lc_results["rp_fit"], 
            bins=bin_edges,
            linewidth=0.4,
            edgecolor="black",
            zorder=0)

        if plot_poisson_errors:
            bin_mids = 0.5*(bins[1:] + bins[:-1])
            axis.errorbar(
                bins[1:]-bin_width/2, 
                nn, 
                yerr=np.sqrt(nn), 
                fmt='none', 
                zorder=1,
                linewidth=0.3,
                e_linewidth=0.00005,
                ecolor="red",
                barsabove=True,
                capsize=1,)

        axis.set_xlim(bins[0], x_lims[1])
        axis.set_ylabel("# Planets", fontsize="large")

    axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
    axis.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
    axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=1))
    axis.yaxis.set_major_locator(plticker.MultipleLocator(base=2))

    axis.set_xlabel(r"Planet radius ($R_{\oplus}$)", fontsize="large")

    axis.tick_params(axis='both', which='major', labelsize="large")

    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig("paper/planet_radii_hist.pdf")
    plt.savefig("paper/planet_radii_hist.png")


def plot_planet_period_vs_radius(lc_results):
    """Plot planet period against planet radii.

    Parameters
    ----------
    lc_results: pandas.core.frame.DataFrame
        Results DataFrame from light curve fitting
    """
    plt.close("all")
    fig, axis = plt.subplots()

    axis.errorbar(
        lc_results["Period (days)"],
        lc_results["rp_fit"],
        xerr=lc_results["Period error"],
        yerr=lc_results["e_rp_fit"],
        fmt=".")

    axis.set_xlabel("Orbital Period (days)")
    axis.set_ylabel(r"Planet radius ($R_E$)")

    axis.set_ylim(-0.1,15)

    plt.tight_layout()
    plt.savefig("paper/planet_period_vs_radius.pdf")
    plt.savefig("paper/planet_period_vs_radius.png")


def plot_confirmed_planet_comparison(
    toi_results,
    confirmed_planet_tab="data/known_planets.tsv",
    rp_rstar_lims=(0.005,0.2),
    a_rstar_lims=(1,90),
    i_lims=(80.5,91),
    rp_lims=(0.1,14),
    rp_rstar_ticks=(0.05,0.025,0.005,0.0025),
    a_rstar_ticks=(20,10,2,1),
    i_ticks=(1,0.5,0.75,0.375),
    rp_ticks=(2,1,0.5,0.25),
    show_offset=True,
    ms=10,
    print_labels=False,
    max_rp_rstar=0.2,
    max_rp=14,
    use_conservative_e_i=True):
    """
    """
    # Import literature data of confirmed planets
    cp_cat = pd.read_csv(confirmed_planet_tab, delimiter="\t", index_col="TOI")

    # Merge
    merged_cat = toi_results.join(cp_cat, on="TOI", how="inner", rsuffix="_lit")

    # If we've been given maximums for rp_rstar and rp, make cuts
    if max_rp_rstar is not None:
        merged_cat = merged_cat[merged_cat["rp_rstar_fit"] < max_rp_rstar]
    
    if max_rp is not None:
        merged_cat = merged_cat[merged_cat["rp_fit"] < max_rp]

    # Set conservative inclinations for those where it is high
    if use_conservative_e_i:
        high_e_i_mask = merged_cat["e_inclination_fit"] > 10
        merged_cat.loc[high_e_i_mask,"e_inclination_fit"] = 5

    plt.close("all")
    fig, (rp_rstar_ax, a_rstar_ax, i_ax, rp_ax) = plt.subplots(1,4)

    # Rp/R*
    plot_std_comp_generic(
        fig,
        rp_rstar_ax, 
        merged_cat["rp_rstar_fit"].values,
        merged_cat["e_rp_rstar_fit"].values,
        merged_cat["rp_rstar"].values,
        [merged_cat["e_rp_rstar_neg"].values, merged_cat["e_rp_rstar_pos"].values],
        None,
        r"$R_P/R_*$ (fit)",
        r"$R_P/R_*$ (literature)",
        "",
        x_lims=rp_rstar_lims,
        y_lims=rp_rstar_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=rp_rstar_ticks,
        plot_scatter=False,
        ms=ms,
        resid_y_lims=(-0.011,0.011),
        text_labels=merged_cat["name"].values,
        print_labels=print_labels,
        offset_sig_fig=4,)

    # a/R*
    plot_std_comp_generic(
        fig,
        a_rstar_ax, 
        merged_cat["sma_rstar_fit"].values,
        merged_cat["e_sma_rstar_fit"].values,
        merged_cat["a_rstar"].values,
        [merged_cat["e_a_rstar_neg"].values, merged_cat["e_a_rstar_pos"].values],
        None,
        r"$a/R_*$ (fit)",
        r"$a/R_*$ (literature)",
        "",
        x_lims=a_rstar_lims,
        y_lims=a_rstar_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=a_rstar_ticks,
        plot_scatter=False,
        ms=ms,
        resid_y_lims=(-5,5),
        text_labels=merged_cat["name"].values,
        print_labels=print_labels)

    # inclination
    plot_std_comp_generic(
        fig,
        i_ax, 
        merged_cat["inclination_fit"].values,
        merged_cat["e_inclination_fit"].values,
        merged_cat["i"].values,
        [merged_cat["e_i_neg"].values, merged_cat["e_i_pos"].values],
        None,
        r"$i$ (fit)",
        r"$i$ (literature)",
        "",
        x_lims=i_lims,
        y_lims=i_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=i_ticks,
        plot_scatter=False,
        ms=ms,
        resid_y_lims=(-1.9,1.9),
        text_labels=merged_cat["name"].values,
        print_labels=print_labels)

    # Rp
    plot_std_comp_generic(
        fig,
        rp_ax, 
        merged_cat["rp_fit"].values,
        merged_cat["e_rp_fit"].values,
        merged_cat["rp"].values,
        [merged_cat["e_rp_neg"].values, merged_cat["e_rp_pos"].values],
        None,
        r"$R_P$ (fit)",
        r"$R_P$ (literature)",
        "",
        x_lims=rp_lims,
        y_lims=rp_lims,
        cmap="viridis",
        show_offset=show_offset,
        ticks=rp_ticks,
        plot_scatter=False,
        ms=ms,
        resid_y_lims=(-1.1,1.1),
        text_labels=merged_cat["name"].values,
        print_labels=print_labels)

    # Wrap up
    plt.gcf().set_size_inches(16, 4)
    plt.tight_layout()
    plt.savefig("paper/lit_planet_comp.pdf")
    plt.savefig("paper/lit_planet_comp.png")


# -----------------------------------------------------------------------------
# Light curve fitting
# ----------------------------------------------------------------------------- 
def plot_lightcurve_fit(
    lightcurve,
    binned_lightcurve,
    folded_lc,
    folded_binned_lc,
    flat_lc_trend,
    bm_lightcurve,
    bm_binned_lightcurve,
    period,
    t0,
    toi,
    tic,
    toi_row,
    plot_path,
    bm_lc_time_unbinned=None,
    bm_lc_flux_unbinned=None,
    rasterized=False,
    plot_in_paper_figure_format=False,
    bin_size_mins=[2,10,]):
    """Plot PDF of TESS light curve fit.

    Parameters
    ----------
    lightcurve: lightkurve.lightcurve.TessLightCurve
        Undfolded TESS light curve.
    
    folded_lc: lightkurve.lightcurve.FoldedLightCurve
        TESS light curve folded about the planet transit epoch and period. Note
        that this means the new 'period' is 1, and the 'epoch' (i.e. time of
        transit is 0).

    flat_lc_trend: TODO

    bm_lightcurve: float array
        Batman transit model fluxes associated with time.

    period: float
        Period of transit. Should be set to one if using units of phase.

    t0: float
        Epoch (i.e time of transit). Should be set to zero if using units of 
        phase.

    trans_dur: float
        Transit duration in days.
    """
    plt.close("all")
    fig = plt.figure()

    # If we're plotting in the paper figure format, don't plot the middle panel
    if plot_in_paper_figure_format:
        gs = fig.add_gridspec(nrows=1, ncols=1)
        #ax_lc_unfolded = fig.add_subplot(gs[0, :])
        ax_lc_folded_transit = fig.add_subplot(gs[0, :])

    # Otherwise plot as a normal diagnostic plot with all three panels
    else:
        gs = fig.add_gridspec(nrows=3, ncols=1)
        ax_lc_unfolded = fig.add_subplot(gs[0, :])
        ax_lc_folded_all = fig.add_subplot(gs[1, :])
        ax_lc_folded_transit = fig.add_subplot(gs[2, :])

    # First plot unfolded/unflattened light curve
    if not plot_in_paper_figure_format:
        lightcurve.errorbar(ax=ax_lc_unfolded, fmt=".", elinewidth=0.1, zorder=1,
            rasterized=rasterized, alpha=0.8, label="unflattened light curve")

        binned_lightcurve.errorbar(ax=ax_lc_unfolded, fmt=".", elinewidth=0.1, 
            zorder=1, rasterized=rasterized, 
            label="unflattened light curve (binned)")

        # Plot the trend flattening trend
        flat_lc_trend.scatter(ax=ax_lc_unfolded, linewidth=0.2, color="green", 
            zorder=2, rasterized=rasterized, label="fitted trend")

        # Plot lines where transits occur (from beginning to end of our window)
        transits = np.arange(t0-period*10000, lightcurve.time[-1], period)
        observed_transits_mask = np.logical_and(
            transits > lightcurve.time[0],
            transits < lightcurve.time[-1]
        )
        transits = transits[observed_transits_mask]

        for transit_i, transit in enumerate(transits):
            # Label only one transit
            if transit_i == 0:
                ax_lc_unfolded.vlines(transit, 0.90, 1.10, colors="red", 
                    linestyles="dashed", linewidth=0.25, alpha=1.0, zorder=3,
                    label="transits")
            else:
                ax_lc_unfolded.vlines(transit, 0.90, 1.10, colors="red", 
                    linestyles="dashed", linewidth=0.25, alpha=1.0, zorder=3,)

        ax_lc_unfolded.set_xlim((lightcurve.time[0], lightcurve.time[-1]))
    
        # Setup legend
        leg_lc_unfolded = ax_lc_unfolded.legend(loc="best")

        # Update width of legend objects
        for legobj in leg_lc_unfolded.legendHandles:
            legobj.set_linewidth(1.5)
    
    # If we've been given an unbinned batman model lightcurve, plot instead
    if bm_lc_time_unbinned is not None and bm_lc_flux_unbinned is not None:
        bm_lc_times = bm_lc_time_unbinned
        bm_lc_flux = bm_lc_flux_unbinned
    else:
        bm_lc_times = folded_lc.time
        bm_lc_flux = bm_lightcurve

    # Plot entire folded lightcurve
    if not plot_in_paper_figure_format:
        folded_lc.errorbar(ax=ax_lc_folded_all, fmt=".", elinewidth=0.1, 
            zorder=1, rasterized=rasterized, alpha=0.8, 
            label="folded light curve ({} min binning)".format(bin_size_mins[0]))

        folded_binned_lc.errorbar(ax=ax_lc_folded_all, fmt=".", elinewidth=0.1, 
            zorder=1, rasterized=rasterized,
            label="folded light curve ({} min binning)".format(bin_size_mins[1]))

        ax_lc_folded_all.plot(bm_lc_times, bm_lc_flux, zorder=2, c="red",
            label="transit model")

        ax_lc_folded_all.set_xlim((-0.5, 0.5))

        # Setup legend
        leg_lc_folded = ax_lc_folded_all.legend(loc="lower left")

        # Update width of legend objects
        for legobj in leg_lc_folded.legendHandles:
            legobj.set_linewidth(1.5)

    # Now plot just the transit
    folded_lc.errorbar(ax=ax_lc_folded_transit, fmt=".", elinewidth=0.2, 
        zorder=1, rasterized=rasterized, alpha=0.8, 
        label="folded light curve ({} min binning)".format(bin_size_mins[0]))

    folded_binned_lc.errorbar(ax=ax_lc_folded_transit, fmt=".", elinewidth=0.2, 
        zorder=1, rasterized=rasterized, marker="o", markersize=2,
        label="folded light curve ({} min binning)".format(bin_size_mins[1]))

    ax_lc_folded_transit.plot(bm_lc_times, bm_lc_flux, zorder=2, c="red",
        label="transit model")

    # Plot lines at zero phase
    if not plot_in_paper_figure_format:
        ax_lc_folded_all.vlines(0, 0.90, 1.10, colors="black",
            linestyles="dashed", linewidth=0.5, alpha=0.5, zorder=3,)
    ax_lc_folded_transit.vlines(0, 0.90, 1.10, colors="black",
        linestyles="dashed", linewidth=0.5, alpha=0.5, zorder=3,)

    trans_dur = toi_row["Duration (hours)"]/24

    ax_lc_folded_transit.set_xlim((-2*trans_dur/period, 2*trans_dur/period))

    # Setup legend
    leg_lc_folded_transit = ax_lc_folded_transit.legend(
        loc="lower left", fontsize="large")

    # Update width of legend objects
    for legobj in leg_lc_folded_transit.legendHandles:
        legobj.set_linewidth(1.5)

    # Figure out our y limits
    transit_mask = np.logical_and(
        folded_lc.time < trans_dur/period/2,
        folded_lc.time > -trans_dur/period/2)
        
    y_min = 1.5*(1-np.nanmean(folded_lc.flux[transit_mask]))

    std_lim = np.nanstd(folded_lc.flux)*3.0 + np.nanmean(folded_lc.flux_err)

    # Use whichever is lower
    y_lim = y_min if y_min > std_lim else std_lim

    if not plot_in_paper_figure_format:
        ax_lc_unfolded.set_ylim((1-y_lim, 1+std_lim))
        ax_lc_folded_all.set_ylim((1-y_lim, 1+std_lim))

    ax_lc_folded_transit.set_ylim((1-y_lim, 1+std_lim))

    # Finally plot the residuals
    divider = make_axes_locatable(ax_lc_folded_transit)
    res_ax = divider.append_axes("bottom", size="30%", pad=0)
    ax_lc_folded_transit.figure.add_axes(res_ax, sharex=ax_lc_folded_transit)

    resid = folded_lc.flux - bm_lightcurve
    resid_binned = folded_binned_lc.flux - bm_binned_lightcurve

    res_ax.errorbar(
        folded_lc.time,
        resid,
        yerr=folded_lc.flux_err,
        fmt=".",
        markersize=1,
        elinewidth=0.1,
        zorder=1,
        alpha=0.8,
        rasterized=rasterized,)

    res_ax.errorbar(
        folded_binned_lc.time,
        resid_binned,
        yerr=folded_binned_lc.flux_err,
        fmt=".",
        markersize=1,
        elinewidth=0.1,
        zorder=1,
        rasterized=rasterized,)

    res_ax.hlines(
        0,
        -2*trans_dur/period,
        2*trans_dur/period,
        colors="black",
        linestyles="dashed",
        zorder=0,)
    ax_lc_folded_transit.set_ylabel("Normalised\nFlux", fontsize="large")
    res_ax.set_ylabel("residuals", fontsize="large")
    res_ax.set_xlabel("Phase", fontsize="large")

    plt.setp(ax_lc_folded_transit.get_xticklabels(), visible=False)
    res_ax.set_xlim((-2*trans_dur/period, 2*trans_dur/period))
    resid_std = np.nanstd(resid)
    res_ax.set_ylim(-4*resid_std, 4*resid_std)

    # Tick size
    ax_lc_folded_transit.tick_params(
        axis='both', which='major', labelsize="large")
    res_ax.tick_params(axis='both', which='major', labelsize="large")

    # Do remaining plot setup depending on whether diagnostic or paper
    if not plot_in_paper_figure_format:
        # Only set title for diagnostic plots
        title = (r"TIC {}, TOI {}    $[T_{{\rm eff}}={:0.0f}\,$K, $R_*={:0.2f}\,"
                r"R_\odot$, T = {:0.6f} days, $\frac{{R_p}}{{R_*}} = {:0.5f}$, "
                r"$R_P$ = {:0.2f} $R_E$, "
                r"$\frac{{a}}{{R_*}} = {:0.2f}$, $i = {:0.2f}]$")

        title = title.format(
            tic,
            toi, 
            toi_row["teff_synth"], 
            toi_row["radius"],
            period,
            toi_row["rp_rstar_fit"],
            toi_row["rp_fit"],
            toi_row["sma_rstar_fit"],
            toi_row["inclination_fit"],)
        plt.suptitle(title)

        # Set save path for diagnostic plots
        suffix = "diagnostic"

        # Set size
        plt.gcf().set_size_inches(16, 9)

    else:
        # Set sufix
        suffix = "paper"

        # Set size
        plt.gcf().set_size_inches(16, 3)

    # Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_save_loc = os.path.join(
        plot_path, 
        "lc_fit_{}_{}_{}.pdf".format(toi, tic, suffix))
    plt.savefig(plt_save_loc)


def plot_all_lightcurve_fits(
    light_curves,
    toi_info,
    tess_info,
    observations,
    binned_light_curves,
    break_tol_days=0.5,
    flat_lc_trends=None,
    bm_lc_times_unbinned=None,
    bm_lc_fluxes_unbinned=None,
    t_min=12/24,
    force_window_length_to_min=False,
    rasterized=False,
    make_paper_plots=False,):
    """Plot PDFs of all TESS light curve fits.

    Parameters
    ----------
    light_curves: dict of lightkurve.lightcurve.TessLightCurve
        Dictionary matching lightcurves to TIC IDs.
    
    toi_info: pandas.core.frame.DataFrame
        Dataframe containing information about each TOI from NASA ExoFOP.

    tess_info: pandas.core.frame.DataFrame
        Dataframe containing literature info on each TIC (e.g. photometry).

    observations: pandas.core.frame.DataFrame
        Dataframe containing info on spectroscopic observations of each TIC, 
        plus results of synthetic fitting.

    binsize: int, default: 4
        Binning to use when plotting light curves.
    """
    # Make plot path if it doesn't already exist
    plot_path = os.path.join("plots", "lc_diagnostics")

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    
    # If it does, clear out pdfs
    else:
        pdfs = glob.glob(os.path.join(plot_path, "*.pdf"))
        for pdf in pdfs:
            os.remove(pdf)

    # Get temporary single jumbo dataframe
    info = toi_info.join(tess_info, on="TIC", how="inner", lsuffix="", rsuffix="_2")
    #info.reset_index(inplace=True)
    comb_info = info.join(observations, on="source_id", lsuffix="", rsuffix="_2", how="inner")
    
    for toi_i, (toi, toi_row) in zip(
        tqdm(range(len(comb_info)),desc="Plotting"), comb_info.iterrows()):
        # Look up TIC ID in tess_info
        tic = toi_row["TIC"]

        source_id = toi_row["source_id"]

        param_cols = ["rp_rstar_fit", "sma_rstar_fit", "inclination_fit"]

        # Skip if period is nan, lightcurve is None, or all params are nan
        if (np.isnan(toi_row["Period (days)"]) or light_curves[tic] is None
            or np.all(np.isnan(toi_row[param_cols].astype(float)))):
            continue

        # Use fitted period if we calculated it
        if not np.isnan(toi_row["period_fit"]):
            period = toi_row["period_fit"]
        else:
            period = toi_row["Period (days)"]

        # Use fitted t0 if we calculated it
        if not np.isnan(toi_row["t0_fit"]):
            t0 = toi_row["t0_fit"]
        else:
            t0 = toi_row["Transit Epoch (BJD)"]

        # Convert between BJD and TESS BJD
        if t0 > transit.BTJD_OFFSET:
            t0 -= transit.BTJD_OFFSET

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Setup lightcurve
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        light_curve = light_curves[tic]
        binned_light_curve = binned_light_curves[tic]

        # Clean and flatten both light curves
        clean_lc, flat_lc_trend = transit.flatten_and_clean_lc(
            light_curve, toi_info, tic, toi, break_tol_days, t0, period, 
            t_min,force_window_length_to_min,)

        clean_binned_lc, _ = transit.flatten_and_clean_lc(
            binned_light_curve, toi_info, tic, toi, break_tol_days, t0,
            period, t_min, force_window_length_to_min,)

        # Use the flattened trend we've been given, otherwise the new one
        if flat_lc_trends[toi] is not None:
            flat_lc_trend = flat_lc_trends[toi]

        # Phase fold the light curves
        folded_lc = clean_lc.fold(period=period, t0=t0)
        folded_binned_lc = clean_binned_lc.fold(period=period, t0=t0)

        # Generate batman model at our unbinned resolution
        bm_params, bm_model, bm_lightcurve = transit.initialise_bm_model(
            t0=0, 
            period=1, 
            rp_rstar=toi_row["rp_rstar_fit"], 
            sma_rstar=toi_row["sma_rstar_fit"], 
            inclination=toi_row["inclination_fit"], 
            ecc=0, 
            omega=0, 
            ld_model="nonlinear", 
            ld_coeff=toi_row[["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]].values, 
            time=folded_lc.time,)

        # And another at the binned resolution
        _, _, bm_binned_lightcurve = transit.initialise_bm_model(
            t0=0, 
            period=1, 
            rp_rstar=toi_row["rp_rstar_fit"], 
            sma_rstar=toi_row["sma_rstar_fit"], 
            inclination=toi_row["inclination_fit"], 
            ecc=0, 
            omega=0, 
            ld_model="nonlinear", 
            ld_coeff=toi_row[["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]].values, 
            time=folded_binned_lc.time,)

        # Make diagnostic plots
        plot_lightcurve_fit(
            lightcurve=light_curve,
            binned_lightcurve=binned_light_curve,
            folded_lc=folded_lc,
            folded_binned_lc=folded_binned_lc, 
            flat_lc_trend=flat_lc_trend,
            bm_lightcurve=bm_lightcurve,
            bm_binned_lightcurve=bm_binned_lightcurve,
            period=period, 
            t0=t0, 
            toi=toi,
            tic=tic,
            toi_row=toi_row,
            plot_path=plot_path,
            #trans_dur=toi_row["Duration (hours)"]/24,
            bm_lc_time_unbinned=bm_lc_times_unbinned[toi],
            bm_lc_flux_unbinned=bm_lc_fluxes_unbinned[toi],
            rasterized=rasterized,)

        # And make paper ready plots
        if make_paper_plots:
            plot_lightcurve_fit(
                lightcurve=light_curve,
                binned_lightcurve=binned_light_curve,
                folded_lc=folded_lc,
                folded_binned_lc=folded_binned_lc, 
                flat_lc_trend=flat_lc_trend,
                bm_lightcurve=bm_lightcurve,
                bm_binned_lightcurve=bm_binned_lightcurve,
                period=period, 
                t0=t0, 
                toi=toi,
                tic=tic,
                toi_row=toi_row,
                plot_path=plot_path,
                #trans_dur=toi_row["Duration (hours)"]/24,
                bm_lc_time_unbinned=bm_lc_times_unbinned[toi],
                bm_lc_flux_unbinned=bm_lc_fluxes_unbinned[toi],
                rasterized=rasterized,
                plot_in_paper_figure_format=True)


def plot_lightcurve_phasefolded_x2(
    light_curve,
    light_curve_binned,
    tic,
    toi,
    delta_period,
    toi_info,
    t_min=12/24,
    break_tol_days=12/24,
    force_window_length_to_min=True,):
    """Plot the phase folded light curve at two different binning levels. 
    Useful for quickly iterating a better fit initial period.
    """
    # Grab epoch, and get new period guess
    t0 = toi_info.loc[toi]["Transit Epoch (BJD)"] - transit.BTJD_OFFSET
    period = toi_info.loc[toi]["Period (days)"] + delta_period
    trans_dur = toi_info.loc[toi]["Duration (hours)"] / 24

    # Clean and flatten both light curves
    clean_lc, flat_lc_trend = transit.flatten_and_clean_lc(
        light_curve, toi_info, tic, toi, break_tol_days, t0, period, 
        t_min,force_window_length_to_min,)

    clean_binned_lc, _ = transit.flatten_and_clean_lc(
        light_curve_binned, toi_info, tic, toi, break_tol_days, t0,
        period, t_min, force_window_length_to_min,)

    # Phase fold the light curves
    folded_lc = clean_lc.fold(period=period, t0=t0)
    folded_binned_lc = clean_binned_lc.fold(period=period, t0=t0)

    # Plot
    plt.close("all")
    fig, ax = plt.subplots()
    folded_lc.errorbar(ax=ax, fmt=".", elinewidth=0.1, 
        zorder=1, rasterized=True, alpha=0.8, markersize=2,
        label="flattened light curve")

    folded_binned_lc.errorbar(ax=ax, fmt=".", elinewidth=0.1, 
        zorder=1, rasterized=True, markersize=2,
        label="flattened light curve (binned)")

    #ax.plot(bm_lc_times, bm_lc_flux, zorder=2, c="red",
    #    label="transit model")

    # Plot line at zero phase
    ax.vlines(0, 0.90, 1.10, colors="black",
        linestyles="dashed", linewidth=0.5, alpha=0.5, zorder=3,)

    ax.set_xlim((-0.4, 0.4))

    # Setup legend
    leg_lc_folded = ax.legend(loc="lower left")

    # Update width of legend objects
    for legobj in leg_lc_folded.legendHandles:
        legobj.set_linewidth(1.5)
    
    # Figure out our y limits
    transit_mask = np.logical_and(
        folded_lc.time < trans_dur/period/2,
        folded_lc.time > -trans_dur/period/2)

    y_min = 1.5*(1-np.nanmean(folded_lc.flux[transit_mask]))

    std_lim = np.nanstd(folded_lc.flux)*3.0 + np.nanmean(folded_lc.flux_err)

    # Use whichever is lower
    y_lim = y_min if y_min > std_lim else std_lim

    ax.set_ylim((1-y_lim, 1+std_lim))

    # Find median at +/-0.01 phase
    transit_mask = np.abs(folded_lc.time) < 0.001
    med = np.median(folded_lc.flux[transit_mask])
    ax.text(0, 1+0.9*(std_lim), "{:0.5f}".format(med), horizontalalignment="center")

    plt.gcf().set_size_inches(20, 6)

    if np.isnan(toi_info.loc[toi]["Period error"]):
        e_period = 0
    else:
        e_period = toi_info.loc[toi]["Period error"]

    # Plot title
    title = (r"$T_O = {:.6f}$ days $(\pm{:.2f} sec)$"
             r" --> $T_N = {:.6f}$ days $({:+.2f}$ sec)")
    plt.title(
        title.format(
            toi_info.loc[toi]["Period (days)"],
            e_period *24*3600,
            period,
            delta_period * 24 * 3600,
        )
    )

    plt.suptitle("TIC {}, TOI {}".format(tic, toi))

# -----------------------------------------------------------------------------
# Radial Velocities
# ----------------------------------------------------------------------------- 
def get_gaia_rv(uid, std_params):
    """
    """
    entry = std_params[std_params["source_id"]==uid]
    if len(entry) == 0:
        rv = np.nan
        e_rv = np.nan
    elif len(entry) == 1:
        rv = float(entry["rv"])
        e_rv = float(entry["e_rv"])
    elif len(entry) > 1:
        raise ValueError("Duplicate standard detected in std_params")

    return rv, e_rv


def plot_rv_comparison():
    """Join catalogues, and plot a comparison of observed RVs with those from
    Gaia.
    """
    # Import catalogues
    tess_info = utils.load_info_cat(
        "data/tess_info.tsv",
        in_paper=True,
        only_observed=True,
        use_mann_code_for_masses=False,
        do_extinction_correction=False,)
    std_info = utils.load_info_cat(
        "data/std_info.tsv",
        in_paper=True,
        only_observed=True,
        use_mann_code_for_masses=False,
        do_extinction_correction=False,)

    # Import observed/fitted parameters tables
    _, _, obs_tess = utils.load_fits("tess", path="spectra")
    _, _, obs_std = utils.load_fits("std", path="spectra")

    # Combine
    comb_tess = tess_info.join(
        obs_tess,
        on="source_id",
        lsuffix="_info",
        rsuffix="",
        how="inner")
    comb_std = std_info.join(
        obs_std,
        on="source_id",
        lsuffix="_info",
        rsuffix="",
        how="inner")
    catalogue = pd.concat([comb_std, comb_tess], axis=0, sort=False)

    is_tess_mask = ~np.isnan(catalogue["TIC"])
    
    plt.close("all")
    fig, axis = plt.subplots(sharex=True)

    # Setup lower panel for residuals
    #plt.setp(axis.get_xticklabels(), visible=False)
    divider = make_axes_locatable(axis)
    res_ax = divider.append_axes("bottom", size="30%", pad=0)
    axis.figure.add_axes(res_ax, sharex=axis)
    
    xx = np.arange(catalogue["rv_info"].min(),catalogue["rv_info"].max()) 
    axis.plot(xx, xx, "k--") 
    plt.setp(axis.get_xticklabels(), visible=False)
    
    plt.hlines(
        0, 
        catalogue["rv_info"].min(),
        catalogue["rv_info"].max(), 
        linestyles="--")

    # Plot TESS
    axis.errorbar(
        x=catalogue[is_tess_mask]["rv_info"], 
        y=catalogue[is_tess_mask]["rv"], 
        xerr=catalogue[is_tess_mask]["e_rv_info"], 
        yerr=catalogue[is_tess_mask]["e_rv"], 
        fmt=".", 
        zorder=0,
        color="#1f77b4",
        ecolor="black",
        label="TESS",
        alpha=0.9)
    
    res_ax.errorbar(
        x=catalogue[is_tess_mask]["rv_info"], 
        y=catalogue[is_tess_mask]["rv_info"]-catalogue[is_tess_mask]["rv"], 
        xerr=catalogue[is_tess_mask]["e_rv_info"], 
        yerr=np.sqrt(
            catalogue[is_tess_mask]["e_rv"]**2 
            + catalogue[is_tess_mask]["e_rv_info"]**2), 
        fmt=".", 
        zorder=0,
        color="#1f77b4",
        ecolor="black",
        alpha=0.9)

    # Plot Standard
    axis.errorbar(
        x=catalogue[~is_tess_mask]["rv_info"], 
        y=catalogue[~is_tess_mask]["rv"], 
        xerr=catalogue[~is_tess_mask]["e_rv_info"], 
        yerr=catalogue[~is_tess_mask]["e_rv"], 
        fmt="*", 
        zorder=0,
        color="#ff7f0e",
        ecolor="black",
        label="Standard",
        alpha=0.9)
    
    res_ax.errorbar(
        x=catalogue[~is_tess_mask]["rv_info"], 
        y=catalogue[~is_tess_mask]["rv_info"]-catalogue[~is_tess_mask]["rv"], 
        xerr=catalogue[~is_tess_mask]["e_rv_info"], 
        yerr=np.sqrt(
            catalogue[~is_tess_mask]["e_rv"]**2 
            + catalogue[~is_tess_mask]["e_rv_info"]**2), 
        fmt="*", 
        zorder=0,
        color="#ff7f0e",
        ecolor="black",
        alpha=0.9,)

    axis.legend(fontsize="large")
    #axis.set_aspect(1./axis.get_data_ratio())
    res_ax.set_xlabel(r"RV, Gaia DR2 (km$\,$s$^{-1}$)", fontsize="large")
    res_ax.set_ylabel(r"Residuals", fontsize="large")
    axis.set_ylabel(r"RV, WiFeS (km$\,$s$^{-1}$)", fontsize="large")

    axis.tick_params(axis='both', which='major', labelsize="large")
    res_ax.tick_params(axis='both', which='major', labelsize="large")

    plt.gcf().set_size_inches(6, 6)
    fig.tight_layout()
    plt.savefig("paper/rv_comp.pdf")
    plt.savefig("paper/rv_comp.png")

    # Also plot RV systematic
    plt.figure()
    rv_systs = np.arange(0,10,0.1) 
    vals = [] 
    for rv_syst in rv_systs: 
        vals.append(1.4826 * np.nanmedian(
            np.abs(catalogue["rv_info"]-catalogue["rv"])
            /np.sqrt(
                catalogue["e_rv_info"]**2
                + catalogue["e_rv"]**2 + rv_syst**2)))

    plt.plot(rv_systs*1000, vals)
    plt.hlines(1,0,10000)
    plt.xlabel("RV (m/s)") 