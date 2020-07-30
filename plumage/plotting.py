"""Various plotting functions"
"""
from __future__ import print_function, division
import os
import numpy as np
import glob
import batman as bm
import matplotlib.pylab as plt
import matplotlib.colors as mplc
import plumage.spectra as spec
import plumage.synthetic as synth
import plumage.transits as transit
from tqdm import tqdm
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    axis.set_ylim(0.0, 1.7)
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


def plot_cmd(colours, abs_mags, ruwe_mask, target_colour, target_abs_mag, 
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
    sid = obs_info["uid"]
    is_obs = info_cat["observed"].values
    star_info = info_cat[is_obs][info_cat[is_obs]["source_id"]==sid].iloc[0]

    # Get the ID and info about the target
    if is_tess:
        toi = star_info["TOI"]
        tic = star_info.name

        plt.suptitle("TOI {}, TIC {}, Gaia DR2 {}".format(
            toi, tic, sid))
    
    else:
        if not use_2mass_id:
            
            id_prefix = "Gaia DR2 "
            id_col = "ID"
            
        else:
            id_prefix = "2MASS J"
            id_col = "2mass"
            
        if len(star_info) < 1:
            return
        elif len(star_info) > 1:
            return
        else:
            star_info = star_info.iloc[0]

        plt.suptitle("{}, Gaia DR2 {}".format(star_info[id_col], sid))

    # Plot Gaia CMD
    plot_cmd(
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
    plot_cmd(
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

    # If lit params are known, display
    if "teff" in star_info and not is_tess:
        lit_params = star_info[
            ["teff", "e_teff", "logg", "feh", "e_feh"]].values
        lit_param_label = (r"Lit Params ({}, {}): $T_{{\rm eff}} = {:0.0f} "
                           r"\pm {:0.0f}\,$K, $\log g = {:0.2f}$, [Fe/H]$ = "
                           r"{:0.2f} \pm {:0.2f}$")
        lit_param_label = lit_param_label.format(
            str(star_info["kind"]),
            str(star_info["source"]), 
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

# -----------------------------------------------------------------------------
# Comparisons & other paper plots
# ----------------------------------------------------------------------------- 
def plot_std_comp_generic(fig, axis, fit, e_fit, lit, e_lit, colour, x_label, 
    y_label, cb_label, lims, cmap, show_median_offset,):
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

    show_median_offset: bool, default: False
        Whether to plot the median uncertainty as text.
    """
    # Plot error bars with overplotted scatter points + colour bar
    axis.errorbar(fit, lit, xerr=e_fit, yerr=e_lit, fmt=".", zorder=0,)

    sc = axis.scatter(fit, lit, c=colour, zorder=1, cmap=cmap)

    cb = fig.colorbar(sc, ax=axis)
    cb.set_label(cb_label)

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    # Plot 1:1 line
    xx = np.arange(lims[0], lims[1], (lims[1]-lims[0])/100)
    axis.plot(xx, xx, "--")

    axis.set_xlim(lims)
    axis.set_ylim(lims)

    axis.set_aspect("equal")

    if show_median_offset:
        med_offset = np.nanmedian(np.abs(fit - lit))
        axis.text(
            x=np.mean(lims), 
            y=0.9*lims[1], 
            s=r"$\pm {:0.2f}$".format(med_offset),
            horizontalalignment="center")


def plot_std_comp(
    observations, 
    std_info,
    teff_lims=[3000,4600],
    feh_lims=[-1.4,0.75],
    show_median_offset=False,
    fn_suffix="",
    title_text="",):
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

    show_median_offset: bool, default: False
        Whether to plot the median offset as text.

    fn_suffix: string, default: ''
        Suffix to append to saved figures
        
    title_text: string, default: ''
        Text for fig.suptitle.
    """
    # Table join
    #observations.rename(columns={"uid":"source_id"}, inplace=True)
    obs_join = observations.join(std_info, "source_id", rsuffix="_info")

    plt.close("all")
    fig, axes = plt.subplots(2,3)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, 
                            wspace=0.5)

    # Unpack to improve readability
    ((ax_teff_m15, ax_teff_ra12, ax_teff_int),
     (ax_feh_m15, ax_feh_ra12, ax_feh_cpm)) = axes

    # Mann+15 temperatures
    plot_std_comp_generic(
        fig,
        ax_teff_m15, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_m15"],
        obs_join["e_teff_m15"],
        obs_join["feh_synth"],
        r"$T_{\rm eff}\,$K (fit)",
        r"$T_{\rm eff}\,$K (Mann+15)",
        "[Fe/H] (fit)",
        lims=teff_lims,
        cmap="viridis",
        show_median_offset=show_median_offset,)
    
    # Rojas-Ayala+12 temperatures
    plot_std_comp_generic(
        fig,
        ax_teff_ra12, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_ra12"],
        obs_join["e_teff_ra12"],
        obs_join["feh_synth"],
        r"$T_{\rm eff}\,$K (fit)",
        r"$T_{\rm eff}\,$K (Rojas-Ayala+12)",
        "[Fe/H] (fit)",
        lims=teff_lims,
        cmap="viridis",
        show_median_offset=show_median_offset,)
    
    # Interferometric temperatures
    plot_std_comp_generic(
        fig,
        ax_teff_int, 
        obs_join["teff_synth"],
        obs_join["e_teff_synth"],
        obs_join["teff_int"],
        obs_join["e_teff_int"],
        obs_join["feh_synth"],
        r"$T_{\rm eff}\,$K (fit)",
        r"$T_{\rm eff}\,$K (interferometric)",
        "[Fe/H] (fit)",
        lims=teff_lims,
        cmap="viridis",
        show_median_offset=show_median_offset,)

    # Mann+15 [Fe/H]
    plot_std_comp_generic(
        fig,
        ax_feh_m15, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_m15"],
        obs_join["e_feh_m15"],
        obs_join["teff_synth"],
        r"[Fe/H] (fit)",
        r"[Fe/H] (Mann+15)",
        r"$T_{\rm eff}\,$K (fit)",
        lims=feh_lims,
        cmap="magma",
        show_median_offset=show_median_offset,)
    
    # Rojas-Ayala+12 [Fe/H]
    plot_std_comp_generic(
        fig,
        ax_feh_ra12, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_ra12"],
        obs_join["e_feh_ra12"],
        obs_join["teff_synth"],
        r"[Fe/H] (fit)",
        r"[Fe/H] (Rojas-Ayala+12)",
        r"$T_{\rm eff}\,$K (fit)",
        lims=feh_lims,
        cmap="magma",
        show_median_offset=show_median_offset,)

    # CPM [Fe/H]
    plot_std_comp_generic(
        fig,
        ax_feh_cpm, 
        obs_join["feh_synth"],
        obs_join["e_feh_synth"],
        obs_join["feh_cpm"],
        obs_join["e_feh_cpm"],
        obs_join["teff_synth"],
        r"[Fe/H] (fit)",
        r"[Fe/H] (CPM)",
        r"$T_{\rm eff}\,$K (fit)",
        lims=feh_lims,
        cmap="magma",
        show_median_offset=show_median_offset,)

    fig.suptitle(title_text)
    plt.gcf().set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig("paper/std_comp{}.pdf".format(fn_suffix))
    plt.savefig("paper/std_comp{}.png".format(fn_suffix))


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


def plot_cmd(
    info_cat, 
    info_cat_2=None,
    plot_toi_ids=False,
    colour="Bp-Rp",
    abs_mag="G_mag_abs",
    x_label=r"$B_P-R_P$",
    y_label=r"$M_{\rm G}$",
    label="tess",):
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

    # Plot our first set of stars
    scatter = axis.scatter(
        info_cat[colour], 
        info_cat[abs_mag], 
        zorder=1,
    )

    # Plot a second set of stars behind (e.g. standards)
    if info_cat_2 is not None:
        scatter = axis.scatter(
        info_cat_2[colour], 
        info_cat_2[abs_mag], 
        marker="*",
        zorder=0,
    )

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

    #cb = fig.colorbar(scatter, ax=axis)
    #cb.set_label("RUWE > 1.4")

    # Flip magnitude axis
    ymin, ymax = axis.get_ylim()
    axis.set_ylim((ymax, ymin))

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

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


def plot_radius_comp(observations, info_cat):
    """Plot a comparison between our radii determined here vs their Mann+15 
    equivalents. Has colourbar for [Fe/H].

    Parameters
    ----------
    observations: pandas.DataFrame
        Table of observations and fit results.

    info_cat: pandas.DataFrame
        Corresponding table of stellar literature info.
    """
    # Table join
    obs_join = observations.join(info_cat, "source_id", rsuffix="_info")

    # Plot errors, then coloured [Fe/H] scatter points on top of them
    plt.close("all")
    plt.errorbar(
        obs_join["radius"],
        obs_join["radii_m19"],
        xerr=obs_join["e_radius"],
        yerr=obs_join["e_radii_m19"],
        fmt=".",
        zorder=0,)

    sc = plt.scatter(
        obs_join["radius"],
        obs_join["radii_m19"],
        c=obs_join["feh_synth"],
        zorder=1,
    )

    cb = plt.colorbar(sc)
    cb.set_label("[Fe/H]")

    # Plot 1:1 line
    plt.plot(np.arange(0.1,0.9,0.1),np.arange(0.1,0.9,0.1),"--",color="black")

    plt.xlabel(r"Radius fit ($R_\odot$)")
    plt.ylabel(r"Radius Mann+15 ($R_\odot$)")

    plt.savefig("paper/radius_comp.pdf")
    plt.savefig("paper/radius_comp.png")


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
# Light curve fitting
# ----------------------------------------------------------------------------- 
def plot_lightcurve_fit(lightcurve, folded_lc, bm_params, bm_model, 
    bm_lightcurve, period, t0, trans_dur):
    """Plot PDF of TESS light curve fit.

    Parameters
    ----------
    lightcurve: lightkurve.lightcurve.TessLightCurve
        Undfolded TESS light curve.
    
    folded_lc: lightkurve.lightcurve.FoldedLightCurve
        TESS light curve folded about the planet transit epoch and period. Note
        that this means the new 'period' is 1, and the 'epoch' (i.e. time of
        transit is 0).

    bm_params: batman.transitmodel.TransitParams
        Batman transit parameter + time object.

    bm_model: batman.transitmodel.TransitModel
        Batman transit model.

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
    BTJD_OFFSET = 2457000

    plt.close("all")
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1)
    ax_lc_unfolded = fig.add_subplot(gs[0, :])
    ax_lc_folded_all = fig.add_subplot(gs[1, :])
    ax_lc_folded_transit = fig.add_subplot(gs[2, :])

    # First plot unfolded light curve
    lightcurve.errorbar(ax=ax_lc_unfolded, fmt=".", elinewidth=0.1)

    # Plot lines where the transits occur
    transits = np.arange(t0-BTJD_OFFSET, lightcurve.time[-1], period)

    for transit in transits:
        ax_lc_unfolded.vlines(transit, 0.95, 1.05, colors="red", 
            linestyles="dashed", linewidth=0.2, alpha=1.0)

    ax_lc_unfolded.set_xlim((lightcurve.time[0], lightcurve.time[-1]))

    # Plot entire folded lightcurve
    folded_lc.errorbar(ax=ax_lc_folded_all, fmt=".", elinewidth=0.1)
    ax_lc_folded_all.plot(folded_lc.time, bm_lightcurve)
    ax_lc_folded_all.set_xlim((-0.5, 0.5))

    # Now plot just the transit
    folded_lc.errorbar(ax=ax_lc_folded_transit, fmt=".", elinewidth=0.2)
    ax_lc_folded_transit.plot(folded_lc.time, bm_lightcurve)

    ax_lc_folded_transit.set_xlim((-2*trans_dur/period, 2*trans_dur/period))

    # Figure out our y limits
    lim = np.nanstd(folded_lc.flux)*5.5 + np.nanmean(folded_lc.flux_err)

    ax_lc_unfolded.set_ylim((1-lim, 1+lim))
    ax_lc_folded_all.set_ylim((1-lim, 1+lim))
    ax_lc_folded_transit.set_ylim((1-lim, 1+lim))


def plot_all_lightcurve_fits(lightcurves, toi_info, tess_info, observations,
        binsize=4):
    """Plot PDFs of all TESS light curve fits.

    Parameters
    ----------
    lightcurves: dict of lightkurve.lightcurve.TessLightCurve
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

    BTJD_OFFSET = 2457000

    info = toi_info.join(tess_info, on="TIC", how="inner", lsuffix="", rsuffix="_2")
    #info.reset_index(inplace=True)
    comb_info = info.join(observations, on="source_id", lsuffix="", rsuffix="_2", how="inner")

    for toi_i, (toi, toi_row) in zip(
        tqdm(range(len(comb_info)),desc="Plotting"), comb_info.iterrows()):
        # Look up TIC ID in tess_info
        tic = toi_row["TIC"]

        # Get the literature info
        #tic_info = tess_info[tess_info.index==tic].iloc[0]

        source_id = toi_row["source_id"]

        #obs_info = observations[observations.index==source_id]

        #if len(obs_info) > 0:
        #    obs_info = observations[observations.index==source_id].iloc[0]
        #else:
        #    continue

        param_cols = ["rp_rstar_fit", "sma_rstar_fit", "inclination_fit"]

        # Skip if period is nan, lightcurve is None, or all params are nan
        if (np.isnan(toi_row["Period (days)"]) or lightcurves[tic] is None
            or np.all(np.isnan(toi_row[param_cols].astype(float)))):
            continue

        # Load in, clean, bin, and fold lightcurve
        lightcurve = lightcurves[tic].remove_outliers(sigma=6).remove_nans()
        lightcurve = lightcurve.bin(binsize=binsize)
        folded_lc = lightcurve.fold(
            period=toi_row["Period (days)"], 
            t0=toi_row["Epoch (BJD)"]-BTJD_OFFSET)

        # Generate batman model
        bm_params, bm_model, bm_lightcurve = transit.initialise_bm_model(
            t0=0, 
            period=1, 
            rp_rstar=toi_row["rp_rstar_fit"], 
            sma_rstar=toi_row["sma_rstar_fit"], 
            inclination=toi_row["inclination_fit"], 
            ecc=0, 
            omega=0, 
            ld_model="nonlinear", 
            ld_coeff=obs_info[["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]].values, 
            time=folded_lc.time,)

        # Make plots
        plot_lightcurve_fit(
            lightcurve,
            folded_lc, 
            bm_params,
            bm_model, 
            bm_lightcurve, 
            toi_row["Period (days)"], 
            toi_row["Epoch (BJD)"], 
            toi_row["Duration (hours)"]/24,)

        # Set title
        title = (r"TOI {}    $[T_{{\rm eff}}={:0.0f}\,$K, $R_*={:0.2f}\,"
                 r"R_\odot$, $\frac{{R_p}}{{R_*}} = {:0.5f}$, "
                 r"$\frac{{a}}{{R_*}} = {:0.2f}$, $i = {:0.2f}]$")

        #import pdb
        #pdb.set_trace()
        title = title.format(toi, 
                     obs_info["teff_synth"], 
                     tic_info["radii_m19"],
                     toi_row["rp_rstar_fit"],
                     toi_row["sma_rstar_fit"],
                     toi_row["inclination_fit"],)
        plt.suptitle(title)

        # Set size and save
        plt.gcf().set_size_inches(16, 9)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt_save_loc = os.path.join(
            plot_path, 
            "lc_fit_{}_{}.pdf".format(toi, tic))
        plt.savefig(plt_save_loc)

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


def plot_std_rv_comparison(observations, std_params):
    """
    """
    rvs_gaia = []
    e_rvs_gaia = []
    markers = []

    for star_i in range(0, len(observations)):
        rv, e_rv = get_gaia_rv(observations.iloc[star_i]["uid"], std_params)
        rvs_gaia.append(rv)
        e_rvs_gaia.append(e_rv)

    rvs_gaia = np.array(rvs_gaia)
    e_rvs_gaia = np.array(e_rvs_gaia)

    e_rvs = np.sqrt(observations["e_rv"].values**2 + 
                    (3*np.ones(len(observations)))**2)

    #e_rvs = observations["e_rv"].values
    plt.close("all")
    fig, axis = plt.subplots(sharex=True)

    # Setup lower panel for residuals
    #plt.setp(axis.get_xticklabels(), visible=False)
    divider = make_axes_locatable(axis)
    res_ax = divider.append_axes("bottom", size="30%", pad=0)
    axis.figure.add_axes(res_ax, sharex=axis)
    xx = np.arange(-200,100) 
    axis.plot(xx, xx, color="black") 
    plt.setp(axis.get_xticklabels(), visible=False)
    
    axis.errorbar(rvs_gaia, observations["rv"].values, yerr=e_rvs, 
                 xerr=e_rvs_gaia, fmt=".", zorder=0)
    scatter = axis.scatter(rvs_gaia, observations["rv"].values, 
                           c=observations["teff_fit_rv"], marker="o", 
                           cmap="magma", zorder=1)

    cb = fig.colorbar(scatter, ax=axis)
    cb.set_label(r"$T_{\rm eff}$")

    # Residuals
    res_ax.errorbar(rvs_gaia,
                    observations["rv"].values-rvs_gaia,  
                    yerr=e_rvs, 
                    xerr=e_rvs_gaia, fmt=".", zorder=0) 
    res_ax.scatter(rvs_gaia, 
                   observations["rv"].values - rvs_gaia, 
                    c=observations["teff_fit_rv"], marker="o", 
                    cmap="magma", zorder=1)
    res_ax.hlines(0, -200, 100, color="black") 

    axis.set_xlim(-200,100)
    res_ax.set_xlim(-200,100)
    res_ax.set_ylim(-20, 20)

    res_ax.set_xlabel(r"RV, Gaia DR2 (km$\,$s$^{-1}$)")
    res_ax.set_ylabel(r"RV, residuals (km$\,$s$^{-1}$)")
    axis.set_ylabel(r"RV, WiFeS (km$\,$s$^{-1}$)")

    med_diff = np.nanmedian(observations["rv"].values-rvs_gaia)
    print("Median RV difference: %f km/s" % med_diff)