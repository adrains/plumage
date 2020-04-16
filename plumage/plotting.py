"""Various plotting functions"
"""
from __future__ import print_function, division
import os
import numpy as np
import glob
import matplotlib.pylab as plt
import matplotlib.colors as mplc
import plumage.spectra as spec
from tqdm import tqdm
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
    teff_order = np.argsort(observations[mask]["teff_fit"].values)
    sorted_spec = spectra[mask][teff_order]
    ids = observations[mask]["id"].values[teff_order]
    uids = observations[mask]["uid"].values[teff_order]
    teffs = observations[mask]["teff_fit"].values[teff_order]
    loggs = observations[mask]["logg_fit"].values[teff_order]
    fehs = observations[mask]["feh_fit"].values[teff_order]
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

    if show_telluric_box:
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
def plot_synthetic_fit(wave, spec_sci, e_spec_sci, spec_synth, bad_px_mask, 
        obs_info, param_cols, date_id, plot_path, fig=None, axis=None):
    """TODO: Sort out proper sharing of axes

    Parameters
    ----------

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

    # Plot residuals
    res_ax.hlines(0, wave[0]-100, wave[-1]+100, linestyles="dotted", linewidth=0.2)
    res_ax.plot(wave, spec_sci-spec_synth, linewidth=0.2, color="red")
    axis.set_xlim(wave[0]-10, wave[-1]+10)
    axis.set_ylim(0.0, 1.7)
    res_ax.set_xlim(wave[0]-10, wave[-1]+10)
    res_ax.set_ylim(-0.3, 0.3)

    # Plot vertical bars
    wl_delta = (wave[1] - wave[0]) / 2
    for xi in range(len(wave)):
        if bad_px_mask[xi]:
            axis.axvspan(wave[xi]-wl_delta, wave[xi]+wl_delta, ymin=0, 
                         ymax=1.7, alpha=0.05, color="red")
            res_ax.axvspan(wave[xi]-wl_delta, wave[xi]+wl_delta, ymin=-1, 
                           ymax=1, alpha=0.05, color="red")

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Label synthetic fit parameters
    params = obs_info[param_cols]
    param_label = r"$T_{{\rm eff}}$ = {:0.0f} K, $\log g$ = {:0.2f}, [Fe/H] = {:0.2f}"
    param_label = param_label.format(params[0], params[1], params[2])
    axis.text(np.nanmean(wave), 1.6, param_label, horizontalalignment="center")

    if "rchi2_synth" in obs_info:
        rchi2_label = r"red. $\chi^2 = {:0.0f}$".format(obs_info["rchi2_synth"])
        axis.text(np.nanmean(wave), 1.525, rchi2_label, horizontalalignment="center")

    # Label RVs
    rv_label = r"RV$ = {:0.2f}\pm{:0.2f}\,$km$\,$s$^{{-1}}$"
    rv_label = rv_label.format(obs_info["rv"], obs_info["e_rv"])
    axis.text(np.nanmean(wave), 1.45, rv_label, horizontalalignment="center")

    # Label SNR
    snr_label = r"SNR (R) $\sim {:0.0f}$".format(obs_info["snr_r"])
    axis.text(np.nanmean(wave), 1.375, snr_label, horizontalalignment="center")

    # Label date
    date_label = r"Date = {}".format(obs_info["date"].split("T")[0])
    axis.text(np.nanmean(wave), 1.30, date_label, horizontalalignment="center")

    # Label airmass
    airmass_label = r"Airmass = {:0.2f}".format(obs_info["airmass"])
    axis.text(np.nanmean(wave), 1.225, airmass_label, horizontalalignment="center")

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
    axis.legend(loc="best")
    axis.set_ylabel("Flux (Normalised)")
    res_ax.set_ylabel("Residuals")
    res_ax.set_xlabel("Wavelength (A)")

    #plt.suptitle(date_id)
    #plt.savefig(plot_path)


def plot_synth_fit_diagnostic(
    wave,
    spec_sci,
    e_spec_sci,
    spec_synth,
    bad_px_mask,
    obs_info,
    info_cat,
    plot_path,
    is_tess=False):
    """Want to plot TESS CMD (highlighting the current star) next to the 
    synthetic fit, along with associated flags.
    """
    # Initialise
    plt.close("all")
    fig, (ax_cmd, ax_spec) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]})
    #fig.subplots_adjust(top=0.9)

    # Plot up the CMD side
    if "pc" in info_cat:
        mask = np.logical_and(info_cat["observed"], info_cat["pc"])
    else:
        mask = info_cat["observed"]

    # First plot all the points
    ax_cmd.plot(
        info_cat[mask]["Bp-Rp"], 
        info_cat[mask]["G_mag_abs"], 
        "o",
        c="blue",
    )

    # ID
    if is_tess:
        sid = float(obs_info["id"].replace("TOI",""))
        star_info = info_cat[info_cat["TOI"]==sid].iloc[0]

        plt.suptitle("TOI {}, Gaia DR2 {}".format(sid, obs_info["uid"]))
    
    else:
        sid = obs_info["uid"]
        is_obs = info_cat["observed"].values
        star_info = info_cat[is_obs][info_cat[is_obs]["source_id"]==sid]

        if len(star_info) < 1:
            return
        elif len(star_info) > 1:
            return
        else:
            star_info = star_info.iloc[0]

        plt.suptitle("{}, Gaia DR2 {}".format(star_info["ID"], sid))
    

    # Then just plot our current star
    ax_cmd.plot(star_info["Bp-Rp"], star_info["G_mag_abs"], "o", c="red")

    # Flip magnitude axis
    ymin, ymax = ax_cmd.get_ylim()
    ax_cmd.set_ylim((ymax, ymin))

    ax_cmd.set_xlabel(r"$B_P-R_P$")
    ax_cmd.set_ylabel(r"$G_{\rm abs}$")

    # Plot diagnostic info
    ax_cmd.text(2.4, 6.4, "RUWE = {:.2f}".format(float(star_info["ruwe"])),
        horizontalalignment="center")
    ax_cmd.text(2.4, 6.6, 
        r"$T_{{{{\rm eff}}, (B_p-R_p)}}  = {:.0f} \pm {:.0f}\,K$".format(
            float(star_info["teff_m15_bprp"]), 
            float(star_info["e_teff_m15_bprp"])),
        horizontalalignment="center")
    ax_cmd.text(2.4, 6.8, 
        r"$T_{{{{\rm eff}}, (B_p-R_p, J-H)}} = {:.0f} \pm {:.0f}\,K$".format(
            float(star_info["teff_m15_bprp_jh"]), 
            float(star_info["e_teff_m15_bprp_jh"])),
        horizontalalignment="center")
    ax_cmd.text(2.4, 7.00, 
        r"$M_{{K_s}} = {:.2f} \pm {:.2f}\,M_{{\odot}}$".format(
            float(star_info["mass_m19"]), 
            float(star_info["e_mass_m19"])),
        horizontalalignment="center")
    ax_cmd.text(2.4, 7.20, 
        "Blended 2MASS = {}".format(bool(int(star_info["blended_2mass"]))),
        horizontalalignment="center")
    ax_cmd.text(2.4, 7.4, 
        "# Obs = {}".format(int(star_info["wife_obs"])),
        horizontalalignment="center")
    ax_cmd.text(2.4, 7.6, 
        "Dist. = {:.2f} pc".format(float(star_info["dist"])),
        horizontalalignment="center")

    # Now do second part
    params = obs_info[["teff_synth", "logg_synth", "feh_synth"]]
    spec_sci, e_spec_sci = spec.norm_spec_by_wl_region(wave, spec_sci, "red", e_spec_sci)

    plot_synthetic_fit(
        wave, 
        spec_sci, 
        e_spec_sci, 
        spec_synth,
        bad_px_mask,
        obs_info,
        ["teff_synth", "logg_synth", "feh_synth"], 
        "", 
        "", 
        fig=fig, 
        axis=ax_spec)

    # If lit params are known, display
    if "teff" in star_info:
        lit_params = star_info[["teff", "e_teff", "logg", "feh", "e_feh"]].values
        lit_param_label = r"Lit Params ({}, {}): $T_{{\rm eff}} = {:0.0f} \pm {:0.0f}\,$K, $\log g = {:0.2f}$, [Fe/H]$ = {:0.2f} \pm {:0.2f}$"
        lit_param_label = lit_param_label.format(
            str(star_info["kind"]),
            str(star_info["source"]), 
            lit_params[0], 
            lit_params[1], 
            lit_params[2], 
            lit_params[3], 
            lit_params[4])
        ax_spec.text(np.nanmean(wave), 0.25, lit_param_label, horizontalalignment="center")

    plt.gcf().set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt_save_loc = os.path.join(plot_path, "{}_TOI{}.pdf".format(int(params[0]), sid))
    plt.savefig(plt_save_loc)


def plot_all_synthetic_fits(
    spectra_r, 
    synth_spec, 
    observations,
    bad_px_masks,
    label,
    info_cat):
    """
    """
    # Make plot path if it doesn't already exist
    plot_path = os.path.join("plots", "synth_diagnostics_{}".format(label))

    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # Plot diagnostics pdf for every spectrum
    for i in tqdm(range(len(observations)), desc="Plotting diagnostics"):
        plot_synth_fit_diagnostic(
            spectra_r[i,0], 
            spectra_r[i,1], 
            spectra_r[i,2], 
            synth_spec[i], 
            bad_px_masks[i], 
            observations.iloc[i], 
            info_cat,
            plot_path)

    # Merge plots
    glob_path = os.path.join(plot_path, "*TOI*")
    merge_spectra_pdfs(glob_path, "plots/synthetic_fits_{}.pdf".format(label))

# -----------------------------------------------------------------------------
# Comparisons
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
                           c=observations["teff_fit"], marker="o", 
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
                    c=observations["teff_fit"], marker="o", 
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


def plot_tess_cmd(
    tess_info, 
    plot_only_obs_pc=True, 
    plot_toi_ids=False,
    colour="Bp-Rp",
    abs_mag="G_mag_abs",
    ms=50
    ):
    """
    """
    # Make a mask
    if plot_only_obs_pc:
        mask = np.logical_and(tess_info["observed"], tess_info["pc"])
    else:
        mask = np.fill(len(tess_info), True)

    plt.close("all")
    fig, axis = plt.subplots()
    scatter = axis.scatter(
        tess_info[mask][colour], 
        tess_info[mask][abs_mag], 
        s=ms,
        c=tess_info[mask]["ruwe"]>1.4,
        cmap="seismic"
    )

    if plot_toi_ids:
        for star_i, star in tess_info[mask].iterrows():
            axis.text(
                star[colour], 
                star[abs_mag]-0.1,
                star["TOI"],
                horizontalalignment="center",
                fontsize="xx-small"
            )

    cb = fig.colorbar(scatter, ax=axis)
    cb.set_label("RUWE > 1.4")

    # Flip magnitude axis
    ymin, ymax = axis.get_ylim()
    axis.set_ylim((ymax, ymin))

    #plt.xlabel(r"$B_P-R_P$")
    #plt.ylabel(r"$G_{\rm abs}$")
    axis.set_xlabel(colour)
    axis.set_ylabel(abs_mag)
    fig.tight_layout()
    plt.savefig("paper/tess_cmd.pdf")


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