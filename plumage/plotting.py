"""Various plotting functions"
"""
from __future__ import print_function, division
import os
import numpy as np
import glob
import matplotlib.pylab as plt
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

def plot_teff_sorted_spectra(spectra, observations, catalogue=None, arm="r",
                             mask=None, suffix="", normalise=False):
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

    for sp_i, (spec, id, teff, logg, feh, rv, e_rv) in enumerate(
        zip(sorted_spec, ids, teffs, loggs, fehs, rvs, e_rvs)): 
        # Rescale if normalising
        if normalise:
            spec_scale = np.nanmedian(spec[1,:])
        else:
            spec_scale = 1

        plt.plot(spec[0,:], sp_i+spec[1,:]/spec_scale, linewidth=0.1) 
        #label = "%s [%i K, %0.2f km/s]" % (id, teff, rv)

        if catalogue is not None:
            uid = uids[sp_i]
            if uid == "":
                program = "?"
                subset = "?"
            else:
                try:
                    idx = int(np.argwhere(catalogue["source_id"].values==uid))  
                except:
                    import pdb
                    pdb.set_trace()
                program = catalogue.iloc[idx]["program"]
                subset = catalogue.iloc[idx]["subset"]

            label = (r"%s [%s, %s, %i K, %0.1f, %0.2f, %0.2f$\pm$%0.2f km/s]"
                    % (id, program, subset, teff, logg, feh, rv, e_rv))
        
        plt.text(spec[0,:].mean(), sp_i+0.5, label, fontsize=4, 
                        ha="center")

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

    
def plot_synthetic_fit(wave, spec_sci, e_spec_sci, spec_synth, params, date_id, 
                      plot_path):
    """TODO: Sort out proper sharing of axes
    """
    plt.close("all")
    fig, axis = plt.subplots()#sharex=True)

    # Setup lower panel for residuals
    plt.setp(axis.get_xticklabels(), visible=False)
    divider = make_axes_locatable(axis)
    res_ax = divider.append_axes("bottom", size="30%", pad=0)
    axis.figure.add_axes(res_ax, sharex=axis)

    axis.errorbar(wave, spec_sci, yerr=e_spec_sci, label="sci", linewidth=0.2,
                  elinewidth=0.2, barsabove=True, capsize=0.3, capthick=0.1)
    axis.plot(wave, spec_synth, "--", label="synth", linewidth=0.2)

    param_label = r"$T_{{\rm eff}}$ = {:0.0f} K, $\log g$ = {:0.2f}, [Fe/H] = {:0.2f}"
    param_label = param_label.format(params[0], params[1], params[2])
    axis.text(np.nanmean(wave), 0.7, param_label, horizontalalignment="center")

    res_ax.hlines(0, wave[0]-100, wave[-1]+100, linestyles="dotted", linewidth=0.2)
    res_ax.plot(wave, spec_sci-spec_synth, linewidth=0.2, color="red")
    axis.set_xlim(wave[0]-10, wave[-1]+10)
    axis.set_ylim(0.1, 1.3)
    res_ax.set_xlim(wave[0]-10, wave[-1]+10)
    res_ax.set_ylim(-0.25, 0.25)

    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.legend(loc="best")
    axis.set_ylabel("Flux (Normalised)")
    res_ax.set_ylabel("Residuals")
    res_ax.set_xlabel("Wavelength (A)")

    plt.suptitle(date_id)
    plt.savefig(plot_path)


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