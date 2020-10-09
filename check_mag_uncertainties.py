"""
"""
import pandas as pd
import numpy as np
import plumage.synthetic as synth
import plumage.plotting as pplt
import plumage.utils as utils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
import astropy.convolution as conv 

spec_path = "spectra"

# Import
spec_std_b, spec_std_r, obs_std = utils.load_fits("std", path=spec_path)
#synth_std_b = utils.load_fits_image_hdu("synth", "std", arm="b")
#synth_std_r = utils.load_fits_image_hdu("synth", "std", arm="r")
synth_std_lit_b = utils.load_fits_image_hdu("synth_lit", "std", arm="b")
synth_std_lit_r = utils.load_fits_image_hdu("synth_lit", "std", arm="r")

# Mask to only those with literature synthetic spectra
mask = ~np.isnan(np.sum(synth_std_lit_b, axis=1))
spec_std_b = spec_std_b[mask]
spec_std_r = spec_std_r[mask]
obs_std = obs_std[mask]
synth_std_lit_b = synth_std_lit_b[mask]
synth_std_lit_r = synth_std_lit_r[mask]

std_info = utils.load_info_cat("data/std_info.tsv")
obs_std = obs_std.join(std_info, "source_id", how="inner", rsuffix="_") 


star_dict = {
    134:("Teff~3192, logg~5.08, [FeH]~-0.02, Gl 447 (3796072592206250624)",
         "phoenix/lte032-5.0-0.0a+0.0.BT-Settl.7",
         "phoenix/lte032.0-5.0-0.0a+0.0.BT-Settl.spec.7"),
    97:("Teff~3261 logg~4.95, [Fe/H]~0.16, GJ 3379 (3316364602541746048)",
        "",
        "",),
    4:("Teff~3298 logg~4.83, [Fe/H]~0.17, Gl 876 (2603090003484152064)",
        "",
        "",),
    23:("Teff~3530, logg~4.78, [FeH]~+0.35, Gl 849 (2627117287488522240)",
        "phoenix/lte035-5.0+0.3a+0.0.BT-Settl.7",
        "phoenix/lte035.0-5.0-0.0a+0.0.BT-Settl.spec.7"),
    6:("Teff~3720, logg~4.72, [FeH]~+0.21, PM I22565+1633 (2828928008202069376)",
        "phoenix/lte037-4.5+0.3a+0.0.BT-Settl.7",
        "phoenix/lte037.0-4.5-0.0a+0.0.BT-Settl.spec.7"),
    10:("Teff~3900, logg~4.67, [FeH]~+0.14, PM I01528-2226 (5134635708766250752)",
        "phoenix/lte039-4.5-0.0a+0.0.BT-Settl.7",
        "phoenix/lte039.0-4.5-0.0a+0.0.BT-Settl.spec.7"),
    48:("Teff~4013, logg~4.65, [FeH]~-0.05, PM I23099+1425W (2815543034682035840)",
        "phoenix/lte040-4.5-0.0a+0.0.BT-Settl.7",
        "phoenix/lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7"),}

use_btsettl = False

teffs = []
flux_pcs_all = []
mag_diffs_all = []

for star_i in range(len(obs_std)):
    #print(star_dict[i][0])

    # Import BT-Settl spectra
    if use_btsettl:
        spec_bts = pd.read_csv(star_dict[i][1], delim_whitespace=True,
            names=["wl", "flux", "bb_flux"]+list(np.arange(0,23)))
        
        sort_bts = np.argsort(spec_bts["wl"].values)
        wl_bts = spec_bts["wl"][sort_bts].values
        flux_bts = 10**(spec_bts["flux"][sort_bts].values-8)

        bts_b_mask = np.logical_and(wl_bts > 3500, wl_bts < 5700)
        bts_r_mask = np.logical_and(wl_bts > 5400, wl_bts < 7000)

        wl_bts_b_all = wl_bts[bts_b_mask]
        flux_bts_b_all = flux_bts[bts_b_mask]

        wl_bts_r_all = wl_bts[bts_r_mask]
        flux_bts_r_all = flux_bts[bts_r_mask]

        # Convolve
        gc_b =  conv.Gaussian1DKernel(0.77, x_size=len(wl_bts_b_all))
        gc_r =  conv.Gaussian1DKernel(0.44, x_size=len(wl_bts_r_all))

        flux_bts_b_conv = conv.convolve_fft(flux_bts_b_all, gc_b)
        flux_bts_r_conv = conv.convolve_fft(flux_bts_r_all, gc_r)

        # Regrid
        n_b = int(np.round(len(wl_bts_b_all) / spec_std_b.shape[2]))
        cutoff_b = len(wl_bts_b_all) % n_b

        if cutoff_b == 0:
            wl_bts_b_conv_rg = wl_bts_b_all.reshape(-1, n_b).mean(axis=1)
            flux_bts_b_conv_rg = flux_bts_b_conv.reshape(-1, n_b).sum(axis=1)
        else:
            wl_bts_b_conv_rg = wl_bts_b_all[:-cutoff_b].reshape(-1, n_b).mean(axis=1)
            flux_bts_b_conv_rg = flux_bts_b_conv[:-cutoff_b].reshape(-1, n_b).sum(axis=1)

        n_r = int(np.round(len(wl_bts_r_all) / spec_std_r.shape[2]))
        cutoff_r = len(wl_bts_r_all) % n_r

        if cutoff_r == 0:
            wl_bts_r_conv_rg = wl_bts_r_all.reshape(-1, n_r).mean(axis=1)
            flux_bts_r_conv_rg = flux_bts_r_conv.reshape(-1, n_r).sum(axis=1)
        else:
            wl_bts_r_conv_rg = wl_bts_r_all[:-cutoff_r].reshape(-1, n_r).mean(axis=1)
            flux_bts_r_conv_rg = flux_bts_r_conv[:-cutoff_r].reshape(-1, n_r).sum(axis=1)

        # Normalise BT-Settl red by max
        flux_bts_r_conv_rg = flux_bts_r_conv_rg / np.nanmax(flux_bts_r_conv_rg)
        norm_bts_r = np.nanmean(flux_bts_r_conv_rg[wl_bts_r_conv_rg < 5445])

        # Normalise BT-Settl blue by overlap
        norm_mask_bts_b = np.logical_and(wl_bts_b_conv_rg > 5400, wl_bts_b_conv_rg < 5445)
        flux_bts_b_conv_rg = flux_bts_b_conv_rg / np.nanmean(flux_bts_b_conv_rg[norm_mask_bts_b]) * norm_bts_r

        # Combine
        wl_bts_br = np.concatenate((wl_bts_b_conv_rg, wl_bts_r_conv_rg))
        flux_bts_br = np.concatenate((flux_bts_b_conv_rg, flux_bts_r_conv_rg))

    # -------------------------------------------------------------------------
    # Normalise red by median
    wl_r = spec_std_r[star_i,0]
    spec_r = spec_std_r[star_i,1] / np.nanmedian(spec_std_r[star_i,1])
    ref_r = synth_std_lit_r[star_i] / np.nanmedian(synth_std_lit_r[star_i])

    # Normalise blue by overlap region

    norm_obs = np.nanmean(spec_r[wl_r < 5445])
    norm_synth = np.nanmean(ref_r[wl_r < 5445])

    wl_b = spec_std_b[star_i,0]
    norm_mask = np.logical_and(wl_b > 5400, wl_b < 5445)

    spec_b = spec_std_b[star_i,1] / np.nanmean(spec_std_b[star_i,1][norm_mask]) * norm_obs
    ref_b = synth_std_lit_b[star_i] / np.nanmean(synth_std_lit_b[star_i][norm_mask]) * norm_synth

    blue_mask = wl_b < 5400

    wl_b = wl_b[blue_mask]
    spec_b = spec_b[blue_mask]
    ref_b = ref_b[blue_mask]

    # Combine
    wl = np.concatenate((wl_b, wl_r))
    spec_obs = np.concatenate((spec_b, spec_r))
    spec_synth = np.concatenate((ref_b, ref_r))

    # Plot
    plt.close("all")
    plt.plot(wl, spec_obs, linewidth=0.5, label="Observed", alpha=0.8)
    plt.plot(wl, spec_synth, linewidth=0.5, label="MARCS", alpha=0.8)

    if use_btsettl:
        plt.plot(wl_bts_br, flux_bts_br, linewidth=0.5, label="BT-Settl", alpha=0.5)

    # Filter profiles
    filters = ["v", "g", "r", "BP", "RP"]

    flux_pcs = []
    mag_diffs = []

    for filt_i, filt in enumerate(filters):
        wl_f, fp = synth.load_filter_profile(filt, 3000, 7000, do_zero_pad=True)
        plt.plot(wl_f, fp, linewidth=1.0)

        # Integrate
        calc_filt_profile = interp1d(wl_f, fp,  bounds_error=False, fill_value=0)
        filt_profile_sci = calc_filt_profile(wl)

        flux_obs = wl*spec_obs*filt_profile_sci
        flux_obs = simps(flux_obs[~np.isnan(flux_obs)])

        flux_synth = wl*spec_synth*filt_profile_sci
        flux_synth = simps(flux_synth[~np.isnan(flux_synth)])

        flux_pc = flux_synth / flux_obs * 100
        mag_diff = -2.5*np.log10(flux_obs) - -2.5*np.log10(flux_synth)

        text = "{} --> {:0.3f} ({:0.2f}%)".format(filt, mag_diff, flux_pc)
        #print(text)
        plt.text(3300, 1-0.05*filt_i, text, horizontalalignment="left",)

        flux_pcs.append(flux_pc)
        mag_diffs.append(mag_diff)

    flux_pcs_all.append(flux_pcs)
    mag_diffs_all.append(mag_diffs)

    # Get temperature
    teff_m15 = obs_std.iloc[star_i]["teff_m15"]
    teff_ra12 = obs_std.iloc[star_i]["teff_ra12"]
    feh_m15 = obs_std.iloc[star_i]["feh_m15"]
    feh_ra12 = obs_std.iloc[star_i]["feh_ra12"]

    teff = teff_m15 if ~np.isnan(teff_m15) else teff_ra12
    feh = feh_m15 if ~np.isnan(feh_m15) else feh_ra12

    teffs.append(teff)

    source_id = obs_std.iloc[star_i]["id"]
    simbad_id = obs_std.iloc[star_i]["simbad_name"]

    title = "{} ({}) - Teff~{:0.0f}, [Fe/H]~{:0.2f}".format(
        simbad_id, source_id, teff, feh)

    plt.legend(loc='lower right')
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux (Normalised)")
    plt.title(title)
    plt.xlim(3200, 7000)
    plt.ylim(0,2.5)
    plt.gcf().set_size_inches(16, 8)
    plt.tight_layout()
    plt.savefig("plots/mag_check/{}_{}.pdf".format(teff, obs_std.iloc[star_i]["id"]))

pplt.merge_spectra_pdfs("plots/mag_check/*.pdf", "plots/integration.pdf")