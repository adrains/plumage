"""Script to flux calibrate MIKE spectra.
"""
import numpy as np
import plumage.spectra_mike as psm
import plumage.utils_mike as pum
import plumage.plotting_mike as ppltm
from PyAstronomy.pyasl import instrBroadGaussFast
from astropy.io import fits
import astropy.constants as const

# Update for line breaks
np.set_printoptions(linewidth=150)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
stellar_templates = {
    "5709390701922940416":"templates/template_HD74000_250624_norm.fits",  # 6211, 4.155, -1.98
    "3510294882898890880":"templates/template_HD111980_250624_norm.fits", # 5578, 3.89, -1.08
    "22745910577134848":None,                                             # B star, no MARCS
    "4201781696994073472":None,                                           # WD, no MARCS
    "5164707970261890560":"templates/template_epsEri_250709_norm.fits",   # 5052, 4.63, -0.08 (GBS)
    "4376174445988280576":"templates/template_bd02d3375_250709_norm.fits",# 5950,3.97, -2.27 (SIMBAD)
    "5957698605530828032":"templates/template_HD160617_250709_norm.fits", # 5931, 3.74, -1.79 (SIMBAD)
    "6342346358822630400":"templates/template_HD185975_250709_norm.fits",# 5570, 4.12, -0.23 (Gaia DR3)
    "6477295296414847232":"templates/template_HD200654_250709_norm.fits", # 5219, 2.7, -2.88 (SIMBAD)
    "1779546757669063552":"templates/template_HD209458_250709_norm.fits", # 6100, 4.5, 0.03 (SIMBAD)
}

# https://github.com/PyWiFeS/pywifes/tree/main/reference_data
calspec_templates = {
    "5709390701922940416":"data/flux_standards/hd074000_stis_007.txt",
    "3510294882898890880":"data/flux_standards/hd111980_stis_007.txt",
    "22745910577134848":"data/flux_standards/ksi2ceti_stis_008.txt",
    "4201781696994073472":"data/flux_standards/gj7541a_stis_004.txt",
    "5164707970261890560":"data/flux_standards/hd022049.dat",
    "4376174445988280576":"data/flux_standards/bd02d3375_stis_008.txt",
    "5957698605530828032":"data/flux_standards/hd160617_stis_006.txt",
    "6342346358822630400":"data/flux_standards/hd185975_stis_008.txt",
    "6477295296414847232":"data/flux_standards/hd200654_stis_008.txt",
    "1779546757669063552":"data/flux_standards/hd209458_stisnic_008.txt",}

# Fitted via scripts_reduction/calc_rv_for_calspec_template.py
calspec_rvs = {
    "5709390701922940416":178,
    "3510294882898890880":133,
    "22745910577134848":10,
    "4201781696994073472":0,        # WD
    "5164707970261890560":-82,
    "4376174445988280576":-442,
    "5957698605530828032":80,
    "6342346358822630400":-79,
    "6477295296414847232":-85,
    "1779546757669063552":-72,
}

# Initial guess scale terms
flux_standard_tau_scale_terms = {
    ("6477295296414847232", "2025-07-02"):(1.1, 1.2),
    ("22745910577134848",   "2021-11-22"):(1.2, 0.6),
    ("4376174445988280576", "2025-07-02"):(1.1, 1.1),
    ("6342346358822630400", "2025-07-02"):(1.3, 1.6),   # problem ?
    ("4201781696994073472", "2025-07-02"):(1.1, 1.1),
    ("1779546757669063552", "2025-07-02"):(1.2, 1.5),   # problem ?
    ("5164707970261890560", "2025-07-02"):(1.3, 1.6),   # problem ?
    ("5957698605530828032", "2025-07-02"):(1.1, 1.0),
    ("3510294882898890880", "2025-05-19"):(1.1, 1.2),
    ("3510294882898890880", "2025-05-20"):(1.1, 0.9),
    ("5709390701922940416", "2025-05-19"):(1.3, 1.4),

}

# Telluric transmission
# https://github.com/mzechmeister/viper/tree/master/lib/atmos
telluric_template = "data/viper_stdAtmos_vis.fits"

resolving_power = 46000

plot_folder = "spectra/mike_MK_unnormalised/flux_calibration/"

# -----------------------------------------------------------------------------
# Prepare telluric template
# -----------------------------------------------------------------------------
with fits.open(telluric_template) as tt:
    wave_tt = tt[1].data["lambda"]
    trans_H2O = tt[1].data["H2O"]
    trans_O2 = tt[1].data["O2"]

trans_H2O_broad = instrBroadGaussFast(
    wvl=wave_tt,
    flux=trans_H2O,
    resolution=resolving_power,
    edgeHandling="firstlast",
    maxsig=5,
    equid=True,)

trans_O2_broad = instrBroadGaussFast(
    wvl=wave_tt,
    flux=trans_O2,
    resolution=resolving_power,
    edgeHandling="firstlast",
    maxsig=5,
    equid=True,)

# Convert telluric wavelength scale to air wavelengths to match MIKE
wave_tt_vac = pum.convert_vacuum_to_air_wl(wave_tt)

# -----------------------------------------------------------------------------
# Import + settings
# -----------------------------------------------------------------------------
# Settings
path= "spectra"
arm = "r"
label = "KM_noflat"
poly_order = 8
optimise_order_overlap = False
fit_for_telluric_scale_terms = False
do_convolution = True
resolving_power_during_fit = 1000
run_on_order_subset = False
order_subset = [50, 51, 52, 53, 54]

# Cleaning
clean_input_spectra = True
edge_px_to_mask = 100
sigma_clip_blaze_corr_spectra = False
sigma_upper = 5

# Settings for polynomial 'blaze correction'
do_flat_field_blaze_corr = True
flat_norm_poly_order = 7
poly_coef_csv_path = "data/"
set_px_to_nan_beyond_domain = True

# Import MIKE spectra
wave, spec, sigma, orders = pum.load_3D_spec_from_fits(
    path=path, label=label, arm=arm)
obs_info = pum.load_fits_table("OBS_TAB", "KM_noflat",)

# Grab dimensions for convenience
(n_star, n_order, n_px) = wave.shape

# [Optional] Clean spectra
if clean_input_spectra:
    bad_px_mask = np.any(a=[spec <= 0, sigma <= 0,], axis=0,)

    if edge_px_to_mask > 0:
        bad_px_mask[:,:,:edge_px_to_mask] = np.nan
        bad_px_mask[:,:,-edge_px_to_mask:] = np.nan

    spec[bad_px_mask] = np.nan
    sigma[bad_px_mask] = np.nan

    # Clean order #71 (bluest order) by trimming the edges.
    o71 = int(np.argwhere(orders == 71))
    o71_clip_mask = np.logical_or(wave[:,o71] < 4818, wave[:,o71] > 4880)

    for star_i in range(n_star):
        spec[star_i,o71,o71_clip_mask[star_i]] = np.nan

# [Optional] Normalise by flat fields
if do_flat_field_blaze_corr:
    spec, sigma = \
        psm.normalise_mike_all_spectra_by_norm_flat_field(
            wave_3D=wave,
            spec_3D=spec,
            sigma_3D=sigma,
            orders=orders,
            arm="r",
            poly_order=flat_norm_poly_order,
            base_path=poly_coef_csv_path,
            set_px_to_nan_beyond_domain=set_px_to_nan_beyond_domain,)

# [Optional] Run on subset of orders for testing
if run_on_order_subset:
    order_mask = np.isin(orders, order_subset)

    wave = wave[:,order_mask]
    spec = spec[:,order_mask]
    sigma = sigma[:,order_mask]
    orders = orders[order_mask]

is_spphot = obs_info["is_spphot"].values
wave_sp = wave[is_spphot]
spec_sp = spec[is_spphot]
sigma_sp = sigma[is_spphot]
obs_info_sp = obs_info[is_spphot]

# Grab dimensions for convenience
n_spphot = np.sum(is_spphot)

# Initialise arrays to hold all polynomial coefficients and the domain values
spphot_poly_coeff = np.full((n_spphot, n_order, poly_order), np.nan)
spphot_wave_mins = np.full((n_spphot, n_order,), np.nan)
spphot_wave_maxes = np.full((n_spphot, n_order,), np.nan)

# -----------------------------------------------------------------------------
# Running on all SpPhot targets
# -----------------------------------------------------------------------------
for si, (star_i, star_data) in enumerate(obs_info_sp.iterrows()):
    # Grab source ID
    source_id = star_data["source_id"]
    ut_date = star_data["ut_date"]

    print("-"*160,
          "{}/{} - Gaia DR3 {}".format(si+1, n_spphot, source_id),
          "-"*160, sep="\n")

    plot_label = "{}_{}".format(
        obs_info_sp.loc[star_i]["ut_date"], source_id)

    rv = star_data["rv_r"]
    bcor = star_data["bcor"]
    airmass = star_data["airmass"]

    # Grab initial guess tau scaling terms
    tau_scale_O2 = flux_standard_tau_scale_terms[(source_id, ut_date)][0]
    tau_scale_H2O = flux_standard_tau_scale_terms[(source_id, ut_date)][1]

    # -------------------------------------------------------------------------
    # Flux standard spectrum
    # -------------------------------------------------------------------------
    # Import flux standard + convert to air wavelengths
    fs = np.loadtxt(calspec_templates[source_id])
    wave_fs_vac = fs[:,0]
    wave_fs_air = pum.convert_vacuum_to_air_wl(fs[:,0])

    # HACK (?) Only do relative flux calibration
    spec_fs = fs[:,1] / np.nanmedian(fs[:,1])

    cs_rv = calspec_rvs[source_id]

    # Doppler shift
    wave_fs_ds_vac = wave_fs_vac * (1-(cs_rv-bcor)/(const.c.si.value/1000))
    wave_fs_ds_air = wave_fs_air * (1-(cs_rv-bcor)/(const.c.si.value/1000))

    # -------------------------------------------------------------------------
    # Synthetic template spectrum
    # -------------------------------------------------------------------------
    # HACK Make a dummy template if we don't have one--only used for plotting
    if stellar_templates[source_id] is None:
        print("No stellar template")
        wave_synth = wave_fs_ds_air.copy()
        spec_synth = np.ones_like(wave_synth)
        
    else:
        ff = fits.open(stellar_templates[source_id])
        wave_synth = ff["WAVE"].data
        spec_synth = ff["SPEC"].data[0]

    # Doppler shift
    wave_synth_ds = wave_synth * (1-(rv-bcor)/(const.c.si.value/1000))

    # -------------------------------------------------------------------------
    # MIKE spectra
    # -------------------------------------------------------------------------
    # Select spectrum and normalise to median, since we don't care to attempt
    # *absolute* flux calibration.
    med = np.nanmedian(spec_sp[si][1:])

    wave_obs_2D = wave_sp[si]
    spec_obs_2D = spec_sp[si] / med
    sigma_obs_2D = sigma[si] / med

    # -------------------------------------------------------------------------
    # Flux calibration
    # -------------------------------------------------------------------------
    fit_dict = psm.fit_atmospheric_transmission(
        wave_obs_2D=wave_obs_2D,
        spec_obs_2D=spec_obs_2D,
        sigma_obs_2D=sigma_obs_2D, 
        wave_telluric=wave_tt_vac,
        trans_H2O_telluric=trans_H2O_broad,
        trans_O2_telluric=trans_O2_broad,
        wave_fluxed=wave_fs_ds_air,
        spec_fluxed=spec_fs,
        wave_synth=wave_synth_ds,
        spec_synth=spec_synth,
        airmass=airmass,
        scale_O2=tau_scale_O2,
        scale_H2O=tau_scale_H2O,
        do_convolution=do_convolution,
        resolving_power_during_fit=resolving_power_during_fit,
        poly_order=poly_order,
        optimise_order_overlap=optimise_order_overlap,
        fit_for_telluric_scale_terms=fit_for_telluric_scale_terms)

    # Diagnostic plot
    ppltm.plot_flux_calibration(
        fit_dict, plot_folder=plot_folder, plot_label=plot_label,)

    # Save coefficients to disk
    pum.save_flux_calibration_poly_coeff(
        poly_order=poly_order,
        poly_coeff=fit_dict["poly_coef"],
        wave_mins=fit_dict["wave_mins"],
        wave_maxes=fit_dict["wave_maxes"],
        orders=orders,
        arm=arm,
        label=plot_label,
        save_path=plot_folder,)

    # Save
    spphot_poly_coeff[si] = fit_dict["poly_coef"]
    spphot_wave_mins[si] = fit_dict["wave_mins"]
    spphot_wave_maxes[si] = fit_dict["wave_maxes"]

# -----------------------------------------------------------------------------
# Compute mean of all polynomials
# -----------------------------------------------------------------------------
# Compute means
poly_coef_mean = np.nanmean(spphot_poly_coeff, axis=0)
wave_mins_mean = np.nanmean(spphot_wave_mins, axis=0)
wave_maxes_mean = np.nanmean(spphot_wave_maxes, axis=0)

fn_label = "{}x_SpPhot_mean".format(n_spphot)

pum.save_flux_calibration_poly_coeff(
    poly_order=poly_order,
    poly_coeff=poly_coef_mean,
    wave_mins=wave_mins_mean,
    wave_maxes=wave_maxes_mean,
    orders=orders,
    arm=arm,
    label=fn_label,
    save_path=plot_folder,)

# -----------------------------------------------------------------------------
# Flux calibrating everything
# -----------------------------------------------------------------------------
# TODO: move this to its own script
# Flux calibrate all spectra
spectra_fc = np.full_like(spec, np.nan)
sigma_fc = np.full_like(spec, np.nan)

for star_i in range(n_star):
    spec_fc, sig_fc = psm.flux_calibrate_mike_spectrum(
        wave_2D=wave[star_i],
        spec_2D=spec[star_i],
        sigma_2D=sigma[star_i],
        airmass=obs_info.iloc[star_i]["airmass"],
        poly_order=poly_order,
        arm=arm,
        coeff_save_folder=plot_folder,
        label=fn_label,)

    spectra_fc[star_i] = spec_fc
    sigma_fc[star_i] = sig_fc

mm = np.argsort(obs_info["teff_template"].values)

ppltm.plot_all_flux_calibrated_spectra(
    wave_3D=wave[mm],
    spec_3D=spectra_fc[mm],
    sigma_3D=sigma_fc[mm],
    object_ids=obs_info["object"].values[mm],
    figsize=(16,50),
    plot_folder=plot_folder,
    plot_label=arm,)