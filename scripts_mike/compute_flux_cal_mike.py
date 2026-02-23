"""Script to flux calibrate MIKE spectra.
"""
import numpy as np
import pandas as pd
import plumage.spectra_mike as psm
import plumage.utils_mike as pum
import plumage.plotting_mike as ppltm
from astropy.io import fits
import astropy.constants as const
import stannon.utils as su

# Update for line breaks
np.set_printoptions(linewidth=150)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

# Import flux standard dataframe.
flux_std_info = pd.read_csv(
    filepath_or_buffer=ms.flux_cal_std_info_fn,
    dtype={"source_id":str},
    delimiter="\t",)
flux_std_info.set_index("source_id", inplace=True)

# Properly handle missing templates by repacing np.nan with None.
marcs_templates = flux_std_info["marcs_template"].values
missing_template = [type(ff) != str for ff in marcs_templates]
marcs_templates[missing_template] = None
flux_std_info["marcs_template"] = marcs_templates

# -----------------------------------------------------------------------------
# Prepare telluric template
# -----------------------------------------------------------------------------
wave_tt_vac, trans_H2O_broad, trans_O2_broad, trans_telluric = \
    psm.read_and_broaden_telluric_transmission(
        telluric_template=ms.telluric_template_fits,
        resolving_power=ms.mike_resolving_power_r,
        do_convert_vac_to_air=True,
        wave_scale_new=None,)

# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
# Import MIKE spectra
wave, spec, sigma, orders, disp = pum.load_3D_spec_from_fits(
    path=ms.fits_folder, label=ms.fits_label, arm=ms.flux_cal_arm)
obs_info = pum.load_fits_table("OBS_TAB", ms.fits_label,)

# Grab dimensions for convenience
(n_star, n_order, n_px) = wave.shape

# [Optional] Clean spectra
if ms.flux_cal_clean_input_spectra:
    bad_px_mask = np.any(a=[spec <= 0, sigma <= 0,], axis=0,)

    if ms.flux_cal_edge_px_to_mask > 0:
        bad_px_mask[:,:,:ms.flux_cal_edge_px_to_mask] = np.nan
        bad_px_mask[:,:,-ms.flux_cal_edge_px_to_mask:] = np.nan

    spec[bad_px_mask] = np.nan
    sigma[bad_px_mask] = np.nan

    # Clean order #71 (bluest order) by trimming the edges.
    o71 = int(np.argwhere(orders == 71))
    o71_clip_mask = np.logical_or(wave[:,o71] < 4818, wave[:,o71] > 4880)

    for star_i in range(n_star):
        spec[star_i,o71,o71_clip_mask[star_i]] = np.nan

# [Optional] Normalise by flat fields
if ms.flux_cal_do_flat_field_blaze_corr:
    spec, sigma = \
        psm.normalise_mike_all_spectra_by_norm_flat_field(
            wave_3D=wave,
            spec_3D=spec,
            sigma_3D=sigma,
            orders=orders,
            arm="r",
            poly_order=ms.flux_cal_blaze_corr_poly_order,
            base_path=ms.flux_cal_blaze_corr_poly_coef_path,
            set_px_to_nan_beyond_domain=\
                ms.flux_cal_blaze_corr_set_px_to_nan_beyond_domain,)

# [Optional] Run on subset of orders for testing
if ms.flux_cal_run_on_order_subset:
    order_mask = np.isin(orders, ms.flux_cal_order_subset)

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
spphot_poly_coeff = np.full(
    (n_spphot, n_order, ms.flux_cal_poly_order), np.nan)
spphot_wave_mins = np.full((n_spphot, n_order,), np.nan)
spphot_wave_maxes = np.full((n_spphot, n_order,), np.nan)

# -----------------------------------------------------------------------------
# Running on all SpPhot targets
# -----------------------------------------------------------------------------
for si, (star_i, star_data) in enumerate(obs_info_sp.iterrows()):
    # Grab source ID
    source_id = star_data["source_id"]
    ut_date = star_data["ut_date"]

    # Grab row info from flux standard dataframe
    flux_std_row = flux_std_info.loc[source_id]

    print("-"*160,
          "{}/{} - Gaia DR3 {}".format(si+1, n_spphot, source_id),
          "-"*160, sep="\n")

    plot_label = "{}_{}".format(
        obs_info_sp.loc[star_i]["ut_date"], source_id)

    rv = star_data["rv_r"]
    bcor = star_data["bcor"]
    airmass = star_data["airmass"]

    # Grab initial guess tau scaling terms
    tau_scale_O2 = flux_std_row["O2_scl_{}".format(ut_date.replace("-", ""))]
    tau_scale_H2O = flux_std_row["H2O_scl_{}".format(ut_date.replace("-", ""))]

    # -------------------------------------------------------------------------
    # Flux standard spectrum
    # -------------------------------------------------------------------------
    # Import flux standard + convert to air wavelengths
    fs = np.loadtxt(flux_std_row["calspec_fn"])
    wave_fs_vac = fs[:,0]
    wave_fs_air = pum.convert_vacuum_to_air_wl(fs[:,0])

    # HACK (?) Only do relative flux calibration
    spec_fs = fs[:,1] / np.nanmedian(fs[:,1])

    cs_rv = flux_std_row["rv_adopted"]

    # Doppler shift
    wave_fs_ds_vac = wave_fs_vac * (1-(cs_rv-bcor)/(const.c.si.value/1000))
    wave_fs_ds_air = wave_fs_air * (1-(cs_rv-bcor)/(const.c.si.value/1000))

    # -------------------------------------------------------------------------
    # Synthetic template spectrum
    # -------------------------------------------------------------------------
    # HACK Make a dummy template if we don't have one--only used for plotting
    if flux_std_row["marcs_template"] is None:
        print("No stellar template")
        wave_synth = wave_fs_ds_air.copy()
        spec_synth = np.ones_like(wave_synth)
        
    else:
        ff = fits.open(flux_std_row["marcs_template"])
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
        do_convolution=ms.flux_cal_do_convolution,
        resolving_power_during_fit=ms.flux_cal_resolving_power_during_fit,
        sci_edge_handling=ms.flux_cal_sci_edge_handling,
        poly_order=ms.flux_cal_poly_order,
        optimise_order_overlap=ms.flux_cal_optimise_order_overlap,
        fit_for_telluric_scale_terms=ms.flux_cal_fit_for_telluric_scale_terms,)

    # Diagnostic plot
    ppltm.plot_flux_calibration(
        fit_dict, plot_folder=ms.flux_cal_plot_folder, plot_label=plot_label,)

    # Save coefficients to disk
    pum.save_flux_calibration_poly_coeff(
        poly_order=ms.flux_cal_poly_order,
        poly_coeff=fit_dict["poly_coef"],
        wave_mins=fit_dict["wave_mins"],
        wave_maxes=fit_dict["wave_maxes"],
        orders=orders,
        arm=ms.flux_cal_arm,
        label=plot_label,
        save_path=ms.flux_cal_plot_folder,)

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

# TODO combine coefficients with the blaze
pass

pum.save_flux_calibration_poly_coeff(
    poly_order=ms.flux_cal_poly_order,
    poly_coeff=poly_coef_mean,
    wave_mins=wave_mins_mean,
    wave_maxes=wave_maxes_mean,
    orders=orders,
    arm=ms.flux_cal_arm,
    label=fn_label,
    save_path=ms.flux_cal_plot_folder,)