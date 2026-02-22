"""Script to prepare reduced MIKE spectra for further analysis with the Cannon.

Things this script does
    1) Flux calibrate MIKE spectra (+save)
    2) Combine MIKE orders onto uniform wavelength scale (+save)
        - This includes an RV correction
        - TODO: also, saving RV shifted tellurics as bad px mask?
    3) [Optional] Smoothing to constant resolution (+save)
    4) Continuum normalise MIKE spectra (+save)

Standard Process
----------------
This script is 5) in the following series of scripts to work with MIKE data:
    1) scripts_reduction/import_spectra_mike.py
    2) scripts_reduction/determine_rv_mike.py
    3) scripts_reduction/fit_tau_scale_components_for_flux_standards.py
    4) scripts_reduction/compute_flux_cal_mike.py
    5) scripts_mike/prepare_mike_spectra.py

Then we move onto standard Cannon scripts:
    1) scripts_cannon/assess_literature_systematics.py 
    2) scripts_cannon/prepare_stannon_training_sample.py
    3) scripts_cannon/train_stannon.py
    4) scripts_cannon/make_stannon_diagnostics.py

And synthetic spectra comparison (to be run on MSO servers):
    1) scripts_synth/get_lit_param_synth.py

Related MIKE files
    - scripts_mike/mike_reduction_settings.yml  TODO!
    - data/mike_info.tsv
"""
import numpy as np
import plumage.spectra_mike as sm
import plumage.utils_mike as um
import plumage.plotting_mike as ppltm
import stannon.utils as su

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
wave, spec, sigma, orders, disp = um.load_3D_spec_from_fits(
    path=ms.fits_folder, label=ms.fits_label, arm=ms.spec_prep_arm)
obs_info = um.load_fits_table("OBS_TAB", "KM_noflat",)

(n_obs, n_order, n_px) = wave.shape

# Normalise by the fitted 'blaze' (i.e. the lamp spectrum).TODO: these
# coefficients should be added to the flux calibration ones for simplicity.
if ms.spec_prep_do_flat_field_blaze_corr:
    spec, sigma = \
        sm.normalise_mike_all_spectra_by_norm_flat_field(
            wave_3D=wave,
            spec_3D=spec,
            sigma_3D=sigma,
            orders=orders,
            arm=ms.spec_prep_arm,
            poly_order=ms.spec_prep_blaze_corr_poly_order,
            base_path=ms.spec_prep_blaze_corr_poly_coef_path,
            set_px_to_nan_beyond_domain=\
                ms.spec_prep_blaze_corr_set_px_to_nan_beyond_domain,)

# -----------------------------------------------------------------------------
# Flux calibrating everything
# -----------------------------------------------------------------------------
# Flux calibrate all spectra
spectra_fc = np.full_like(spec, np.nan)
sigma_fc = np.full_like(spec, np.nan)

for star_i in range(n_obs):
    spec_fc, sig_fc = sm.flux_calibrate_mike_spectrum(
        wave_2D=wave[star_i],
        spec_2D=spec[star_i],
        sigma_2D=sigma[star_i],
        airmass=obs_info.iloc[star_i]["airmass"],
        poly_order=ms.spec_prep_flux_cal_poly_order,
        arm=ms.spec_prep_arm,
        coeff_save_folder=ms.spec_prep_flux_cal_folder,
        label=ms.spec_prep_flux_cal_poly_order_fn_label,)

    spectra_fc[star_i] = spec_fc
    sigma_fc[star_i] = sig_fc

mm = np.argsort(obs_info["teff_template"].values)

ppltm.plot_all_flux_calibrated_spectra(
    wave_3D=wave[mm],
    spec_3D=spectra_fc[mm],
    sigma_3D=sigma_fc[mm],
    object_ids=obs_info["object"].values[mm],
    is_spphot_1D=obs_info["is_spphot"].values[mm],
    figsize=(16,50),
    plot_folder=ms.spec_prep_flux_cal_folder,
    plot_label=ms.spec_prep_arm,)

# -----------------------------------------------------------------------------
# Combine MIKE orders
# -----------------------------------------------------------------------------
# [Optional] Drop orders known to have problems
if ms.spec_prep_do_drop_orders:
    dropped_orders_mask = np.isin(orders, ms.spec_prep_orders_to_drop)

else:
    dropped_orders_mask = np.full(orders.shape, False)

# Combine MIKE spectral orders (+interpolation onto uniform wavelength scale)
wave_1D, spec_2D, sigma_2D = sm.combine_echelle_orders_for_all_observations(
    wave_3D=wave[:,~dropped_orders_mask],
    spec_3D=spectra_fc[:,~dropped_orders_mask],
    sigma_3D=sigma_fc[:,~dropped_orders_mask],
    disp_2D=disp[:,~dropped_orders_mask],)

# [Optional] Smooth to constant spectral resolution, resample wavelength scale
pass

# -----------------------------------------------------------------------------
# Pseudocontinuum Normalisation
# -----------------------------------------------------------------------------
# Continuum normalise MIKE spectra
spec_2D_norm, sigma_2D_norm, continua_2D = \
    sm.pseudocontinuum_normalise_spectra(
        wave_1D,
        spec_2D,
        sigma_2D,
        resolving_power_smoothed=ms.pseudocontinuum_smoothing_resolution,)

# Plot continuum normalisation diagnostics
ppltm.plot_pseudocontinuum_normalisation_diagnostics(
    obs_info=obs_info,
    wave_1D=wave_1D,
    spec_2D=spec_2D,
    spec_2D_norm=spec_2D_norm,
    sigma_2D_norm=sigma_2D_norm,
    continua_2D=continua_2D,
    arm=ms.spec_prep_arm,
    plot_folder=ms.plot_folder_cn)

# -----------------------------------------------------------------------------
# RV shift to restframe
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Wrapping up
# -----------------------------------------------------------------------------
# Saving back to file, etc
pass