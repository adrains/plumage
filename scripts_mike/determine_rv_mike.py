"""Script to perform RV fits to MIKE spectra to a) correct for slight
wavelength scale offsets relative to the telluric frame, and b) determine the
systemic velocity of each star.

TODO: fully implement the blue arm
"""
import numpy as np
import plumage.spectra_mike as sm
import plumage.utils_mike as um
import plumage.utils as pu
import plumage.plotting_mike as pm
from astropy.io import fits
import astropy.constants as const
import stannon.utils as su

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

# -----------------------------------------------------------------------------
# Import MIKE spectra
# -----------------------------------------------------------------------------
# Blue arm
wave_b_3D, spec_b_3D, sigma_b_3D, orders_b, disp_b = um.load_3D_spec_from_fits(
    path=ms.fits_folder, label=ms.fits_label, arm="b")

# Red arm
wave_r_3D, spec_r_3D, sigma_r_3D, orders_r, disp_r = um.load_3D_spec_from_fits(
    path=ms.fits_folder, label=ms.fits_label, arm="r")

# Observational info
obs_info = um.load_fits_table(
    "OBS_TAB", path=ms.fits_folder, label=ms.fits_label,)

# Grab dimensions for convenience
(n_star, n_order, n_px) = wave_b_3D.shape

# Construct title/filenames date/obj pairs
dates = obs_info["ut_date"].values
obj_names = obs_info["object"].values
fit_labels = ["{}_{}".format(date, obj) for date, obj in zip(dates, obj_names)]

# And a second list of labels for the wavelength scale correction
fit_labels_tt = ["tt_{}".format(fl) for fl in fit_labels]

# -----------------------------------------------------------------------------
# Normalise science spectra by normalised flat field
# -----------------------------------------------------------------------------
# Normalise spectra by the normalised flat field transfer functions
# Blue
pass

# Red
spec_r_bc_3D, sigma_r_bc_3D = \
    sm.normalise_mike_all_spectra_by_norm_flat_field(
        wave_3D=wave_r_3D,
        spec_3D=spec_r_3D,
        sigma_3D=sigma_r_3D,
        orders=orders_r,
        arm="r",
        poly_order=ms.poly_order_for_rv_flat_norm,
        base_path=ms.poly_coef_for_rv_path,
        set_px_to_nan_beyond_domain=ms.set_px_to_nan_beyond_domain,)

# -----------------------------------------------------------------------------
# Additional normalisation by low-order polynomial
# -----------------------------------------------------------------------------
# Blue
pass

# Red
spec_r_rv_norm_3D, sigma_r_rv_norm_3D = \
    sm.normalise_all_mike_spectra_by_low_order_poly(
        wave_3D=wave_r_3D,
        spec_3D=spec_r_bc_3D,
        sigma_3D=sigma_r_bc_3D,
        poly_order=1)

# -----------------------------------------------------------------------------
# Telluric reference
# -----------------------------------------------------------------------------
# Import telluric tranmission spectrum convolved to science resolving power and
# converted from vacuum to air wavelengths.
wave_telluric_vac, _, _, trans_telluric = \
    sm.read_and_broaden_telluric_transmission(
        telluric_template=ms.telluric_template_fits,
        resolving_power=ms.mike_resolving_power_r,
        do_convert_vac_to_air=True,
        wave_scale_new=None,)

# Grab length for convenience later
n_px_tt = len(wave_telluric_vac)

# -----------------------------------------------------------------------------
# Import template spectra
# -----------------------------------------------------------------------------
# Import template TODO: move all this into a function
with fits.open(ms.rv_stellar_template_fn) as ref_fits:
    wave_ref = ref_fits["WAVE"].data
    spec_ref_2D = ref_fits["SPEC"].data
    teffs_ref = ref_fits["PARAMS"].data["teff"]

# Select templates
template_spec_selected = []

for star_i in range(n_star):
    teff = obs_info.loc[star_i]["teff_template"]
    template_i = int(np.argwhere(teffs_ref == teff))
    template_spec_selected.append(spec_ref_2D[template_i])

template_spec_selected = np.stack(template_spec_selected)

# -----------------------------------------------------------------------------
# Cross-correlation with *telluric* spectrum to correct wavelength scale
# -----------------------------------------------------------------------------
# There can be issues where the MIKE wavelength scale is not aligned to the
# absolute telluric frame, as determined by cross-correlating with a telluric
# template spectrum. Tests on 16/02/2026 showed sigma_rv = -0.09 +/- 0.76 km/s
# using the red arm for the sample of 78 F/G/K--K/M binaries, interferometric
# benchmarks, and flux standards. The following code aims to correct this
# (potential) offset for each star.

# Blue
pass

# Red
print("Cross-correlating for red arm wavelength scale...")
rvs_tt_r, all_rv_fit_dicts_tt = sm.fit_all_rvs(
    wave_3D=wave_r_3D,
    spec_3D=spec_r_rv_norm_3D,
    sigma_3D=sigma_r_rv_norm_3D,
    orders=orders_r,
    wave_telluric=wave_telluric_vac,
    trans_telluric=np.ones_like(wave_telluric_vac),     # No telluric masking
    wave_template=wave_telluric_vac,
    spec_template=np.broadcast_to(trans_telluric[None,:], (n_star, n_px_tt)),
    bcors=np.zeros(n_star),                             # No barycentric RV
    segment_contamination_threshold=0.0,                # No telluric masking
    px_absorption_threshold=0.0,                        # No telluric masking
    rv_min=ms.wavelength_scale_adjustment_cc_bounds[0],
    rv_max=ms.wavelength_scale_adjustment_cc_bounds[1],
    delta_rv=ms.wavelength_scale_adjustment_cc_delta,
    interpolation_method="cubic",)

pm.plot_all_cc_rv_diagnostics(
    all_rv_fit_dicts=all_rv_fit_dicts_tt,
    obj_names=fit_labels_tt,
    figsize=(16,4),
    fig_save_path=ms.rv_diagnostic_folder,
    run_in_wavelength_scale_debug_mode=True,)

# Duplicate original wavelength scale for safety
wave_r_3D_old = wave_r_3D.copy()

# Correct wavelength scale
# TODO: do this in a function
for star_i in range(n_star):
    rv = rvs_tt_r[star_i]
    wave_r_3D[star_i] =  wave_r_3D[star_i] * (1-rv/(const.c.si.value/1000))
    all_rv_fit_dicts_tt[star_i]['wave_2D'] = wave_r_3D[star_i]

# Save this back to fits
pu.save_fits_image_hdu(
    data=wave_r_3D,
    extension="wave_3D",
    fn_base="mike_spectra",
    label=ms.fits_label,
    path=ms.fits_folder,)

# -----------------------------------------------------------------------------
# Cross-correlation with *stellar* spectrum for systemic RV
# -----------------------------------------------------------------------------
# Blue
pass

# Red
print("Cross-correlating for red arm stellar systemic RVs...")
rvs_r, all_rv_fit_dicts = sm.fit_all_rvs(
    wave_3D=wave_r_3D,
    spec_3D=spec_r_rv_norm_3D,
    sigma_3D=sigma_r_rv_norm_3D,
    orders=orders_r,
    wave_telluric=wave_telluric_vac,
    trans_telluric=trans_telluric,
    wave_template=wave_ref,
    spec_template=template_spec_selected,
    bcors=obs_info["bcor"].values,
    segment_contamination_threshold=ms.segment_contamination_threshold,
    px_absorption_threshold=ms.px_absorption_threshold,
    rv_min=ms.stellar_systemic_rv_cc_bounds[0],
    rv_max=ms.stellar_systemic_rv_cc_bounds[1],
    delta_rv=ms.stellar_systemic_rv_cc_delta,
    interpolation_method="cubic",)

pm.plot_all_cc_rv_diagnostics(
    all_rv_fit_dicts=all_rv_fit_dicts,
    obj_names=fit_labels,
    figsize=(16,4),
    fig_save_path=ms.rv_diagnostic_folder,)

# -----------------------------------------------------------------------------
# Save RV back to DataFrame
# -----------------------------------------------------------------------------
# Blue
pass

# Red
obs_info["rv_r"] = rvs_r
obs_info["rv_wave_corr_r"] = rvs_tt_r

pu.save_fits_table(
    extension="OBS_TAB",
    dataframe=obs_info,
    fn_base="mike_spectra",
    label=ms.fits_label,
    path=ms.fits_folder,)