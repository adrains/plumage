"""Script to perform RV fits to MIKE spectra.
"""
import numpy as np
import plumage.spectra_mike as sm
import plumage.utils_mike as um
from astropy.io import fits
from PyAstronomy.pyasl import instrBroadGaussFast

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Details of observed fits file to import
path= "spectra"
label = "KM_noflat"

# Settings for polynomial 'blaze correction'
flat_norm_poly_order = 7
poly_coef_csv_path = "data/"
set_px_to_nan_beyond_domain = True

# Adjustment polynomial to put the 'blaze corrected' spectra to the 'continuum'
continuum_adust_poly_order = 1

# Telluric settings
telluric_template_fits = "data/viper_stdAtmos_vis.fits"
resolving_power = 46000
telluric_contamination_threshold = 0.1

# Stellar template
rv_template_fn = "templates/template_MIKE_R_44000_251121.fits"

run_on_order_subset = False
order_subset = [50, 51, 52, 53, 54]

# -----------------------------------------------------------------------------
# Import MIKE spectra
# -----------------------------------------------------------------------------
# Blue arm
wave_b_3D, spec_b_3D, sigma_b_3D, orders_b = um.load_3D_spec_from_fits(
    path=path, label=label, arm="b")

# Red arm
wave_r_3D, spec_r_3D, sigma_r_3D, orders_r = um.load_3D_spec_from_fits(
    path=path, label=label, arm="r")

# Observational info
obs_info = um.load_fits_table("OBS_TAB", label=label,)

# Grab dimensions for convenience
(n_star, n_order, n_px) = wave_b_3D.shape

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
        poly_order=flat_norm_poly_order,
        base_path=poly_coef_csv_path,
        set_px_to_nan_beyond_domain=set_px_to_nan_beyond_domain,)

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
with fits.open(telluric_template_fits) as tt:
    wave_telluric = tt[1].data["lambda"]
    trans_telluric = tt[1].data["H2O"] * tt[1].data["O2"]

trans_H2O_broad = instrBroadGaussFast(
    wvl=wave_telluric,
    flux=trans_telluric,
    resolution=resolving_power,
    edgeHandling="firstlast",
    maxsig=5,
    equid=True,)

# Convert telluric wavelength scale to air wavelengths to match MIKE
wave_telluric_vac = um.convert_vacuum_to_air_wl(wave_telluric)

# -----------------------------------------------------------------------------
# Import template spectra
# -----------------------------------------------------------------------------
# Import template HACK
with fits.open(rv_template_fn) as ref_fits:
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
# Calculate RVs
# -----------------------------------------------------------------------------
# Blue
pass

# Red
all_rv_fit_dicts = sm.fit_all_rvs(
    wave_3D=wave_r_3D,
    spec_3D=spec_r_rv_norm_3D,
    sigma_3D=sigma_r_rv_norm_3D,
    orders=orders_r,
    wave_telluric=wave_telluric_vac,
    trans_telluric=trans_telluric,
    wave_template=wave_ref,
    spec_template=template_spec_selected,
    bcors=obs_info["bcor"].values,
    segment_contamination_threshold=0.99,
    px_absorption_threshold=0.9,
    rv_min=-400,
    rv_max=400,
    delta_rv=1,
    do_diagnostic_plots=True,
    figsize=(16,4),
    fig_save_path="plots/rv_diagnostics",
    obj_names=obs_info["object"].values,
    interpolation_method="cubic",)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
# Save, but now with RV
#utils.save_fits_table("OBS_TAB", observations, label, path=spec_path)
#utils.save_fits_image_hdu(bad_px_masks_b, "bad_px", label, arm="b")
#utils.save_fits_image_hdu(bad_px_masks_r, "bad_px", label, arm="r")