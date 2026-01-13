"""Script to investigate best 'fit' (by eye) telluric optical depth scale terms
for H2O and O2 to use when flux calibrating flux standards.
"""
import numpy as np
import plumage.spectra_mike as psm
import plumage.utils_mike as pum
import plumage.plotting_mike as ppltm
from PyAstronomy.pyasl import instrBroadGaussFast
from astropy.io import fits

# Telluric transmission
# https://github.com/mzechmeister/viper/tree/master/lib/atmos
telluric_template = "data/viper_stdAtmos_vis.fits"
resolving_power = 46000

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

    med = np.nanmedian(spec_sp[si][1:])

    wave_obs_2D = wave_sp[si]
    spec_obs_2D = spec_sp[si] / med
    sigma_obs_2D = sigma[si] / med

    plot_label = "{}_{}".format(source_id, ut_date)

    test_O2_scale = np.arange(0.5,2.0,0.1)

    ppltm.plot_telluric_scale_terms(
        wave_2D=wave_obs_2D,
        spec_2D=spec_obs_2D,
        wave_tt=wave_tt_vac,
        trans_tt=trans_O2_broad,
        test_tt_scale=test_O2_scale,
        plot_label=plot_label,
        species="O2",
        wave_min=7590,
        wave_max=7700,)
    
    test_H2O_scale = np.arange(0.5,2.0,0.1)

    ppltm.plot_telluric_scale_terms(
        wave_2D=wave_obs_2D,
        spec_2D=spec_obs_2D,
        wave_tt=wave_tt_vac,
        trans_tt=trans_H2O_broad,
        test_tt_scale=test_H2O_scale,
        plot_label=plot_label,
        species="H2O",
        wave_min=8125,
        wave_max=8250,)
    
    print('"{}", "{}"'.format(source_id, ut_date))