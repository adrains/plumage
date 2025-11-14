"""Script to flux calibrate MIKE spectra.
"""
import numpy as np
import plumage.spectra_mike as psm
import plumage.utils_mike as pum
import plumage.plotting_mike as ppltm
from PyAstronomy.pyasl import instrBroadGaussFast
from astropy.io import fits
import astropy.constants as const
from scipy.interpolate import interp1d

# Update for line breaks
np.set_printoptions(linewidth=150)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
spphot_rvs = {
    "5709390701922940416":205.94,       # Gaia DR3
    "3510294882898890880":144.52,       # Gaia DR3 (+40 km/s needed)
    "22745910577134848":13.27,          # Gaia DR3
    "4201781696994073472":np.nan,       # NA?
    "5164707970261890560":16.60,        # 2023A&A...676A.129H
    "4376174445988280576":-397.263,     # Gaia DR3
    "5957698605530828032":100.840645,   # Gaia DR3
    "6342346358822630400":-19.51975,    # Gaia DR3
    "6477295296414847232":-44.77844,    # Gaia DR3
    "1779546757669063552":-14.7757635,  # Gaia DR3
}

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
poly_order = 5
optimise_order_overlap = True
do_convolution = True
resolving_power_during_fit = 2200

# Import MIKE spectra
wave, spec, sigma, orders = pum.load_3D_spec_from_fits(
    path=path, label=label, arm=arm)
obs_info = pum.load_fits_table("OBS_TAB", "KM",)

is_spphot = obs_info["is_spphot"]
wave_sp = wave[is_spphot]
spec_sp = spec[is_spphot]
sigma_sp = sigma[is_spphot]
obs_info_sp = obs_info[is_spphot]

n_spphot = np.sum(is_spphot)

# -----------------------------------------------------------------------------
# Running on all SpPhot targets
# -----------------------------------------------------------------------------
for si, (star_i, star_data) in enumerate(obs_info_sp.iterrows()):
    # Grab source ID
    source_id = star_data["source_id"]

    if source_id != "4376174445988280576":
        continue

    print("-"*160,
          "{}/{} - Gaia DR3 {}".format(si+1, n_spphot, source_id),
          "-"*160, sep="\n")

    # Skip if we don't have a template stellar spectrum
    if stellar_templates[source_id] is None:
        print("Skipping")
        continue

    plot_label = "{}_{}".format(
        obs_info_sp.loc[star_i]["ut_date"], source_id)

    rv = spphot_rvs[source_id]
    bcor = obs_info_sp.loc[star_i]["bcor"]
    airmass = obs_info_sp.loc[star_i]["airmass"]

    # ---------------------------
    # Import flux standard + convert to air wavelengths
    fs = np.loadtxt(calspec_templates[source_id])
    wave_fs = pum.convert_vacuum_to_air_wl(fs[:,0])
    spec_fs = fs[:,1] / np.nanmedian(fs[:,1])   # Relative flux calibration is fine

    # Doppler shift to match barycentric velocity
    interp_synth = interp1d(
        x=wave_fs,
        y=spec_fs,
        bounds_error=False,
        fill_value=1.0,
        kind="cubic",)

    # Doppler shift
    wave_fs_ds = wave_fs * (1-(-bcor)/(const.c.si.value/1000))
    spec_fs_ds = interp_synth(wave_fs_ds)

    # ---------------------------
    # Import stellar template
    ff = fits.open(stellar_templates[source_id])
    wave_synth = ff["WAVE"].data
    spec_synth = ff["SPEC"].data[0]

    interp_synth = interp1d(
        x=wave_synth,
        y=spec_synth,
        bounds_error=False,
        fill_value=1.0,
        kind="cubic",)

    # Doppler shift
    wave_synth_ds = wave_synth * (1-(rv-bcor)/(const.c.si.value/1000))
    spec_synth_ds = interp_synth(wave_synth_ds)

    # ---------------------------
    # MIKE spectrum
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
        wave_fluxed=wave_fs,
        spec_fluxed=spec_fs_ds,
        wave_synth=wave_synth,
        spec_synth=spec_synth_ds,
        airmass=airmass,
        do_convolution=do_convolution,
        resolving_power_during_fit=resolving_power_during_fit,
        max_line_depth=0.0,
        poly_order=poly_order,
        edge_px_to_mask=0,
        optimise_order_overlap=optimise_order_overlap,)

    # Diagnostic plot
    ppltm.plot_flux_calibration(
        fit_dict, plot_folder=plot_folder, plot_label=plot_label,)

    # Save coefficients to disk
    pum.save_flux_calibration_poly_coeff(
        poly_order=poly_order,
        poly_coeff=fit_dict["poly_coef"],
        orders=orders,
        arm=arm,
        label=plot_label,
        save_path=plot_folder,)

"""
# Flux calibrate arbitrary spectrum
test_i = 6

spec_fc, sigma_fc = psm.flux_calibrate_mike_spectrum(
    wave_2D=wave[test_i],
    spec_2D=spec[test_i],
    sigma_2D=sigma[test_i],
    airmass=obs_info.iloc[test_i]["airmass"],
    poly_order=poly_order,
    arm=arm,
    coeff_save_folder=plot_folder,
    label=plot_label,)
"""