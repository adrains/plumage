"""Script to flux calibrate MIKE spectra.
"""
import numpy as np
import plumage.spectra_mike as psm
import plumage.utils_mike as pum
from PyAstronomy.pyasl import instrBroadGaussFast
from astropy.io import fits
import astropy.constants as const
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
spphot_rvs = {
    "5709390701922940416":205.94,       # Gaia DR3
    "3510294882898890880":144.52 + 40,  # Gaia DR3 TODO: figure out why this RV is bad 
    "22745910577134848":13.27,          # Gaia DR3
}

stellar_templates = {
    "5709390701922940416":"templates/template_HD74000_250624_norm.fits",  # 6211, 4.155, -1.98
    "3510294882898890880":"templates/template_HD111980_250624_norm.fits", # 5578, 3.89, -1.08
    "22745910577134848":None,}                                            # B star, no MARCS

# https://github.com/PyWiFeS/pywifes/tree/main/reference_data
calspec_templates = {
    "5709390701922940416":"data/flux_standards/hd074000_stis_007.txt",
    "3510294882898890880":"data/flux_standards/hd111980_stis_007.txt",
    "22745910577134848":"data/flux_standards/ksi2ceti_stis_008.txt",}

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
# Import
# -----------------------------------------------------------------------------
path= "spectra"
arm = "r"
label = "KM"

source_id = "3510294882898890880"

# Import MIKE spectra
wave, spec, sigma, orders = pum.load_3D_spec_from_fits(path=path, label=label, arm=arm)
obs_info = pum.load_fits_table("OBS_TAB", "KM",)

is_spphot = obs_info["kind"] == "SpPhot"
wave_sp = wave[is_spphot]
spec_sp = spec[is_spphot]
sigma_sp = sigma[is_spphot]
obs_info_sp = obs_info[is_spphot]

sp_i = 1

rv = spphot_rvs[source_id]
bcor = obs_info_sp.iloc[sp_i]["bcor"]

# Import flux standard
fs = np.loadtxt(calspec_templates[source_id])
wave_fs = fs[:,0]
spec_fs = fs[:,1] / np.nanmedian(fs[:,1])   # Relative flux calibration is fine

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

# Select spectrum and normalise to median, since we don't care to attempt
# *absolute* flux calibration.
med = np.nanmedian(spec_sp[sp_i][1:])

wave_obs_2D = wave_sp[sp_i][1:]
spec_obs_2D = spec_sp[sp_i][1:] / med
sigma_obs_2D = sigma[sp_i][1:] / med

# -----------------------------------------------------------------------------
# Flux calibration
# -----------------------------------------------------------------------------
fit_dict = psm.fit_atmospheric_transmission(
    wave_obs_2D=wave_obs_2D,
    spec_obs_2D=spec_obs_2D,
    sigma_obs_2D=sigma_obs_2D, 
    wave_telluric=wave_tt_vac,
    trans_H2O_telluric=trans_H2O_broad,
    trans_O2_telluric=trans_O2_broad,
    wave_fluxed=wave_fs,
    spec_fluxed=spec_fs,
    wave_synth=wave_synth,
    spec_synth=spec_synth_ds,
    max_line_depth=0.9,
    poly_order=5,
    edge_px_to_mask=20,)
