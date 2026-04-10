"""Script to compare a Cannon model trained on MIKE spectra, versus MARCS
synthetic spectra generated at benchmark stellar parameters.

Note that this script expects to be run on the MSO servers where Thomas 
Nordlander's MARCS grid can be queried.
"""
import os
import numpy as np
import plumage.utils as pu
import stannon.utils as su
import stannon.plotting as splt
import stannon.stannon as stannon
import plumage.spectra_mike as psm

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

#------------------------------------------------------------------------------
# Training sample
#------------------------------------------------------------------------------
# Import dataframe with benchmark parameters
obs_join = pu.load_fits_table(
    extension="CANNON_INFO",
    label=ms.cannon_fits_label,
    fn_base=ms.cannon_fits_fn_base,
    path=ms.cannon_fits_folder,)

#------------------------------------------------------------------------------
# Cannon model
#------------------------------------------------------------------------------
# Import model
model_name = "stannon_model_{}_{}_{}L_{}P_{}S_{}.pkl".format(
    ms.cannon_model_type,
    ms.cannon_model_name,
    ms.cannon_model_L,
    ms.cannon_model_P,
    ms.cannon_model_S,
     "_".join(ms.cannon_model_labels))

cannon_model_path = os.path.join("spectra", model_name)

sm = stannon.load_model(cannon_model_path)

#------------------------------------------------------------------------------
# MARCS vs Cannon spectra comparison plots
#------------------------------------------------------------------------------
# Note: we run get_lit_param_synth.py first to already have MARCS spectra.
# Since the MARCS grid doesn't have an abundance dimension, we can only make
# this plot for a 3 label model.

fits_ext_label = "{}_{}L_{}P_{}S".format(
    ms.cannon_model_name,
    ms.cannon_model_L,
    ms.cannon_model_P,
    ms.cannon_model_S,)

cannon_df = pu.load_fits_table(
    extension="CANNON_MODEL",
    label=ms.cannon_fits_label,
    path=ms.cannon_fits_folder,
    fn_base=ms.cannon_fits_fn_base,
    ext_label=fits_ext_label)

adopted_benchmark = cannon_df["adopted_benchmark"].values

wls = pu.load_fits_image_hdu(
    extension="rest_frame_wave",
    label=ms.cannon_fits_label,
    fn_base=ms.cannon_fits_fn_base,
    path=ms.cannon_fits_folder,
    arm="r",)

spec_marcs_br = pu.load_fits_image_hdu(
    "rest_frame_synth_lit",
    label=ms.cannon_fits_label,
    fn_base=ms.cannon_fits_fn_base,
    path=ms.cannon_fits_folder,
    arm="r",)

e_spec_marcs_br = np.ones_like(spec_marcs_br)

# TODO, HACK, replace all nan fluxes
for spec_i in range(len(spec_marcs_br)):
    if np.sum(np.isnan(spec_marcs_br[spec_i])) > 1000:
        spec_marcs_br[spec_i] = np.ones_like(spec_marcs_br[spec_i])

# Continuum normalise MIKE spectra
spec_2D_norm, sigma_2D_norm, continua_2D = \
    psm.pseudocontinuum_normalise_spectra(
        wls,
        spec_marcs_br,
        e_spec_marcs_br,
        resolving_power_smoothed=ms.pseudocontinuum_smoothing_resolution,)

# Grab MARCS spectra for just our benchmarks
wls, fluxes_norm, ivars_norm, bad_px_mask, adopted_wl_mask = \
        stannon.prepare_cannon_spectra_mike(
            wave=wls,
            spectra_2D=spec_2D_norm,
            sigmas_2D=sigma_2D_norm,
            wl_min_model=ms.cannon_fits_wl_min_model,
            wl_max_model=ms.cannon_fits_wl_max_model,
            telluric_trans_2D=np.ones_like(spec_marcs_br),
            telluric_absorption_threshold=0.95,
            allowable_NaN_telluric_px=5,)

# Plot Cannon vs MARCS spectra comparison over the entire spectral range
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[adopted_benchmark],
    fluxes=fluxes_norm[adopted_benchmark],
    bad_px_masks=bad_px_mask[adopted_benchmark],
    labels_all=sm.training_labels,
    source_ids=obs_join[adopted_benchmark].index.values,
    sort_col_name="BP_RP_dr3",
    x_lims=(ms.cannon_fits_wl_min_model,ms.cannon_fits_wl_max_model),
    fn_label="mike_marcs_all",
    data_label="",
    data_plot_label="MARCS",
    data_plot_colour="mediumblue",
    fig_size=(30,80),)

# Plot *difference* between these fluxes for all benchmarks
splt.plot_delta_cannon_vs_marcs(
    obs_join=obs_join[adopted_benchmark],
    sm=sm,
    fluxes_marcs_norm=fluxes_norm[adopted_benchmark],
    delta_thresholds=ms.cannon_flux_delta_thresholds,)