"""Script to compare performance across different Cannon models.
"""
import numpy as np
import plumage.utils as pu
import stannon.utils as su
import stannon.plotting as splt
import stannon.stannon as stannon
import stannon.tables as st

#------------------------------------------------------------------------------
# Import Cannon settings file
#------------------------------------------------------------------------------
cannon_settings_yaml = "scripts_cannon/cannon_settings.yml"
cs = su.load_cannon_settings(cannon_settings_yaml)

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
# Load in Cannon models
sm1 = stannon.load_model(cs.sm_1_path)
sm2 = stannon.load_model(cs.sm_2_path)

#------------------------------------------------------------------------------
# MARCS vs Cannon spectra comparison plots
#------------------------------------------------------------------------------
# Note: we run get_lit_param_synth.py first to already have MARCS spectra.
# Since the MARCS grid doesn't have an abundance dimension, we can only make
# this plot for a 3 label model.
obs_join = pu.load_fits_table("CANNON_INFO", "cannon")
is_cannon_benchmark = obs_join["is_cannon_benchmark"].values

wls = pu.load_fits_image_hdu("rest_frame_wave", cs.std_label, arm="br")
spec_marcs_br = \
    pu.load_fits_image_hdu("rest_frame_synth_lit", cs.std_label, arm="br")
e_spec_marcs_br = np.ones_like(spec_marcs_br)

# TODO, HACK, replace all nan fluxes
for spec_i in range(len(spec_marcs_br)):
    if np.sum(np.isnan(spec_marcs_br[spec_i])) > 1000:
        spec_marcs_br[spec_i] = np.ones_like(spec_marcs_br[spec_i])

# Grab MARCS spectra for just our benchmarks
fluxes_marcs_norm, _, bad_px_mask, _, _ = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wls,
        spectra=spec_marcs_br[is_cannon_benchmark],
        e_spectra=e_spec_marcs_br[is_cannon_benchmark],
        wl_min_model=cs.wl_min_model,
        wl_max_model=cs.wl_max_model,
        wl_min_normalisation=cs.wl_min_normalisation,
        wl_broadening=cs.wl_broadening,
        do_gaussian_spectra_normalisation=\
            cs.do_gaussian_spectra_normalisation,
        poly_order=cs.poly_order)

# Plot Cannon vs MARCS spectra comparison over the entire spectral range
splt.plot_spectra_comparison(
    sm=sm1,
    obs_join=obs_join[is_cannon_benchmark],
    fluxes=fluxes_marcs_norm,
    bad_px_masks=bad_px_mask,
    labels_all=sm1.training_labels,
    source_ids=cs.representative_stars_source_ids,
    sort_col_name="BP_RP_dr3",
    x_lims=(cs.wl_min_model,cs.wl_max_model),
    fn_label="marcs",
    data_label="",
    data_plot_label="MARCS",
    data_plot_colour="b",)

# Plot *difference* between these fluxes for all benchmarks
splt.plot_delta_cannon_vs_marcs(
    obs_join=obs_join,
    sm=sm1,
    fluxes_marcs_norm=fluxes_marcs_norm,
    delta_thresholds=cs.delta_thresholds,)

#------------------------------------------------------------------------------
# Theta plots
#------------------------------------------------------------------------------
# Print scatter std
print("Scatter:", "\n--------")
print("{} Label:  {:1.7f}±{:1.7f}".format(
    sm1.L, np.mean(sm1.s2**0.5), np.std(sm1.s2**0.5)))
print("{} Label:  {:1.7f}±{:1.7f}".format(
    sm2.L, np.mean(sm2.s2**0.5), np.std(sm2.s2**0.5)))

# Scatter comparison between both models
splt.plot_scatter_histogram_comparison(
    sm1=sm1,
    sm2=sm2,
    sm_1_label=cs.sm_1_label,
    sm_2_label=cs.sm_2_label,
    n_bins=250,
    hist_bin_lims=(0, 0.0005),)

# Label
fn_suffix = "_comparison"

# Import line lists
line_list_b = pu.load_linelist(
    filename=cs.line_list_file,
    wl_lower=cs.wl_min_model,
    wl_upper=cs.wl_grating_changeover,
    ew_min_ma=cs.ew_min_ma_b,)

line_list_r = pu.load_linelist(
    filename=cs.line_list_file,
    wl_lower=cs.wl_grating_changeover,
    wl_upper=cs.wl_max_model,
    ew_min_ma=cs.ew_min_ma_r,)

# Plot theta coefficients for each WiFeS arm
splt.plot_theta_coefficients(
    sm=sm2,
    teff_scale=1.0,
    x_lims=(cs.wl_min_model,cs.wl_grating_changeover),
    y_spec_lims=(0,2.25),
    y_theta_linear_lims=(-0.12,0.12),
    y_theta_quadratic_lims=(-0.2,0.2),
    y_theta_cross_lims=(-0.3,0.3),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    fn_label="b",
    linewidth=0.5,
    alpha=0.8,
    fn_suffix=fn_suffix,
    line_list=line_list_b,
    species_to_plot=cs.species_to_plot,
    only_plot_first_order_coeff=False,
    sm2=sm1,
    sm1_label="4 Label",
    sm2_label="3 Label",)


splt.plot_theta_coefficients(
    sm=sm2,
    teff_scale=1.0,
    x_lims=(cs.wl_grating_changeover,cs.wl_max_model),
    y_spec_lims=(0,2.25),
    y_theta_linear_lims=(-0.12,0.12),
    y_theta_quadratic_lims=(-0.1,0.1),
    y_theta_cross_lims=(-0.2,0.2),
    y_s2_lims=(-0.0001, 0.005),
    x_ticks=(200,100),
    fn_label="r",
    linewidth=0.5,
    alpha=0.8,
    fn_suffix=fn_suffix,
    line_list=line_list_r,
    species_to_plot=cs.species_to_plot,
    only_plot_first_order_coeff=False,
    sm2=sm1,
    sm1_label="4 Label",
    sm2_label="3 Label",)

#------------------------------------------------------------------------------
# Tables
#------------------------------------------------------------------------------
# Comparison table of theta standard deviations
st.make_theta_comparison_table(
    sms=[sm1, sm2],
    sm_labels=[cs.sm_1_label, cs.sm_2_label],
    print_table=True)