"""Script to generate all paper figures and tables
"""
import plumage.plotting as pplt
import plumage.utils as utils
import plumage.spectra as spec
import plumage.paper as paper

spec_path = "spectra"

# Import literature info for both standards and TESS targets
std_info = utils.load_info_cat("data/std_info.tsv")
tic_info = utils.load_info_cat("data/tess_info.tsv")

# Load results tables for both standards and TESS targets
obs_std = utils.load_fits_obs_table("std", path="spectra")
obs_tess = utils.load_fits_obs_table("tess", path="spectra")

# Load spectra and results for standards
spec_std_b, spec_std_r, obs_std = utils.load_fits("std", path=spec_path)
bad_px_masks_std_b = utils.load_fits_image_hdu("bad_px", "std", arm="b")
bad_px_masks_std_r = utils.load_fits_image_hdu("bad_px", "std", arm="r")
synth_std_b = utils.load_fits_image_hdu("synth", "std", arm="b")
synth_std_r = utils.load_fits_image_hdu("synth", "std", arm="r")
synth_std_lit_b = utils.load_fits_image_hdu("synth_lit", "std", arm="b")
synth_std_lit_r = utils.load_fits_image_hdu("synth_lit", "std", arm="r")

# Load spectra and results for TESS targets
spec_tess_b, spec_tess_r, obs_tess = utils.load_fits("tess", path=spec_path)
bad_px_masks_tess_b = utils.load_fits_image_hdu("bad_px", "tess", arm="b")
bad_px_masks_tess_r = utils.load_fits_image_hdu("bad_px", "tess", arm="r")
synth_tess_b = utils.load_fits_image_hdu("synth", "tess", arm="b")
synth_tess_r = utils.load_fits_image_hdu("synth", "tess", arm="r")

# Import light curve fitting results
lc_results = utils.load_fits_table(
    "TRANSIT_FITS", 
    label="tess", 
    path="spectra")

"""
pplt.plot_all_synthetic_fits(
    spec_tess_r, 
    synth_tess_r, 
    bad_px_masks_tess_r,
    obs_tess,
    "tess",
    tess_info,
    spectra_b=spec_tess_b, 
    synth_spec_b=synth_tess_b, 
    bad_px_masks_b=bad_px_masks_tess_b,
    is_tess=True,
    use_2mass_id=False,
    spec_synth_lit_b=None,
    spec_synth_lit_r=None,)

pplt.plot_all_synthetic_fits(
    spec_std_r, 
    synth_std_r, 
    bad_px_masks_std_r,
    obs_std,
    "std",
    std_info,
    spectra_b=spec_std_b, 
    synth_spec_b=synth_std_b, 
    bad_px_masks_b=bad_px_masks_std_b,
    is_tess=False,
    use_2mass_id=False,
    spec_synth_lit_b=synth_std_lit_b,
    spec_synth_lit_r=synth_std_lit_r,)
"""

# -----------------------------------------------------------------------------
# Paper Plots
# -----------------------------------------------------------------------------
# TESS CMD
pplt.plot_cmd(tic_info, std_info[~np.isnan(std_info["mass_m19"])])

# Fluxes
#pplt.plot_fbol_comp(observations, tic_info, ncols=9)

# Radii comparison
pplt.plot_radius_comp(obs_std, std_info)

# HR Diagram
pplt.plot_hr_diagram(observations, tic_info,)

# Standard comparison (fit to lit, and fit teff vs phot teff)
teff_syst = -30

pplt.plot_std_comp(
    obs_std,
    std_info,
    show_offset=True,
    fn_suffix="_2_param",
    teff_syst=teff_syst,
    undo_teff_syst=True,)

pplt.plot_std_comp(
    obs_std_3_param,
    std_info,
    show_offset=True,
    fn_suffix="_3_param",
    teff_syst=teff_syst,
    undo_teff_syst=True,)

# TESS comparison (fit teff vs phot teff)

# Plot spectra summary (standard)

# Plot spectra summary (TESS)

# Plot light curve summary

# Plot planet results
pplt.plot_planet_radii_hist(
    toi_results,
    bin_width=0.35,
    plot_smooth_hist=False,
    x_lims=(0,14))

# -----------------------------------------------------------------------------
# Paper Tables
# -----------------------------------------------------------------------------
# TESS
paper.make_table_targets(tess_info=tess_info)
paper.make_table_final_results(
    label="tess",
    info_cat=tess_info,
    break_row=64,
    do_activity_crossmatch=True)
paper.make_table_planet_params(60)
paper.make_table_ld_coeff(info_cat=tess_info)
paper.make_table_planet_lit_comp()

# Standards
paper.make_table_targets()
paper.make_table_observations(obs_std, std_info, "std")
paper.make_table_final_results(
    label="std",
    info_cat=std_info,
    break_row=64,
    do_activity_crossmatch=False)   



# Cool stars figure
pplt.plot_representative_spectral_model_limitations(
    "3796072592206250624",
    label="std_dec",
    plot_size=(9,4.5),
    plot_suffix="_cs")