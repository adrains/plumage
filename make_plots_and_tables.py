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
pplt.plot_cmd(tic_info)

# Fluxes
pplt.plot_fbol_comp(observations, tic_info, ncols=9)

# Radii comparison
pplt.plot_radius_comp(observations, tic_info)

# HR Diagram
pplt.plot_hr_diagram(observations, tic_info,)

# Standard comparison (fit to lit, and fit teff vs phot teff)
pplt.plot_std_comp(obs_std, std_info, show_median_offset=True)
pplt.plot_teff_comp(
    obs_std, 
    std_info, 
    x_col="teff_synth", 
    phot_teff_col="teff_m15_bprp_jh")

# TESS comparison (fit teff vs phot teff)
pplt.plot_teff_comp(
    observations, 
    std_info, 
    x_col="teff_synth", 
    phot_teff_col="teff_m15_bprp_jh")

# Plot spectra summary (standard)

# Plot spectra summary (TESS)

# Plot light curve summary

# Plot planet results
pplt.plot_planet_radii_hist(lc_results, bin_width=0.4)
pplt.plot_planet_period_vs_radius(lc_results)

# -----------------------------------------------------------------------------
# Paper Tables
# -----------------------------------------------------------------------------
paper.make_table_targets()
paper.make_table_final_results(label="tess",info_cat=tess_info, break_row=64, do_activity_crossmatch=True)
paper.make_table_final_results()
paper.make_table_planet_params()
#paper.make_table_fbol()
#paper.make_table_ldc()
#paper.make_table_lc_fit_params()

# Standards
paper.make_table_targets()
paper.make_table_observations(obs_std, std_info, "std")
paper.make_table_final_results(label="std",info_cat=std_info, break_row=64, do_activity_crossmatch=False)   

"""
pplt.plot_teff_comp(
    observations,
    std_info,
    x_col="teff_synth",
    phot_teff_col="teff_m15_bprp",
    fn_suffix="_phot_x{}".format(scale_fac),
    title_text=scale_fac,)

pplt.plot_std_comp(
    observations,
    std_info,
    show_median_offset=True,
    fn_suffix="_phot_x{}".format(scale_fac),
    title_text=scale_fac,)
"""