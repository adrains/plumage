"""Script to prepare the training dataset for use when training a Cannon model.
The resulting training labels are saved to our fits file. Note that benchmarks
prepared in this way represent *potential* benchmarks that might be used as
part of the training sample for a given Cannon model--we need not use all
possible benchmarks.

This script is part of a series of Cannon scripts. The main sequence is:
 1) assess_literature_systematics.py       --> benchmark chemistry compilation
 2) prepare_stannon_training_sample.py     --> label preparation
 3) train_stannon.py                       --> training and cross validation
 4) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 5) run_stannon.py                         --> running on science spectra
"""
import pickle
import numpy as np
import pandas as pd
import plumage.utils as pu
import plumage.parameters as pp
import stannon.utils as su
import stannon.parameters as params
import stannon.plotting as splt
import radius_estimation.radius_estimation as re

#------------------------------------------------------------------------------
# Import Settings
#------------------------------------------------------------------------------
label_settings = "scripts_cannon/label_settings.yml"
ls = su.load_yaml_settings(label_settings)

#------------------------------------------------------------------------------
# Imports and crossmatch catalogues
#------------------------------------------------------------------------------
def drop_cols(df, suffixes_to_drop,):
    """Function to drop unnecessary columns based on suffix. This is used to
    drop previous literature chemistry crossmatches that we now no longer
    need due to the standardised approach made available by having a single
    DataFrame of systematic-corrected literature chemistry."""
    columns = df.columns.values
    col_keep_mask = np.full_like(columns, True).astype(bool)

    for col_i, col in enumerate(df.columns.values):
        if np.any([suffix in col for suffix in suffixes_to_drop]):
            col_keep_mask[col_i] = False

    df.drop(columns=columns[~col_keep_mask], inplace=True)

#=========================================
# Crossmatch observed sample to literature DF
#=========================================
# Import literature info
#  - Note that since we don't use any photometry with the Cannon, and since our
#  stellar parameters come from our results table, we can set a bunch of the
#  extra import parameters to False to speed things up.
std_info = pu.load_info_cat(
    path=ls.std_info_fn,
    use_mann_code_for_masses=ls.use_mann_code_for_masses,
    only_import_in_paper=ls.only_import_in_paper,
    only_import_observed=ls.only_import_observed,
    gdr=ls.gdr,
    do_use_mann_15_JHK=True,)

std_info.reset_index(inplace=True)

# Load results table, update index name to DR2
obs_std = pu.load_fits_table(
    extension="OBS_TAB",
    label=ls.fits_label,
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,)

# To crossmatch with our fitted parameters from Rains+21 we need to do a
# crossmatch on DR2 coordinates instead for legacy reasons (as these were the
# IDs used when observing). Once we've done the crossmatch, we want to update
# the index be the Gaia DR3 source ID instead.
if ls.do_support_legacy_DR2_source_IDs:
    std_info.set_index("source_id_dr2", inplace=True)
    obs_std.index.rename("source_id_dr2", inplace=True)

    obs_join = obs_std.join(std_info, "source_id_dr2", rsuffix="_info")
    obs_join.set_index("source_id_dr3", inplace=True)

# If not, we can assume the targets are indexed via Gaia DR3 IDs which
# simplifies things.
else:
    std_info.set_index("source_id_dr3", inplace=True)
    obs_std.index.rename("source_id_dr3", inplace=True) # Rename if not already

    obs_join = obs_std.join(std_info, "source_id_dr3", rsuffix="_info")

# Drop unnecessary columns
obs_suffixes_to_drop = [
    "g14", "m15", "ra12", "vf05", "s08", "L18", "b16", "rb20", "m18", "a12"]

drop_cols(obs_join, obs_suffixes_to_drop)

#=========================================
# Crossmatch to corrected chemistry DF
#=========================================
lit_chem_df = pd.read_csv(
    ls.lit_chem,
    sep="\t",
    dtype={"source_id_dr3":str},)
lit_chem_df.set_index("source_id_dr3", inplace=True)

obs_join = obs_join.join(lit_chem_df, "source_id_dr3", rsuffix="_chem")

#=========================================
# F/G/K - K/M Binaries
#=========================================
# Import our two tables of CPM information
tsv_primaries = "data/cpm_primaries_dr3.tsv"
tsv_secondaries = "data/cpm_secondaries_dr3.tsv"

cpm_prim = pu.load_info_cat(
    path=tsv_primaries,
    make_observed_col_bool_on_yes=False,
    use_mann_code_for_masses=False,
    gdr="dr3",
    has_2mass=False,)

cpm_sec = pu.load_info_cat(
    path=tsv_secondaries,
    make_observed_col_bool_on_yes=False,
    use_mann_code_for_masses=False,
    gdr="dr3",
    has_2mass=True,)

# Remove secondaries without DR3 IDs as the NaN IDs mess with the crossmatch
cpm_sec = cpm_sec[~np.isnan(cpm_sec["ra_dr3"].values)].copy()

# Make sure indices are handled properly
cpm_prim.reset_index(inplace=True)
cpm_prim.rename(columns={"source_id_dr3":"source_id_dr3_prim"}, inplace=True)
cpm_prim.set_index("prim_name", inplace=True)

# Append '_prim' to all column names as to distinguish primary star information
# for benchmarks in binaries, from K dwarf benchmarks with values from the same
# literature sources.
cols_orig = cpm_prim.columns.values
cols_new = ["{}_prim".format(col) for col in cols_orig]
cols_dict = dict(zip(cols_orig, cols_new))
cpm_prim.rename(columns=cols_dict, inplace=True)

# Now merge prim and sec CPM info on prim_name
cpm_join = cpm_sec.join(cpm_prim, "prim_name", rsuffix="_prim")
cpm_join.rename(
    columns={"source_id_dr3_prim_prim":"source_id_dr3_prim"}, inplace=True)

# Drop unnecessary columns
cpm_suffixes_to_drop = [
    "n14", "m13", "m14", "other", "vf05", "s08", "L18", "b16", "rb20", "m18",
    "a12",]

drop_cols(cpm_join, cpm_suffixes_to_drop)

# Merge CPM table with literature chemistry table, but drop M-dwarf specific
# chemistry information as these aren't relevant to F/G/K primaries.
lit_chem_suffixes_to_drop = ["RA12", "RA12", "M15"]
drop_cols(lit_chem_df, lit_chem_suffixes_to_drop)
lit_chem_df.reset_index(inplace=True)
cols = lit_chem_df.columns.values

new_cols = ["{}_prim".format(col) for col in cols]
lit_chem_df.rename(columns=dict(zip(cols, new_cols)), inplace=True)
lit_chem_df.set_index("source_id_dr3_prim", inplace=True)
cpm_join = cpm_join.join(lit_chem_df, "source_id_dr3_prim",)

# Finally, merge the CPM info with obs_join to have everything in one dataframe
obs_join = obs_join.join(cpm_join, "source_id_dr3", rsuffix="_sec").copy()

# And make sure our CPM column is correct
is_cpm = np.array([type(val) == str for val in obs_join["prim_name"].values])
obs_join["is_cpm"] = is_cpm

#=========================================
# General cleanup
#=========================================
# Add a column for K dwarf benchmarks.
# TODO: either use ABUND_ORDER_K from YAML file, or perpendicular cut to main
# sequence in BP-RP and M_Ks.
has_mid_k_reference = np.any([
        ~np.isnan(obs_join["Fe_H_VF05"].values),
        ~np.isnan(obs_join["Fe_H_A12"].values),
        ~np.isnan(obs_join["Fe_H_B16"].values),
        ~np.isnan(obs_join["Fe_H_M18"].values),
        ~np.isnan(obs_join["Fe_H_L18"].values),
        ~np.isnan(obs_join["Fe_H_RB20"].values),],
    axis=0).astype(bool)

# Enforce our BP-RP bounds for what we consider a mid-K dwarf (i.e. where we
# trust the models for direct [Fe/H] determination from high-R spectroscopy)
is_mid_k_dwarf = np.logical_and(
    has_mid_k_reference,
    obs_join["BP-RP_dr3"].values < ls.mid_K_BP_RP_bound)

obs_join["is_mid_k_dwarf"] = is_mid_k_dwarf

# Load in and crossmatch with sampled params
sp_df = pd.read_csv(ls.sampled_param_csv, dtype={"source_id_dr3":str},)
sp_df.set_index("source_id_dr3", inplace=True)
obs_join = obs_join.join(sp_df, "source_id_dr3")

# Finally clean this and remove duplicate columns
cols = obs_join.columns.values
mm = np.array(["_sec" in cv for cv in cols])
obs_join = obs_join[cols[~mm]].copy()

# Grab n_star for convenience
N_STAR_ALL = len(obs_join)

#------------------------------------------------------------------------------
# Collate [Fe/H]
#------------------------------------------------------------------------------
# Select adopted [Fe/H] values which are needed for empirical Teff relations
feh_info_all = []

for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()): 
    feh_info = params.select_abund_label(
        star_info=star_info,
        abund="Fe_H",
        mid_K_BP_RP_bound=ls.mid_K_BP_RP_bound,
        mid_K_MKs_bound=ls.mid_K_MKs_bound,
        abund_order_k=ls.ABUND_ORDER_K,
        abund_order_m=ls.ABUND_ORDER_M,
        abund_order_binary=ls.ABUND_ORDER_BINARY,
        mid_K_BP_RP_trustworthy_X_Fe=ls.mid_K_BP_RP_trustworthy_X_Fe,)
    feh_info_all.append(feh_info)

feh_info_all = np.vstack(feh_info_all)

feh_values = feh_info_all[:,0].astype(float)

# TEMPORARY archive of chosen [Fe/H]
obs_join["feh_temp"] = feh_values.copy()

#------------------------------------------------------------------------------
# Calculate R* and logg via empirical relations
#------------------------------------------------------------------------------
# ---------------------------------------------
# Mann+2015 -- metal rich M-dwarfs
# ---------------------------------------------
r_star_m15, e_r_star_m15 = pp.compute_mann_2015_radii(
    k_mag_abs=obs_join["K_mag_abs"].values,
    fehs=feh_values.copy(),
    enforce_M_Ks_bounds=True,
    enforce_feh_bounds=True,)

obs_join["radius_m15"] = r_star_m15
obs_join["e_radius_m15"] = e_r_star_m15

# ---------------------------------------------
# Kesseli+2019 -- metal poor M-dwarfs
# ---------------------------------------------
r_star_k19, e_r_star_k19 = pp.compute_kesseli_2019_radii(
    k_mag_abs=obs_join["K_mag_abs"].values,
    fehs=feh_values.copy(),
    enforce_M_Ks_bounds=True,
    enforce_feh_bounds=True,)

obs_join["radius_k19"] = r_star_k19
obs_join["e_radius_k19"] = e_r_star_k19

# ---------------------------------------------
# Kiman+2024 -- K dwarfs
# ---------------------------------------------
G_RP = obs_join["G_mag_dr3"].values - obs_join["RP_mag_dr3"].values
e_G_RP = (obs_join["e_G_mag_dr3"].values**2
          + obs_join["e_RP_mag_dr3"].values**2)**0.5

e_BP_RP = (obs_join["e_BP_mag_dr3"].values**2
           + obs_join["e_RP_mag_dr3"].values**2)**0.5

r_star_k24, e_low_r_star_k24, e_high_r_star_k24 = re.calc_radius(
    flux_col="Fg",
    color_col="bp_rp",
    parallax=obs_join["plx_dr3"].values,
    e_parallax=obs_join["e_plx_dr3"].values,
    color=obs_join["BP-RP_dr3"].values,
    e_color=e_BP_RP,
    mag=obs_join["G_mag_dr3"].values,
    e_mag=obs_join["e_G_mag_dr3"].values,
    feh=obs_join["feh_temp"].values.copy(),)

# HACK Simply adopt the maximum uncertainty from the low/high uncertainties
e_r_star_k24 = np.max(np.stack([e_low_r_star_k24, e_high_r_star_k24]), axis=0)

obs_join["radius_k24"] = r_star_k24
obs_join["e_radius_k24"] = e_r_star_k24

# ---------------------------------------------
# Adopting radii
# ---------------------------------------------
# Initialise radii, e_radii, and reference arrays. Note that these will end up
# remaining undefined (NaN, NaN, or "") for cases where the star is beyond the
# bounds of the relation in magnitude, colour, or [Fe/H].
r_star_adopt = np.full_like(r_star_m15, np.nan)
e_r_star_adopt = np.full_like(r_star_m15, np.nan)
radius_adopted_ref = np.full(N_STAR_ALL, "").astype(object)

# Loop over all stars
for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
    # Adopt Mann+2015 for (BP-RP) >= 1.7, [Fe/H] >= -0.5
    if star_info["K_mag_abs"] >= 4.6 and star_info["feh_temp"] >= -0.5:
        r_star_adopt[star_i] = r_star_m15[star_i]
        e_r_star_adopt[star_i] = e_r_star_m15[star_i]
        radius_adopted_ref[star_i] = "M15"

    # Adopt Kesseli+2019 for (BP-RP) >= 1.7, [Fe/H] < -0.5
    elif star_info["K_mag_abs"] > 4.6 and star_info["feh_temp"] < -0.5:
        r_star_adopt[star_i] = r_star_k19[star_i]
        e_r_star_adopt[star_i] = e_r_star_k19[star_i]
        radius_adopted_ref[star_i] = "K19"

    # Adopt Kiman+2024 for (BP-RP) < 1.7
    elif star_info["K_mag_abs"] < 4.6:
        r_star_adopt[star_i] = r_star_k24[star_i]
        e_r_star_adopt[star_i] = e_r_star_k24[star_i]
        radius_adopted_ref[star_i] = "K24"

obs_join["r_star_adopt"] = r_star_adopt
obs_join["e_r_star_adopt"] = e_r_star_adopt
obs_join["r_star_adopt_ref"] = radius_adopted_ref

# ---------------------------------------------
# Update logg
# ---------------------------------------------
logg, e_logg = pp.compute_logg(
    masses=obs_join["mass_m19"].values,
    e_masses=obs_join["e_mass_m19"].values,
    radii=r_star_adopt,
    e_radii=e_r_star_adopt,)

obs_join["logg_m19"] = logg
obs_join["e_logg_m19"] = e_logg

#------------------------------------------------------------------------------
# Calculate Teff via empirical relations
#------------------------------------------------------------------------------
# Calculate Mann+2015 Teff
teff_M15, e_teff_M15 = pp.compute_mann_2015_teff(
    colour=obs_join["BP-RP_dr3"].values,
    feh=feh_values.copy(),
    relation="BP - RP (DR3), [Fe/H], M15+K19")

obs_join["teff_M15_BP_RP_feh"] = teff_M15
obs_join["e_teff_M15_BP_RP_feh"] = e_teff_M15

# Calculate Casagrande+2021 Teff
teff_C21, e_teff_C21 = pp.compute_casagrande_2021_teff(
    colour=obs_join["BP-RP_dr3"].values,
    logg=obs_join["logg_m19"].values,
    feh=feh_values.copy(),
    relation="(BP-RP)",
    enforce_bounds=True,
    regime="dwarf",)

obs_join["teff_C21_BP_RP_logg_feh"] = teff_C21
obs_join["e_teff_C21_BP_RP_logg_feh"] = e_teff_C21

#------------------------------------------------------------------------------
# Science target vetting
#------------------------------------------------------------------------------
n_star = len(obs_join)

# Apply RUWE cut, but *only* for non-interferometric targets, since unresolved
# binarity would be revealed there), and apply the mask
if ls.enforce_ruwe:
    bad_ruwe_mask = np.logical_and(
        ~np.isnan(obs_join["teff_int"]),
        obs_join["ruwe_dr3"] > ls.ruwe_threshold)
else:
    bad_ruwe_mask = np.full(n_star, False)
        
# In local bubble (i.e. minimal reddening)
if ls.enforce_in_local_bubble:
    too_distant_mask = obs_join["dist"] >= params.LOCAL_BUBBLE_DIST_PC
else:
    too_distant_mask = np.full(n_star, False)

# 2MASS photometry is unblended
if ls.enforce_2mass_unblended:
    blended_2mass_mask = np.logical_or(
        obs_join["blended_2mass"].values.astype(bool),
        (obs_join["blended_2mass_prim"] == "yes").values.astype(bool))
else:
    blended_2mass_mask = np.full(n_star, False)

# Enforce that stars do not have aberrant photometric vs spectroscopic logg
# (but allow selected exceptions from the edges of the parameter space).
if ls.enforce_aberrant_logg and ls.allow_aberrant_logg_exceptions:
    has_aberrant_logg = np.logical_and(
        obs_join["flagged_logg"].values.astype(bool),
        ~obs_join["flagged_logg_exception"].values.astype(bool))

# As above, but do *not* allow exceptions
elif ls.enforce_aberrant_logg and not ls.allow_aberrant_logg_exceptions:
    has_aberrant_logg = obs_join["flagged_logg"].values.astype(bool)
else:
    has_aberrant_logg = np.full(n_star, False)

# Combine into a single mask for science target quality
sci_keep_mask = np.all(np.stack([
    ~bad_ruwe_mask,
    ~too_distant_mask,
    ~blended_2mass_mask,
    ~has_aberrant_logg]), axis=0)

#------------------------------------------------------------------------------
# Binary vetting
#------------------------------------------------------------------------------
# Enforce the system has not been marked as not useful
if ls.enforce_system_useful:
    binary_syst_useful = (obs_join["useful"] != "no").values.astype(bool)
else:
    binary_syst_useful = np.full(n_star, True)

# Primary RUWE <= 1.4
if ls.enforce_primary_ruwe:
    good_fgk_primary_ruwe = obs_join["ruwe_dr3_prim"] <= ls.ruwe_threshold
else:
    good_fgk_primary_ruwe = np.full(n_star, True)
    
# Primary BP-RP is not too cool
if ls.enforce_primary_BP_RP_colour:
    good_primary_bp_rp_mask = np.logical_and(
        obs_join["BP-RP_dr3_prim"].values > ls.binary_primary_BP_RP_bound[0],
        obs_join["BP-RP_dr3_prim"].values < ls.binary_primary_BP_RP_bound[1])
else:
    good_primary_bp_rp_mask = np.full(n_star, True)

# Parallaxes are consistent
if ls.enforce_parallax_consistency:
    delta_plx = np.abs(obs_join["plx_dr3"]-obs_join["plx_dr3_prim"]).values
    consistent_syst_plx = delta_plx < ls.binary_max_delta_parallax
else:
    consistent_syst_plx = np.full(n_star, True)

# Proper motions are consistent
if ls.enforce_pm_consistency:
    total_pm_prim = np.sqrt(
        obs_join["pmra_dr3_prim"]**2+obs_join["pmdec_dr3_prim"]**2)
    total_pm_sec = np.sqrt(obs_join["pmra_dr3"]**2+obs_join["pmdec_dr3"]**2)
    pm_norm_diff = \
        total_pm_prim/obs_join["dist_prim"] - total_pm_sec/obs_join["dist"]

    consistent_syst_pm = pm_norm_diff.values < ls.binary_max_delta_norm_PM
else:
    consistent_syst_pm = np.full(n_star, True)

# RVs are consistent
if ls.enforce_rv_consistency:
    delta_rv = np.abs(obs_join["rv_dr3_prim"]-obs_join["rv_dr3"]).values
    consistent_syst_rvs = delta_rv < ls.binary_max_delta_rv
else:
    consistent_syst_rvs = np.full(n_star, True)

# Produce a single mask for binary system quality
syst_keep_mask = np.all(np.stack([
    binary_syst_useful,
    good_fgk_primary_ruwe,
    good_primary_bp_rp_mask,
    consistent_syst_plx,
    consistent_syst_rvs,]), axis=0)

#------------------------------------------------------------------------------
# Collating masks
#------------------------------------------------------------------------------
# Now combine general science target mask with the binary mask (but only apply
# the binary mask to binary stars.)
keep_mask = np.logical_and(
    sci_keep_mask,
    np.logical_or(syst_keep_mask, ~is_cpm),)

# Store the masks in our dataframe
obs_join["passed_quality_cuts"] = keep_mask
obs_join["passed_sci_quality_cuts"] = sci_keep_mask
obs_join["passed_syst_quality_cuts"] = syst_keep_mask

# Finally, correct for any misc exceptions
if ls.allow_misc_exceptions:
    for source_id in ls.misc_exceptions_source_ids:
        obs_join.at[source_id, "passed_quality_cuts"] = True
        obs_join.at[source_id, "passed_sci_quality_cuts"] = True
        obs_join.at[source_id, "passed_syst_quality_cuts"] = True

#------------------------------------------------------------------------------
# Correcting chemodynamic [X/Fe]
#------------------------------------------------------------------------------
if ls.do_CD_polynomial_correction:
    # Load in the fitted polynomials
    with open(ls.CD_polynomial_fn, 'rb') as input_file:
        poly_dict_CD = pickle.load(input_file)

    # Loop over all abundance labels (ignoring [Fe/H]) and perform correction
    for X_Fe in ls.abundance_labels[1:]:
        # Column name for abundance
        X_Fe_col = "{}_SM25".format(X_Fe)

        # Grab the polynomial and its bounds in [Fe/H]
        poly = poly_dict_CD[("SM25", X_Fe)]
        Fe_H_bounds = poly.Fe_H_bounds
        
        # Work out beyond bounds
        abund = obs_join[X_Fe_col].values

        is_below = feh_values < Fe_H_bounds[0]
        within_bounds = np.logical_and(
            feh_values >= Fe_H_bounds[0], feh_values <= Fe_H_bounds[1])
        is_above = feh_values > Fe_H_bounds[1]

        # Correct for the fit (below, overlapping, & above [Fe/H] bounds)
        abund_corr = np.full_like(abund, np.nan)
        abund_corr[is_below] = abund[is_below] + poly(Fe_H_bounds[0])
        abund_corr[within_bounds] = \
            abund[within_bounds] + poly(feh_values[within_bounds])
        abund_corr[is_above] = abund[is_above] + poly(Fe_H_bounds[1])

        obs_join[X_Fe_col] = abund_corr

#------------------------------------------------------------------------------
# Setup training labels
#------------------------------------------------------------------------------
# Prepare our labels (update DataFrame in place)
params.prepare_labels(
    obs_join=obs_join,
    abundance_labels=ls.abundance_labels,
    abund_order_k=ls.ABUND_ORDER_K,
    abund_order_m=ls.ABUND_ORDER_M,
    abund_order_binary=ls.ABUND_ORDER_BINARY,
    synth_params_available=False,
    mid_K_BP_RP_bound=ls.mid_K_BP_RP_bound,
    mid_K_MKs_bound=ls.mid_K_MKs_bound,
    mid_K_BP_RP_trustworthy_X_Fe=ls.mid_K_BP_RP_trustworthy_X_Fe,)

#------------------------------------------------------------------------------
# Saving DataFrame
#------------------------------------------------------------------------------
# Format our dataframe so it will save correctly
for col in obs_join.columns.values:
    if obs_join[col].dtype == np.dtype("O"):
        obs_join = obs_join.astype(dtype={col:str}, copy=False)

    # Replace any '-' with '_'
    if "-" in col:
        obs_join.rename(columns={col:col.replace("-","_")}, inplace=True)

    # Make sure we don't have any columns that start with numbers that would
    # be invalid python variable names (e.g. '2mass_<name>')
    if col[0].isnumeric():
        obs_join.rename(columns={col:"_{}".format(col)}, inplace=True)

#------------------------------------------------------------------------------
# Plotting
#------------------------------------------------------------------------------
icb = obs_join["is_cannon_benchmark"].values

splt.plot_cannon_cmd(
    benchmark_colour=obs_join["BP_RP_dr3"][icb].values,
    benchmark_mag=obs_join["K_mag_abs"][icb].values,
    benchmark_feh=obs_join["label_adopt_Fe_H"][icb].values,
    highlight_mask=obs_join["is_cpm"][icb].values,
    highlight_mask_label="Binary Benchmark",
    highlight_mask_2=obs_join["is_mid_k_dwarf"][icb].values,
    highlight_mask_label_2="Early-Mid K Dwarf",
    plot_folder="paper",)

#------------------------------------------------------------------------------
# Save dataframe
#------------------------------------------------------------------------------
pu.save_fits_table(
    extension="CANNON_INFO",
    dataframe=obs_join,
    label=ls.fits_label,
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,)

# TODO: do this again to avoid the 1e10 instead of NaN issue on motley
obs_join = pu.load_fits_table(
    extension="CANNON_INFO",
    label=ls.fits_label,
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,)

pu.save_fits_table(
    extension="CANNON_INFO",
    dataframe=obs_join,
    label=ls.fits_label,
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,)