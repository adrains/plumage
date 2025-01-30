"""Script to prepare the training dataset for use when training a Cannon model.
The resulting training labels are saved to our fits file.

This script is part of a series of Cannon scripts. The main sequence is:
 1) prepare_stannon_training_sample.py     --> label preparation
 2) train_stannon.py                       --> training and cross validation
 3) make_stannon_diagnostics.py            --> diagnostic plots + result tables
 4) run_stannon.py                         --> running on science spectra
"""
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
# Import literature info
#  - Note that since we don't use any photometry with the Cannon, and since our
#  stellar parameters come from our results table, we can set a bunch of the
#  extra import parameters to False to speed things up.
std_info = pu.load_info_cat(
    path=ls.std_info_fn,
    use_mann_code_for_masses=ls.use_mann_code_for_masses,
    in_paper=ls.in_paper,
    only_observed=ls.only_observed,
    do_extinction_correction=ls.do_extinction_correction,
    do_skymapper_crossmatch=ls.do_skymapper_crossmatch,
    gdr=ls.gdr,
    do_use_mann_15_JHK=True,)

std_info.reset_index(inplace=True)
std_info.set_index("source_id_dr2", inplace=True)

# Load results table, update index name to DR2
obs_std = pu.load_fits_table("OBS_TAB", ls.spectra_label, path="spectra")
obs_std.index.rename("source_id_dr2", inplace=True)

# To crossmatch with our fitted parameters from Rains+21 we need to do a
# crossmatch on DR2 coordinates instead for legacy reasons (as these were the
# IDs used when observing). Once we've done the crossmatch, we want to update
# the index be the Gaia DR3 source ID instead.
obs_join = obs_std.join(std_info, "source_id_dr2", rsuffix="_info")
obs_join.set_index("source_id_dr3", inplace=True)

# Import our two tables of CPM information
tsv_primaries = "data/cpm_primaries_dr3.tsv"
tsv_secondaries = "data/cpm_secondaries_dr3.tsv"

cpm_prim = pu.load_info_cat(
    tsv_primaries,
    clean=False,
    allow_alt_plx=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="dr3",
    has_2mass=False,)

cpm_sec = pu.load_info_cat(
    tsv_secondaries,
    clean=False,
    allow_alt_plx=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
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

# Finally, merge the CPM info with obs_join to have everything in one dataframe
obs_join = obs_join.join(cpm_join, "source_id_dr3", rsuffix="_sec").copy()

# And make sure our CPM column is correct
is_cpm = np.array([type(val) == str for val in obs_join["prim_name"].values])
obs_join["is_cpm"] = is_cpm

# Add a column for K dwarf benchmarks. Note that we need to exclude the bad
# parameters for RB20 for cool stars.
is_nan = [True if type(rr) == float else False
          for rr in obs_join["useful_rb20"].values]
is_useful_rb20 = obs_join["useful_rb20"].values
is_useful_rb20[is_nan] = False

obs_join.astype({"useful_rb20":bool}, copy=False)
obs_join["useful_rb20"] = is_useful_rb20

has_mid_k_reference = np.any([
    ~np.isnan(obs_join["Teff_b16"].values),
    np.logical_and(~np.isnan(obs_join["Teff_rb20"].values), is_useful_rb20),
    ~np.isnan(obs_join["teff_vf05"].values),
    ~np.isnan(obs_join["Teff_m18"].values),
    ~np.isnan(obs_join["teff_s08"].values),
    ~np.isnan(obs_join["Teff_L18"].values),], axis=0).astype(bool)

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
    feh_info = params.select_Fe_H_label(
        star_info=star_info,
        mid_K_BP_RP_bound=ls.mid_K_BP_RP_bound,
        mid_K_MKs_bound=ls.mid_K_MKs_bound)
    feh_info_all.append(feh_info)

feh_info_all = np.vstack(feh_info_all)

feh_values = feh_info_all[:,0].astype(float)

# TEMPORARY archive of chosed [Fe/H]
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
    
    else:
        print("#{}: {} has no radius.".format(star_i, source_id))

        if source_id == "3545469496823737856":
            import pdb; pdb.set_trace()

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
# Setup training labels
#------------------------------------------------------------------------------
# Prepare our labels (update DataFrame in place)
params.prepare_labels(
    obs_join=obs_join,
    max_teff=5000,
    synth_params_available=False,
    mid_K_BP_RP_bound=ls.mid_K_BP_RP_bound,
    mid_K_MKs_bound=ls.mid_K_MKs_bound,)

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
# HACK: Dump unneeded columns so we can save the fits file.
#------------------------------------------------------------------------------
# This is a list of unneeded columns (e.g. extra abundances) that for now we're
# dropping so that the fits file has < 1000 columns. Eventually we'll need to
# update the saving/loading routine to split the dataframe into multiple HDUs
# on save (though really this is the result of poor databasing and having
# everything in a single colossal table for convenience).
cols_to_remove = [
    "Fbol_m15_old", "e_fbol_m15_old", "radius_m15_old", "e_radius_m15_old", 
    "m_star_m15_old", "e_m_star_m15_old", "logg_m15_old", "e_logg_m15_old", 
    "teff_m15_old", "e_teff_m15_old", "feh_m15_old", "e_feh_m15_old", 
    "teff_ra12_old", "e_teff_ra12_old", "feh_ra12_old", "e_feh_ra12_old", 
    "mh_old", "e_mh_old", "teff_other", "e_teff_other", "radius_other",
    "e_radius_other", "mass_other", "e_mass_other", "logg_other",
    "e_logg_other", "feh_other", "e_feh_other", "in_tess_paper", "has_ra12",
    "has_g14", "has_m15", "has_vf05", "has_s08", "has_L18", "has_b16",
    "has_m18", "has_rb20", "has_sc", "collated_by_prim", "has_n14_prim",
    "has_m13_prim", "has_m14_prim", "has_m18_prim", "has_vf05_prim",
    "has_a12_prim", "has_s08_prim", "has_b18_prim", "has_rb20_prim",
    "ra_dr2", "dec_dr2", "plx_dr2", "e_plx_dr2", "pm_ra_dr2",
    "pm_dec_dr2", "dup_dr2", "G_mag_dr2", "e_G_mag_dr2", "BP_mag_dr2",
    "e_BP_mag_dr2", "Rp_mag_dr2", "e_RP_mag_dr2", "BP_RP_excess_dr2",
    "BP_RP_dr2", "rv_dr2", "e_rv_dr2", "ruwe_dr2", "comments_gaia_dr2", 
    'Cr_H_L18', 'Cr_H_b16', 'Cr_H_rb20', 'Cr_H_m18', 'eCr_H_m18', 
    'Cr_H_m18_prim', 'eCr_H_m18_prim', 'Cr_H_b16_prim', 'Cr_H_rb20_prim',
    'Ni_H_vf05', 'Ni_H_L18', 'Ni_H_b16', 'Ni_H_rb20', 'Ni_H_m18', 'eNi_H_m18',
    'Ni_H_a12', 'e_Ni_H_a12', 'o_Ni_H_a12','Ni_H_m18_prim', 'eNi_H_m18_prim',
    'Ni_H_vf05_prim', 'Ni_H_a12_prim', 'e_Ni_H_a12_prim', 'o_Ni_H_a12_prim',
    'Ni_H_b16_prim', 'Ni_H_rb20_prim', 'Sc_H_L18', 'Sc_H_m18', 'eSc_H_m18',
    'Sc_H_m18_prim', 'eSc_H_m18_prim', 'Mn_H_L18', 'Mn_H_b16', 'Mn_H_rb20', 
    'Mn_H_m18', 'eMn_H_m18', 'Mn_H_a12', 'e_Mn_H_a12', 'o_Mn_H_a12',
    'Mn_H_m18_prim', 'eMn_H_m18_prim', 'Mn_H_a12_prim', 'e_Mn_H_a12_prim',
    'o_Mn_H_a12_prim', 'Mn_H_b16_prim', 'Mn_H_rb20_prim', 'Co_H_L18',
    'Co_H_m18', 'eCo_H_m18', 'Co_H_a12', 'e_Co_H_a12', 'o_Co_H_a12',
    'Co_Hc_a12', 'Co_H_m18_prim', 'eCo_H_m18_prim', 'Co_H_a12_prim',
    'e_Co_H_a12_prim', 'o_Co_H_a12_prim', 'Co_Hc_a12_prim', 'Y_H_L18',
    'Y_H_b16', 'Y_H_rb20', 'Y_H_b16_prim', 'Y_H_rb20_prim', 'Al_H_L18',
    'Al_H_b16', 'Al_H_rb20', 'Al_H_m18', 'eAl_H_m18', 'Al_H_a12', 'e_Al_H_a12',
    'o_Al_H_a12', 'Al_Hc_a12', 'Al_H_m18_prim', 'eAl_H_m18_prim',
    'Al_H_a12_prim', 'e_Al_H_a12_prim', 'o_Al_H_a12_prim', 'Al_Hc_a12_prim',
    'Al_H_b16_prim', 'Al_H_rb20_prim', 'Mg_H_L18', 'Mg_H_b16', 'Mg_H_rb20',
    'Mg_H_m18', 'eMg_H_m18', 'Mg_H_a12', 'e_Mg_H_a12', 'o_Mg_H_a12',
    'Mg_H_m18_prim', 'eMg_H_m18_prim', 'Mg_H_a12_prim', 'e_Mg_H_a12_prim',
    'o_Mg_H_a12_prim', 'Mg_H_b16_prim', 'Mg_H_rb20_prim', 'Ca_H_L18',
    'Ca_H_b16', 'Ca_H_rb20', 'Ca_H_m18', 'eCa_H_m18', 'Ca_H_a12', 'e_Ca_H_a12',
    'o_Ca_H_a12', 'Ca_H_m18_prim', 'eCa_H_m18_prim', 'Ca_H_a12_prim',
    'e_Ca_H_a12_prim', 'o_Ca_H_a12_prim', 'Ca_H_b16_prim', 'Ca_H_rb20_prim',
    'C_Hmean_L18', 'C_H_b16', 'C_H_rb20', 'C_H_b16_prim', 'C_H_rb20_prim',
    'N_H_b16', 'N_H_rb20', 'N_H_b16_prim', 'N_H_rb20_prim', 'O_Hmean_L18',
    'O_H_b16', 'O_H_rb20', 'O_H_b16_prim','O_H_rb20_prim', 'V_H_L18',
    'V_H_b16', 'V_H_rb20', 'V_H_m18', 'eV_H_m18', 'V_H_a12','e_V_H_a12',
    'o_V_H_a12', 'V_Hc_a12', 'V_H_m18_prim', 'eV_H_m18_prim', 'V_H_a12_prim',
    'e_V_H_a12_prim', 'o_V_H_a12_prim', 'V_Hc_a12_prim', 'V_H_b16_prim',
    'V_H_rb20_prim', 'Na_H_vf05', 'Na_H_L18', 'Na_H_b16', 'Na_H_rb20',
    'Na_H_m18', 'eNa_H_m18', 'Na_H_a12', 'e_Na_H_a12', 'o_Na_H_a12',
    'Na_Hc_a12', 'Na_H_m18_prim', 'eNa_H_m18_prim', 'Na_H_vf05_prim',
    'Na_H_a12_prim', 'e_Na_H_a12_prim', 'o_Na_H_a12_prim', 'Na_Hc_a12_prim',
    'Na_H_b16_prim', 'Na_H_rb20_prim',]

obs_join.drop(columns=cols_to_remove, inplace=True)

#------------------------------------------------------------------------------
# Plotting
#------------------------------------------------------------------------------
icb = obs_join["is_cannon_benchmark"].values

splt.plot_cannon_cmd(
    benchmark_colour=obs_join["BP_RP_dr3"][icb].values,
    benchmark_mag=obs_join["K_mag_abs"][icb].values,
    benchmark_feh=obs_join["label_adopt_feh"][icb].values,
    highlight_mask=obs_join["is_cpm"][icb].values,
    highlight_mask_label="Binary Benchmark",
    highlight_mask_2=obs_join["is_mid_k_dwarf"][icb].values,
    highlight_mask_label_2="Early-Mid K Dwarf",
    plot_folder="paper",)

#------------------------------------------------------------------------------
# Save dataframe
#------------------------------------------------------------------------------
pu.save_fits_table("CANNON_INFO", obs_join, ls.spectra_label)

# TODO: do this again to avoid the 1e10 instead of NaN issue on motley
obs_join = pu.load_fits_table("CANNON_INFO", ls.spectra_label)
pu.save_fits_table("CANNON_INFO", obs_join, ls.spectra_label)