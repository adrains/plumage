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
    gdr=ls.gdr,)

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

# Now merge prim and sec CPM info on prim_name
cpm_join = cpm_sec.join(cpm_prim, "prim_name", rsuffix="_prim")

# Finally, merge the CPM info with obs_join to have everything in one dataframe
obs_join = obs_join.join(cpm_join, "source_id_dr3", rsuffix="_sec").copy()

# And make sure our CPM column is correct
is_cpm = np.array([type(val) == str for val in obs_join["prim_name"].values])
obs_join["is_cpm"] = is_cpm

# Load in and crossmatch with sampled params
sp_df = pd.read_csv(ls.sampled_param_csv, dtype={"source_id_dr3":str},)
sp_df.set_index("source_id_dr3", inplace=True)
obs_join = obs_join.join(sp_df, "source_id_dr3")

# Finally clean this and remove duplicate columns
cols = obs_join.columns.values
mm = np.array(["_sec" in cv for cv in cols])
obs_join = obs_join[cols[~mm]].copy()


#------------------------------------------------------------------------------
# Collate [Fe/H]
#------------------------------------------------------------------------------
# Select adopted [Fe/H] values which are needed for empirical Teff relations
# TODO: HACK in that we just run the function twice...
feh_info_all = []

for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()): 
    feh_info = params.select_Fe_H_label(star_info)
    feh_info_all.append(feh_info)

feh_info_all = np.vstack(feh_info_all)

feh_values = feh_info_all[:,0].astype(float)

#------------------------------------------------------------------------------
# Calculate empirical relations
#------------------------------------------------------------------------------
# Calculate Mann+2015 Teff
teff_M15, e_teff_M15 = pp.compute_mann_2015_teff(
    colour=obs_join["BP-RP_dr3"].values,
    feh=feh_values,
    relation="BP - RP (DR3), [Fe/H]")

obs_join["teff_M15_BP_RP_feh"] = teff_M15
obs_join["e_teff_M15_BP_RP_feh"] = e_teff_M15

# Calculate Casagrande+2021 Teff
teff_C21, e_teff_C21 = pp.compute_casagrande_2021_teff(
    colour=obs_join["BP-K"].values,
    logg=obs_join["logg_m19"].values,
    feh=feh_values,
    relation="(BP-Ks)",)

obs_join["teff_C21_BP_Ks_logg_feh"] = teff_C21
obs_join["e_teff_C21_BP_Ks_logg_feh"] = e_teff_C21

# Sanity check the limits, since the relations break down at extreme colours
pass

#------------------------------------------------------------------------------
# Make quality cuts
#------------------------------------------------------------------------------
cpm_keep_mask = np.full(len(obs_join), True)

# Enforce the system has not been marked as not useful
if ls.enforce_system_useful:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["useful"] != "no")

# Primary RUWE <= 1.4
if ls.enforce_primary_ruwe:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["ruwe_dr3_prim"] <= ls.ruwe_threshold)

# Primary BP-RP
if ls.enforce_primary_BP_RP_colour:
    bp_rp_mask = np.logical_and(
        obs_join["BP-RP_dr3_prim"] > ls.binary_primary_BP_RP_bound[0],
        obs_join["BP-RP_dr3_prim"] < ls.binary_primary_BP_RP_bound[1])
    cpm_keep_mask = np.logical_and(cpm_keep_mask, bp_rp_mask)

# In local bubble (i.e. minimal reddening)
if ls.enforce_in_local_bubble:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["dist"] <= params.LOCAL_BUBBLE_DIST_PC)

# Secondary RUWE <= 1.4
# TODO: surely this is unnecessary since it's not doing anything different
# to the general RUWE check?
if ls.enforce_secondary_ruwe:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["ruwe_dr3"] <= ls.ruwe_threshold)

# Secondary 2MASS unblended
if ls.enforce_secondary_2mass_unblended:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["blended_2mass"] != "yes")

# Parallaxes are consistent
if ls.enforce_parallax_consistency:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        np.abs(obs_join["plx_dr3"]-obs_join["plx_dr3_prim"]) < ls.binary_max_delta_parallax)

# Proper motions are consistent
if ls.enforce_pm_consistency:
    total_pm_prim = np.sqrt(
        obs_join["pmra_dr3_prim"]**2+obs_join["pmdec_dr3_prim"]**2)
    total_pm_sec = np.sqrt(obs_join["pmra_dr3"]**2+obs_join["pmdec_dr3"]**2)
    pm_norm_diff = \
        total_pm_prim/obs_join["dist_prim"] - total_pm_sec/obs_join["dist"]

    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        pm_norm_diff < ls.binary_max_delta_norm_PM)

# RVs are consistent
if ls.enforce_rv_consistency:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        np.abs(obs_join["rv_dr3_prim"]-obs_join["rv_dr3"]) < ls.binary_max_delta_rv)

# Note that we only want to apply this cut to the binary stars
keep_mask = np.logical_or(~is_cpm, cpm_keep_mask)

# Now apply the general RUWE cut (*only* for non-interferometric targets, since
# unresolved binarity would be revealed there), and apply the mask
if ls.enforce_general_ruwe:
    keep_mask = np.logical_and(
        keep_mask,
        np.logical_or(
            ~np.isnan(obs_join["teff_int"]),
            obs_join["ruwe_dr3"] <= ls.ruwe_threshold))

# Update the mask
obs_join["passed_quality_cuts"] = keep_mask

# Manually correct any exceptions
if ls.allow_exceptions:
    for source_id in ls.exception_source_ids:
        obs_join.at[source_id, "passed_quality_cuts"] = True

#------------------------------------------------------------------------------
# Setup training labels
#------------------------------------------------------------------------------
# Prepare our labels (update DataFrame in place)
params.prepare_labels(obs_join=obs_join, max_teff=5000, synth_params_available=False)

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

pu.save_fits_table("CANNON_INFO", obs_join, ls.spectra_label)

# TODO: do this again to avoid the 1e10 instead of NaN issue on motley
obs_join = pu.load_fits_table("CANNON_INFO", ls.spectra_label)
pu.save_fits_table("CANNON_INFO", obs_join, ls.spectra_label)