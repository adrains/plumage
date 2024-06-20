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

is_mid_k_dwarf = np.any([
    ~np.isnan(obs_join["Teff_b16"].values),
    np.logical_and(~np.isnan(obs_join["Teff_rb20"].values), is_useful_rb20),
    ~np.isnan(obs_join["teff_vf05"].values),
    ~np.isnan(obs_join["Teff_m18"].values),
    ~np.isnan(obs_join["teff_s08"].values),
    ~np.isnan(obs_join["Teff_L18"].values),], axis=0).astype(bool)

obs_join["is_mid_k_dwarf"] = is_mid_k_dwarf

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
    delta_plx = np.abs(obs_join["plx_dr3"]-obs_join["plx_dr3_prim"])
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        delta_plx < ls.binary_max_delta_parallax)

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
    delta_rv = np.abs(obs_join["rv_dr3_prim"]-obs_join["rv_dr3"])
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        delta_rv < ls.binary_max_delta_rv)

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
params.prepare_labels(
    obs_join=obs_join,
    max_teff=5000,
    synth_params_available=False)

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
    "Fbol_m15_old", "e_fbol_m15_old", "radius_m15", "e_radius_m15_old", 
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
# Save dataframe
#------------------------------------------------------------------------------
pu.save_fits_table("CANNON_INFO", obs_join, ls.spectra_label)

# TODO: do this again to avoid the 1e10 instead of NaN issue on motley
obs_join = pu.load_fits_table("CANNON_INFO", ls.spectra_label)
pu.save_fits_table("CANNON_INFO", obs_join, ls.spectra_label)