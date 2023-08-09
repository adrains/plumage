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
import plumage.parameters as params

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
# Fits file to load from plumage/spectra
std_label = "cannon"

# If True, we calculate [X/Fe] and our Cannon model will work in [X/Fe] space.
# Otherwise, we calculate [X/H] and model in [X/H] space.
calc_x_fe_abund = True

# At the moment we're only supporting Ti_H. Eventually this will be more
# generic.
abundance_labels = ["Ti_H"]

label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = len(label_names)

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Import literature info
#  - Note that since we don't use any photometry with the Cannon, and since our
#  stellar parameters come from our results table, we can set a bunch of the
#  extra import parameters to False to speed things up.
std_info = pu.load_info_cat(
    "data/std_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="dr3",)

std_info.reset_index(inplace=True)
std_info.set_index("source_id_dr2", inplace=True)

# Load results table, update index name to DR2
obs_std = pu.load_fits_table("OBS_TAB", std_label, path="spectra")
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
obs_join = obs_join.join(cpm_join, "source_id_dr3", rsuffix="_cpm")

# And make sure our CPM column is correct
is_cpm = np.array([type(val) == str for val in obs_join["prim_name"].values])
obs_join["is_cpm"] = is_cpm

#------------------------------------------------------------------------------
# Make quality cuts
# TODO: this code is shared with the phot [Fe/H] relation--generalise
#------------------------------------------------------------------------------
DELTA_PARALLAX = 0.2
DELTA_NORM_PM = 5
DELTA_RV = 5

RUWE_THRESHOLD = 1.4
BP_RP_BOUNDS_PRIM = (-100, 1.5)

enforce_system_useful = True

# Primary data quality + suitability
enforce_primary_ruwe = False
enforce_primary_bp_rp_colour = True

# Secondary data quality
enforce_secondary_ruwe = True
enforce_secondary_2mass_unblended = True

# Distance and velocity consistency
enforce_parallax_consistency = True
enforce_pm_consistency = True
enforce_rv_consistency = True
enforce_in_local_bubble = False

# General RUWE cut for non-binaries
enforce_general_ruwe = True

# Allow exceptions for certain stars that fail
allow_exceptions = True
exception_source_ids = [
    "5853498713190525696",  # Proxima Cen
]

cpm_keep_mask = np.full(len(obs_join), True)

# Enforce the system has not been marked as not useful
if enforce_system_useful:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["useful"] != "no")

# Primary RUWE <= 1.4
if enforce_primary_ruwe:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["ruwe_dr3_prim"] <= RUWE_THRESHOLD)

# Primary BP-RP
if enforce_primary_bp_rp_colour:
    bp_rp_mask = np.logical_and(
        obs_join["BP-RP_dr3_prim"] > BP_RP_BOUNDS_PRIM[0],
        obs_join["BP-RP_dr3_prim"] < BP_RP_BOUNDS_PRIM[1])
    cpm_keep_mask = np.logical_and(cpm_keep_mask, bp_rp_mask)

# In local bubble (i.e. minimal reddening)
if enforce_in_local_bubble:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["dist"] <= params.LOCAL_BUBBLE_DIST_PC)

# Secondary RUWE <= 1.4
if enforce_secondary_ruwe:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["ruwe_dr3"] <= RUWE_THRESHOLD)

# Secondary 2MASS unblended
if enforce_secondary_2mass_unblended:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        obs_join["blended_2mass"] != "yes")

# Parallaxes are consistent
if enforce_parallax_consistency:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        np.abs(obs_join["plx_dr3"]-obs_join["plx_dr3_prim"]) < DELTA_PARALLAX)

# Proper motions are consistent
if enforce_pm_consistency:
    total_pm_prim = np.sqrt(
        obs_join["pmra_dr3_prim"]**2+obs_join["pmdec_dr3_prim"]**2)
    total_pm_sec = np.sqrt(obs_join["pmra_dr3"]**2+obs_join["pmdec_dr3"]**2)
    pm_norm_diff = \
        total_pm_prim/obs_join["dist_prim"] - total_pm_sec/obs_join["dist"]

    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        pm_norm_diff < DELTA_NORM_PM)

# RVs are consistent
if enforce_rv_consistency:
    cpm_keep_mask = np.logical_and(
        cpm_keep_mask,
        np.abs(obs_join["rv_dr3_prim"]-obs_join["rv_dr3"]) < DELTA_RV)

# Note that we only want to apply this cut to the binary stars
keep_mask = np.logical_or(~is_cpm, cpm_keep_mask)

# Now apply the general RUWE cut (*only* for non-interferometric targets, since
# unresolved binarity would be revealed there), and apply the mask
if enforce_general_ruwe:
    keep_mask = np.logical_and(
        keep_mask,
        np.logical_or(
            ~np.isnan(obs_join["teff_int"]),
            obs_join["ruwe_dr3"] <= RUWE_THRESHOLD))

# Update the mask
obs_join["passed_quality_cuts"] = keep_mask

# Manually correct any exceptions
if allow_exceptions:
    for source_id in exception_source_ids:
        obs_join.at[source_id, "passed_quality_cuts"] = True

#------------------------------------------------------------------------------
# Setup training labels
#------------------------------------------------------------------------------
# Import Montes+18 abundance trends to have a better naive guess at abundances
# for stars with unknown abundances
montes18_abund_trends = pd.read_csv("data/montes18_abundance_trends.csv") 

# Prepare our labels
label_values, label_sigmas, std_mask, label_sources, label_nondefault = \
    params.prepare_labels(
        obs_join=obs_join,
        n_labels=n_labels,
        abundance_labels=abundance_labels,
        abundance_trends=montes18_abund_trends,
        calc_x_fe_abund=calc_x_fe_abund,)

# Compute the variances
label_var_all = label_sigmas**2

# Add the mask and adopted labels to the dataframe
obs_join["has_complete_label_set"] = std_mask

for lbl_i, lbl in enumerate(label_names):
    obs_join["label_adopt_{}".format(lbl)] = label_values[:,lbl_i]
    obs_join["label_adopt_sigma_{}".format(lbl)] = label_sigmas[:,lbl_i]
    obs_join["label_adopt_var_{}".format(lbl)] = label_var_all[:,lbl_i]
    obs_join["label_source_{}".format(lbl)] = label_sources[:,lbl_i]
    obs_join["label_nondefault_{}".format(lbl)] = label_nondefault[:,lbl_i]

# And combine masks to get adopted benchmarks
is_cannon_benchmark = np.logical_and(
    obs_join["passed_quality_cuts"],
    obs_join["has_complete_label_set"]
)

obs_join["is_cannon_benchmark"] = is_cannon_benchmark

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

pu.save_fits_table("CANNON_INFO", obs_join, "cannon")

# TODO: do this again to avoid the 1e10 instead of NaN issue on motley
obs_join = pu.load_fits_table("CANNON_INFO", "cannon")
pu.save_fits_table("CANNON_INFO", obs_join, "cannon")