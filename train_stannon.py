"""Script to train, cross validate, and save a Cannon model. After a model has
been created, the scripts make_stannon_diagnostics.py and run_stannon.py should
be used.
"""
import numpy as np
import pandas as pd
import plumage.utils as pu
import plumage.parameters as params
import stannon.stannon as stannon

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
# Suppress ouput from Stan during training (recommended True unless debugging)
suppress_stan_output = False

# Whether to initialise theta and s2 vectors for a label uncertainty model
# using the vectors from a trained basic model. The idea is that, even though
# these will ultimately be different, it's a better initial guess than just 
# starting with an array of zeroes.
init_uncertainty_model_with_basic_model = True

# The maximum amount of iterations Stan will run while fitting the model
max_iter = 100000

# Whether to run leave-one-out cross validation on Cannon model
do_cross_validation = False

# Whether to do sigma clipping using trained Cannon model. If True, an initial
# Cannon model is trained and its model spectra are used to sigma clip bad 
# pixels to not be considered for the subsequently trained and adopted model.
do_iterative_bad_px_masking = False
flux_sigma_to_clip = 5

# Normalisation - using using a Gaussian smoothed version of the spectrum, or 
# a much simpler polynomial fit. Only wavelengths > than wl_min_normalisation
# will be considered during either approach to avoid low-SNR blue pixels for
# the coolest stars. TODO: save these parameters in the Cannon model.
do_gaussian_spectra_normalisation = True
wl_min_normalisation = 4000
wl_broadening = 50
poly_order = 4

# Minimum and maximum wavelengths for Cannon model
wl_min_model = 6500
wl_max_model = 6600

# The Cannon model to use - either the 'basic' traditional Cannon model, or a
# model with label uncertainties. If modelling abundances, the version with
# label uncertainties should be used. Either 'basic' or 'label_uncertainties'.
# For a 3-term model, the basic model (on motley) takes of order ~1 min to
# train, and the label uncertainties model takes of order ~20 min. The latter 
# increases to ~33 min when training a 4 term model with [Ti/H]. Note that
# these numbers are for a *single* model, and that cross validation increases
# the runtime by a factor of N_stars.
model_type = "label_uncertainties"
use_label_uniform_variances = False

model_save_path = "spectra"
std_label = "cannon"

# Whether to fit for abundances. At the moment our abundance heirarchy is
# Montes+18 > Valenti+Fischer05 > Adibekyan+12. Not recommended to fit > 1-2.
# Available options (for Montes+18, which is the most complete): 
# Na, Mg, Al, Si, Ca, Sc, Ti, V, Cr, Mn, Co, Ni
# Select as e.g.["X_H",..] or leave empty to not use abundances.
abundance_labels = []

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

# Load in RV corrected standard spectra
wls = pu.load_fits_image_hdu("rest_frame_wave", std_label, arm="br")
spec_std_br = pu.load_fits_image_hdu("rest_frame_spec", std_label, arm="br")
e_spec_std_br = pu.load_fits_image_hdu("rest_frame_sigma", std_label, arm="br")

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
label_values_all, label_sig_all, std_mask, label_sources_all = \
    stannon.prepare_labels(
        obs_join=obs_join,
        n_labels=n_labels,
        abundance_labels=abundance_labels,
        abundance_trends=montes18_abund_trends)

# Compute the variances
label_var_all = label_sig_all**2

# Optional for testing: run with uniform variances
if use_label_uniform_variances:
    label_var_all = label_var_all*0 + 1e-3  # NaN values will remain NaN

# Add the mask and adopted labels to the dataframe
obs_join["has_complete_label_set"] = std_mask

for lbl_i, lbl in enumerate(label_names):
    obs_join["label_adopt_{}".format(lbl)] = label_values_all[:,lbl_i]
    obs_join["label_adopt_sigma_{}".format(lbl)] = label_sig_all[:,lbl_i]
    obs_join["label_adopt_var_{}".format(lbl)] = label_var_all[:,lbl_i]
    obs_join["label_source_{}".format(lbl)] = label_sources_all[:,lbl_i]

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

#------------------------------------------------------------------------------
# Flux preparation
#------------------------------------------------------------------------------
fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wls,
        spectra=spec_std_br[is_cannon_benchmark],
        e_spectra=e_spec_std_br[is_cannon_benchmark],
        wl_min_model=wl_min_model,
        wl_max_model=wl_max_model,
        wl_min_normalisation=wl_min_normalisation,
        wl_broadening=wl_broadening,
        do_gaussian_spectra_normalisation=do_gaussian_spectra_normalisation,
        poly_order=poly_order)

#------------------------------------------------------------------------------
# Make and train model
#------------------------------------------------------------------------------
# Diagnostic summary
print("\n\n", "%"*80, "\n", sep="")
print("\tRunning Cannon model:\n\t", "-"*21, sep="")
print("\tmodel: \t\t\t = {}".format(model_type))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(label_names)))
print("\tlabels: \t\t = {}".format(label_names))
print("\tn benchmarks: \t\t = {:0.0f}".format(np.sum(is_cannon_benchmark)))
print("\tGaussian Normalisation:\t = {}".format(
    do_gaussian_spectra_normalisation))
if do_gaussian_spectra_normalisation:
    print("\twl broadening: \t\t = {:0.0f} A".format(wl_broadening))
else:
    print("\tpoly order: \t\t = {:0.0f}".format(poly_order))
print("\tuniform variances: \t = {}".format(use_label_uniform_variances))
print("\tcross validation: \t = {}".format(do_cross_validation))
print("\titerative masking: \t = {}".format(do_iterative_bad_px_masking))
print("\n", "%"*80, "\n\n", sep="")

# Make model
sm = stannon.Stannon(
    training_data=fluxes_norm,
    training_data_ivar=ivars_norm,
    training_labels=label_values_all[is_cannon_benchmark],
    training_ids=obs_join[is_cannon_benchmark].index.values,
    label_names=label_names,
    wavelengths=wls,
    model_type=model_type,
    training_variances=label_var_all[is_cannon_benchmark],
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask,)

# Train model
print("\nRunning initial training with {} benchmarks...".format(
    np.sum(is_cannon_benchmark)))
sm.train_cannon_model(
    suppress_stan_output=suppress_stan_output,
    init_uncertainty_model_with_basic_model=init_uncertainty_model_with_basic_model,
    max_iter=max_iter,)

# If we run the iterative bad px masking, train again afterwards
if do_iterative_bad_px_masking:
    print("\nRunning iterative sigma clipping for bad px...")
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=flux_sigma_to_clip)
    sm.train_cannon_model(suppress_stan_output=suppress_stan_output)

# Run cross validation
if do_cross_validation:
    sm.run_cross_validation()

    labels_pred = sm.cross_val_labels

# ...or just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(model_save_path)

print("Model training complete!")