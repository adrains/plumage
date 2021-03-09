"""Script to fit saved TESS light curves using previously determined values for
stellar radius, mass, limb-darkening (from stellar params), and transit params
per values from NASA ExoFOP
"""
import numpy as np
import pandas as pd
from datetime import datetime
import plumage.utils as utils
import plumage.transits as transit
import plumage.parameters as params
import plumage.plotting as pplt
import astropy.constants as const

# Timing
start_time = datetime.now()

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
label = "tess"
spec_path =  "spectra"

# Load in literature info for our stars
tic_info = utils.load_info_cat(
    remove_fp=True,
    only_observed=True,
    use_mann_code_for_masses=True,).reset_index()
tic_info.set_index("TIC", inplace=True)

# Load NASA ExoFOP info on TOIs
toi_info = utils.load_exofop_toi_cat(do_ctoi_merge=True)

# Load in info on observations and fitting results
observations = utils.load_fits_table("OBS_TAB", label, path=spec_path,)

# Choose binning, and import light curves to fit with
fitting_bin_fac = 2

# Load in light curve for fitting. Note that these have already been binned by
# fitting_bin_fac and have had nans removed.
light_curves, sector_dict = transit.load_all_light_curves(
    tic_info.index.values,
    bin_fac=fitting_bin_fac,)

logg_col = "logg_synth"
#logg_col = "logg_m19"

# Temporary join to get combined info in single datastructure
info = toi_info.join(tic_info, on="TIC", how="inner", lsuffix="", rsuffix="_2")
comb_info = info.join(
    observations, 
    on="source_id", 
    lsuffix="", 
    rsuffix="_2", 
    how="inner")

# Intitialise limb darkening coefficient columns
ldc_cols = ["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]

# Set a default period uncertainty of 10 sec if we don't have one
mean_e_period = 1 / 3600 / 24 * 10

# Default (conservative, wide) duration if we don't have one
default_duration = 4

# Amount of lightcurve to fit to in units of transit duration
n_trans_dur = 1.2

# Fit options
base_cadence = 2 / 60 / 24  # Assume 2 min base cadence meaning bin 20 sec data
binsize_fit = 2
bin_lightcurve_fit = True
break_tol_days = 12/24

# Period fitting/transit fitting
fitting_iterations = 3
fit_for_period = True
fit_for_t0 = False
dt_step_fac = 5
do_period_fit_plot = False
n_trans_dur_period_fit = 4

# Min window size for flattening in days
t_min = 12/24
force_window_length_to_min = True

# Plotting settings
rasterized = True
plotting_bin_fac = 10
bin_lightcurve_plot = False
make_paper_plots = True

# List of stars to do a period fit for if we suspect the ExoFOP one
periods_to_fit = [
    73649615
]

# List of bad regions to exclude
bad_btjds_dict = {
    261108236:(1700,2060),  # Bad data
    261257684:(2000,3000),  # Exclude single transit from extended mission
    #259962054:(2000,3000),  # Exclude extended mission
    122613513:(2000,3000),  # Exclude extended mission
}

# And associated update for which sectors we used
force_sectors_for_bad_btjds = {
    261257684:"12-13",  # TOI 904.01
    #259962054:"1-3",    # TOI 203.01
    122613513:"3-4",    # TOI 279.01
}

# List of TOIs to exclude
tois_to_exclude = [
    256.01,         # Only have two transits with TESS
    507.01,         # Actually a double-lined equal mass binary
    969.01,         # Only FFI data
    302.01,         # Only FFI data
    260004324.01,   # Now TOI 704.01 so is a duplicate
    415969908.02,   # Only a single transit
    1080.01,        # Didn't get reduced
    203.01,         # Few transits
    # ------------------------------------------------
    # All beyond this have been removed due to low SNR
    1216.01,        
    253.01,
    260417932.02,
    285.01,
    696.02,
    785.01,
    864.01,
    98796344.02,
]

# Import ballpark periods dataframe
ballpark_periods = pd.read_csv(
    "data/ballpark_long_baseline_periods.tsv",
    delimiter="\t",)
ballpark_periods.set_index("TOI", inplace=True)

# For testing
do_save_and_merge = True

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
# Initialise array to hold dictionaries from fit
all_fits = {}
flat_lc_trends = {}
bm_lc_times_unbinned = {}
bm_lc_fluxes_unbinned = {}
binning_times = []

# Initialise new dataframe to hold results, which will be appended to 
# toi_info and saved once we're done
result_cols = ["sma", "e_sma", "sma_rstar", "e_sma_rstar", "rp_rstar_fit", 
               "e_rp_rstar_fit",  "sma_rstar_fit", "e_sma_rstar_fit", 
               "inclination_fit", "e_inclination_fit", "rp_fit", "e_rp_fit", 
               "window_length", "niters_flat", "rchi2", "period_fit", 
               "e_period_fit", "t0_fit", "e_t0_fit", "sma_rstar_mismatch_flag",
               "sectors", "excluded"]

result_df = pd.DataFrame(
    data=np.full((len(toi_info), len(result_cols)), np.nan), 
    index=toi_info.index, 
    columns=result_cols)

result_df["sma_rstar_mismatch_flag"] = np.zeros(len(toi_info)).astype(bool)
result_df["excluded"] = np.zeros(len(toi_info)).astype(bool)
result_df["sectors"] = np.full(len(toi_info), "").astype(str)

# Given stars may have multiple planets, we have to loop over TOI ID
for toi_i, (toi, toi_row) in enumerate(comb_info.iterrows()):
    # Look up TIC ID in tess_info
    tic = toi_row["TIC"]
    source_id = toi_row["source_id"]

    print("{}\n{}/{} - TOI {} (TIC {})\n{}".format(
        "-"*40, toi_i+1, len(toi_info), toi, tic, "-"*40))
    
    do_skip = False

    # Import light curve
    if light_curves[tic] is None:
        print("Skipping: No light curve\n")
        do_skip = True
    #elif np.isnan(toi_row["Duration (hours)"]):
    #    print("Skipping: No transit duration\n")
    #    do_skip = True
    elif toi in tois_to_exclude:
        print("Skipping: bad TOI\n")
        result_df.loc[toi, "excluded"] = True
        do_skip = True
    #elif toi not in [1067.01]:#elif toi != 468.01:#270.02: #406.01:#
    #   do_skip = True

    # Regardless, save the sectors we're using (or not using) here
    result_df.loc[toi, "sectors"] = sector_dict[tic]

    # We met a skip condition, skip this planet
    if do_skip:
        all_fits[toi] = None
        flat_lc_trends[toi] = None
        bm_lc_times_unbinned[toi] = None
        bm_lc_fluxes_unbinned[toi] = None
        continue

    # Otherwise, we can go ahead
    lightcurve = light_curves[tic]

    # Exclude any bad data
    if tic in bad_btjds_dict:
        mask = np.logical_or(
            lightcurve.time < bad_btjds_dict[tic][0],
            lightcurve.time > bad_btjds_dict[tic][1])

        lightcurve = lightcurve[mask]

        # Save back to dict
        light_curves[tic] = lightcurve

        # And update the sectors we're using (if we've excluded whole sectors)
        if tic in force_sectors_for_bad_btjds.keys():
            result_df.loc[toi, "sectors"] = force_sectors_for_bad_btjds[tic]
    
    # Sort out period (to give us a better first guess)
    if (toi in ballpark_periods.index 
        and ~np.isnan(ballpark_periods.loc[toi]["period_new"])):
        # Set period to this value
        print("Using ballpark period")
        period = ballpark_periods.loc[toi]["period_new"]
    else:
        period = toi_row["Period (days)"]

    # And a period error if we only have nans
    if np.isnan(toi_row["Period error"]):
        e_period = mean_e_period
    else:
        e_period = toi_row["Period error"]
    
    # Add a transit duration if we don't have one, and update table
    if np.isnan(toi_row["Duration (hours)"]):
        duration = default_duration
        toi_info.loc[toi, "Duration (hours)"] = default_duration
    else:
        duration = toi_row["Duration (hours)"]

    # Calculate semi-major axis and scaled semimajor axis
    sma, e_sma, sma_rstar, e_sma_rstar = params.compute_semi_major_axis(
        toi_row["mass_m19"], 
        toi_row["e_mass_m19"],
        period,
        e_period,
        toi_row["radius"],
        toi_row["e_radius"],
        )
    
    # Get mask for all transits with this system
    mask = transit.make_transit_mask_all_periods(
        lightcurve, 
        toi_info, 
        tic)

    # Determine if we need to fit for the period - only do if time baseline is
    # > 1 yr *or* we request it
    total_time = lightcurve.time[-1] - lightcurve.time[0]

    if total_time > 365 or tic in periods_to_fit:
        do_period_and_t0_ls_fit = True
    else:
        do_period_and_t0_ls_fit = False

    # Fit light light curve, but catch and handle any errors with batman
    try:
        opt_res = transit.fit_light_curve(
            lightcurve, 
            toi_row["Transit Epoch (BJD)"],
            toi_row["Transit Epoch error"],
            period,
            e_period,
            duration / 24,  # convert to days 
            toi_row[ldc_cols].values,
            sma_rstar, 
            e_sma_rstar,
            mask,
            ld_model="nonlinear",
            t_min=t_min,
            verbose=True,
            n_trans_dur=n_trans_dur,
            bin_lightcurve=bin_lightcurve_fit,
            do_period_and_t0_ls_fit=do_period_and_t0_ls_fit,
            fitting_iterations=fitting_iterations,
            force_window_length_to_min=force_window_length_to_min,
            break_tol_days=break_tol_days,
            fit_for_period=fit_for_period,
            fit_for_t0=fit_for_t0,
            dt_step_fac=dt_step_fac,
            do_period_fit_plot=do_period_fit_plot,
            n_trans_dur_period_fit=n_trans_dur_period_fit,)

    except transit.BatmanError:
        print("\nBatman failure, skipping\n")
        all_fits[toi] = None
        flat_lc_trends[toi] = None
        bm_lc_times_unbinned[toi] = None
        bm_lc_fluxes_unbinned[toi] = None
        continue

    # Before we go any further, check we actually got uncertainties out - 
    # otherwise this was a singular matrix, and something is wrong. In this
    # case, we shouldn't save any of the fitted parameters
    if np.all(np.isnan(opt_res["std"])):
        continue

    # Calclate the planet radii
    rp_rstar_fit = opt_res["x"][0]
    e_rp_rstar_fit = opt_res["std"][0]

    r_e = const.R_sun.si.value / const.R_earth.si.value
    
    r_p = rp_rstar_fit * toi_row["radius"] * r_e
    e_r_p = r_p * np.sqrt(
        (toi_row["e_radius"]/toi_row["radius"])**2
        + (e_rp_rstar_fit/rp_rstar_fit)**2)

    # Record fit dictionaries
    all_fits[toi] = opt_res

    # Record flattened lightcurve trends, and batman model fluxes
    flat_lc_trends[toi] = opt_res["flat_lc_trend"]
    bm_lc_times_unbinned[toi] = opt_res["folded_lc"].time
    bm_lc_fluxes_unbinned[toi] = opt_res["bm_lightcurve"]

    # Save calculated params + uncertainties
    result_df.loc[toi,["sma", "e_sma"]] = [sma, e_sma]
    result_df.loc[toi, ["sma_rstar", "e_sma_rstar"]] = [sma_rstar, e_sma_rstar]
    result_df.loc[toi, ["rp_fit", "e_rp_fit"]] = [r_p, e_r_p]

    # Save fitted parameters
    param_cols = ["rp_rstar_fit", "sma_rstar_fit", "inclination_fit"]
    result_df.loc[toi, param_cols] = opt_res["x"]

    e_param_cols = ["e_rp_rstar_fit", "e_sma_rstar_fit", "e_inclination_fit"]
    result_df.loc[toi, e_param_cols] = opt_res["std"]

    # Now set the period. We prioritise the fitted period, but we also want to
    # record a period from 'ballpark_periods' in the case where we didn't fit,
    # but didn't use the TESS one either. Set to nan otherwise.
    if ~np.isnan(opt_res["period_fit"]):
        result_df.loc[toi, "period_fit"] = opt_res["period_fit"]
        result_df.loc[toi, "e_period_fit"] = opt_res["e_period_fit"]
    elif period != toi_row["Period (days)"]:
        result_df.loc[toi, "period_fit"] = period
        result_df.loc[toi, "e_period_fit"] = e_period
    else:
        result_df.loc[toi, "period_fit"] = np.nan
        result_df.loc[toi, "e_period_fit"] = np.nan

    result_df.loc[toi, "t0_fit"] = opt_res["t0_fit"]
    result_df.loc[toi, "e_t0_fit"] = opt_res["e_t0_fit"] 

    # Save details of flattening + fit
    result_df.loc[toi, "rchi2"] = opt_res["rchi2"]
    result_df.loc[toi, "window_length"] = opt_res["window_length"]
    result_df.loc[toi, "niters_flat"] = opt_res["niters_flat"]

    # Save a flag if SMA are very different
    sma_diff = np.abs(sma_rstar-opt_res["x"][1])
    e_sma_diff = (e_sma_rstar**2 + opt_res["std"][1]**2)**0.5
    e_flag = sma_diff > e_sma_diff
    result_df.loc[toi, "sma_rstar_mismatch_flag"] = e_flag

    print("\n---Result---")
    print("Rp/R* = {:0.5f} +/- {:0.5f},".format(
            result_df.loc[toi]["rp_rstar_fit"], 
            result_df.loc[toi]["e_rp_rstar_fit"]), 
          "\na = {:0.2f} +/- {:0.5f},".format(
            result_df.loc[toi]["sma_rstar_fit"], 
            result_df.loc[toi]["e_sma_rstar_fit"]), 
          "\ni = {:0.2f} +/- {:0.5f}\n".format(
            result_df.loc[toi]["inclination_fit"], 
            result_df.loc[toi]["e_inclination_fit"]),)

    print("Final rchi^2 = {:0.5f}".format(opt_res["rchi2"]))

# Concatenate our two dataframes
toi_info = pd.concat((toi_info, result_df), axis=1)

nan_mask = np.isnan(toi_info["rp_fit"])
small_mask = toi_info["rp_fit"] < 0.1

print("{} TOI fits were nan".format(len(toi_info[nan_mask])))
print("TOIs: {}\n".format(str(list(toi_info.index[nan_mask]))))

print("{} TOI fits unfeasibly small".format(len(toi_info[small_mask])))
print("TOIs: {}\n".format(str(list(toi_info.index[small_mask]))))

# Save results
if do_save_and_merge:
    toi_info.index.set_names(["TOI"], inplace=True)
    utils.save_fits_table("TRANSIT_FITS", toi_info, "tess")

# Conclude timing
time_elapsed = datetime.now() - start_time
print("Fitting duration (hh:mm:ss.ms) {}".format(time_elapsed))

# Import the other set of light curves that have been more highly binned for
# the purpose of plotting
print("Importing light curves for plotting")
binned_light_curves, _ = transit.load_all_light_curves(
    tic_info.index.values,
    bin_fac=plotting_bin_fac,)

# Plotting
pplt.plot_all_lightcurve_fits(
    light_curves,
    toi_info,
    tic_info,
    observations,
    binned_light_curves=binned_light_curves,
    break_tol_days=break_tol_days,
    flat_lc_trends=flat_lc_trends,
    bm_lc_times_unbinned=bm_lc_times_unbinned,
    bm_lc_fluxes_unbinned=bm_lc_fluxes_unbinned,
    t_min=t_min,
    force_window_length_to_min=force_window_length_to_min,
    rasterized=rasterized,
    make_paper_plots=make_paper_plots,)

if do_save_and_merge:
    pplt.merge_spectra_pdfs(
        "plots/lc_diagnostics/*diagnostic.pdf", 
    "plots/lc_diagnostics.pdf",)