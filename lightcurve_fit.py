"""Script to fit saved TESS light curves using previously determined values for
stellar radius, mass, limb-darkening (from stellar params), and transit params
per values from NASA ExoFOP
"""
import numpy as np
import pandas as pd
import plumage.utils as utils
import plumage.transits as transit
import plumage.parameters as params
import plumage.plotting as pplt
import astropy.constants as const

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

# Load in lightcurves
light_curves = transit.load_all_light_curves(tic_info.index.values)

logg_col = "logg_synth"
#logg_col = "logg_m19"

# Temporary join to get combined info in single datastructure
info = toi_info.join(tic_info, on="TIC", how="inner", lsuffix="", rsuffix="_2")
#info.reset_index(inplace=True)
comb_info = info.join(observations, on="source_id", lsuffix="", rsuffix="_2", how="inner")

# Intitialise limb darkening coefficient columns
ldc_cols = ["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]

# Set a default period uncertainty if we don't have one
mean_e_period = 10 / 60 / 24

# Amount of lightcurve to fit to inn units of transit duration
n_trans_dur = 1.2

# Fit options
binsize_fit = 2
bin_lightcurve_fit = True
break_tol_days = 12/24

# Period fitting/transit fitting
fitting_iterations = 3
fit_for_period = True
fit_for_t0 = False
dt_step_fac = 2
do_period_fit_plot = False
n_trans_dur_period_fit = 4

# Min window size for flattening in days
t_min = 12/24
force_window_length_to_min = True

# Plotting settings
rasterized = True
binsize_plot = 10
bin_lightcurve_plot = False
make_paper_plots = True

# List of bad regions to exclude
bad_btjds_dict = {
    261108236:(1700,2060)
}

# List of TOIs to exclude
tois_to_exclude = [
    256.01,     # Only have one transit with TESS
]
# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
# Initialise array to hold dictionaries from fit
all_fits = {}
flat_lc_trends = {}
bm_lc_times_unbinned = {}
bm_lc_fluxes_unbinned = {}

# Initialise new dataframe to hold results, which will be appended to 
# toi_info and saved once we're done
result_cols = ["sma", "e_sma", "sma_rstar", "e_sma_rstar", "rp_rstar_fit", 
               "e_rp_rstar_fit",  "sma_rstar_fit", "e_sma_rstar_fit", 
               "inclination_fit", "e_inclination_fit", "rp_fit", "e_rp_fit", 
               "window_length", "niters_flat", "rchi2", "period_fit", 
               "e_period_fit", "t0_fit", "e_t0_fit", "sma_rstar_mismatch_flag"]

result_df = pd.DataFrame(
    data=np.full((len(toi_info), len(result_cols)), np.nan), 
    index=toi_info.index, 
    columns=result_cols)

# To save time when plotting, save the binned lightcurves
binned_lightcurves = light_curves.copy()

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
    elif np.isnan(toi_row["Duration (hours)"]):
        print("Skipping: No transit duration\n")
        do_skip = True
    elif toi in tois_to_exclude:
        print("Skipping: bad TOI\n")
        do_skip = True
    #elif toi != 700.01:#elif toi != 468.01:#270.02: #406.01:#
    #   do_skip = True

    # We met a skip condition, skip this planet
    if do_skip:
        all_fits[toi] = None
        flat_lc_trends[toi] = None
        bm_lc_times_unbinned[toi] = None
        bm_lc_fluxes_unbinned[toi] = None
        continue

    # Otherwise, we can go ahead and import the light curve
    lightcurve = light_curves[tic]

    # Exclude any bad data
    if tic in bad_btjds_dict:
        mask = np.logical_or(
            lightcurve.time < bad_btjds_dict[tic][0],
            lightcurve.time > bad_btjds_dict[tic][1])

        lightcurve = lightcurve[mask]

        # Save back to dict
        light_curves[tic] = lightcurve

    # Bin and remove nans
    if bin_lightcurve_fit:
        lightcurve = lightcurve.remove_nans().bin(binsize=binsize_fit)
    else:
        lightcurve = lightcurve.remove_nans()

    binned_lightcurves[tic] = lightcurve

    # Calculate semi-major axis and scaled semimajor axis
    if np.isnan(toi_row["Period error"]):
        e_period = mean_e_period
    else:
        e_period = toi_row["Period error"]

    sma, e_sma, sma_rstar, e_sma_rstar = params.compute_semi_major_axis(
        toi_row["mass_m19"], 
        toi_row["e_mass_m19"],
        toi_row["Period (days)"],
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
    # > 1 yr
    if (lightcurve.time[-1] - lightcurve.time[0]) > 365:
        do_period_and_t0_ls_fit = True
    else:
        do_period_and_t0_ls_fit = False

    # Fit light light curve, but catch and handle any errors with batman
    try:
        opt_res = transit.fit_light_curve(
            lightcurve, 
            toi_row["Transit Epoch (BJD)"],
            toi_row["Transit Epoch error"],
            toi_row["Period (days)"],
            toi_row["Period error"],
            toi_row["Duration (hours)"] / 24,  # convert to days 
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
    result_df.loc[toi][["sma", "e_sma"]] = [sma, e_sma]
    result_df.loc[toi][["sma_rstar", "e_sma_rstar"]] = [sma_rstar, e_sma_rstar]
    result_df.loc[toi][["rp_fit", "e_rp_fit"]] = [r_p, e_r_p]

    # Save fitted parameters
    param_cols = ["rp_rstar_fit", "sma_rstar_fit", "inclination_fit"]
    result_df.loc[toi][param_cols] = opt_res["x"]

    e_param_cols = ["e_rp_rstar_fit", "e_sma_rstar_fit", "e_inclination_fit"]
    result_df.loc[toi][e_param_cols] = opt_res["std"]

    result_df.loc[toi]["period_fit"] = opt_res["period_fit"]
    result_df.loc[toi]["e_period_fit"] = opt_res["e_period_fit"]

    result_df.loc[toi]["t0_fit"] = opt_res["t0_fit"]
    result_df.loc[toi]["e_t0_fit"] = opt_res["e_t0_fit"] 

    # Save details of flattening + fit
    result_df.loc[toi]["rchi2"] = opt_res["rchi2"]
    result_df.loc[toi]["window_length"] = opt_res["window_length"]
    result_df.loc[toi]["niters_flat"] = opt_res["niters_flat"]

    # Save a flag if SMA are very different
    sma_diff = np.abs(sma_rstar-opt_res["x"][1])
    e_sma_diff = (e_sma_rstar**2 + opt_res["std"][1]**2)**0.5
    e_flag = sma_diff > e_sma_diff
    result_df.loc[toi]["sma_rstar_mismatch_flag"] = int(e_flag)

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
toi_info.index.set_names(["TOI"], inplace=True)
utils.save_fits_table("TRANSIT_FITS", toi_info, "tess")

# If we want to bin the plots, send the unbinned lightcurves. If not binning,
# send our light curves through with whatever binning we used for the fit
if not bin_lightcurve_plot or binsize_fit == binsize_fit:
    all_lcs = binned_lightcurves
else:
    all_lcs = light_curves

# Plotting
pplt.plot_all_lightcurve_fits(
    all_lcs,
    toi_info,
    tic_info,
    observations,
    binsize=binsize_plot,
    break_tol_days=break_tol_days,
    flat_lc_trends=flat_lc_trends,
    bm_lc_times_unbinned=bm_lc_times_unbinned,
    bm_lc_fluxes_unbinned=bm_lc_fluxes_unbinned,
    bin_lightcurve=bin_lightcurve_plot,
    t_min=t_min,
    force_window_length_to_min=force_window_length_to_min,
    rasterized=rasterized,
    make_paper_plots=make_paper_plots,)

pplt.merge_spectra_pdfs(
    "plots/lc_diagnostics/*diagnostic.pdf", 
    "plots/lc_diagnostics.pdf",) 