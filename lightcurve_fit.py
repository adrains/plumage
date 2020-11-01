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
tic_info = utils.load_info_cat(remove_fp=True, only_observed=True).reset_index()
tic_info.set_index("TIC", inplace=True)

# Load NASA ExoFOP info on TOIs
toi_info = utils.load_exofop_toi_cat(do_ctoi_merge=True)

# Load in info on observations and fitting results
observations = utils.load_fits_table("OBS_TAB", label, path=spec_path)

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

# Inflate the error on our fitted stellar radii
e_radius_mult = 10

# Min window size for flattening in days
t_min = 8/24

# Plotting settings
binsize = 10
bin_lightcurve = True
break_tolerance = 10

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
# Initialise array to hold dictionaries from fit
all_fits = {}

# Initialise new dataframe to hold results, which will be appended to 
# toi_info and saved once we're done
result_cols = ["sma", "e_sma", "rp_rstar_fit", "e_rp_rstar_fit", 
               "sma_rstar_fit", "e_sma_rstar_fit", "inclination_fit", 
               "e_inclination_fit", "rp_fit", "e_rp_fit", "window_length", 
               "niters_flat"]

result_df = pd.DataFrame(
    data=np.full((len(toi_info), len(result_cols)), np.nan), 
    index=toi_info.index, 
    columns=result_cols)

# Given stars may have multiple planets, we have to loop over TOI ID
for toi_i, (toi, toi_row) in enumerate(comb_info.iterrows()):
    # Look up TIC ID in tess_info
    tic = toi_row["TIC"]
    source_id = toi_row["source_id"]

    print("{}\n{}/{} - TOI {} (TIC {})\n{}".format(
        "-"*40, toi_i+1, len(toi_info), toi, tic, "-"*40))
    
    if light_curves[tic] is None:
        print("Skipping: No light curve\n")
        all_fits[toi] = None
        continue
    elif np.isnan(toi_row["Duration (hours)"]):
        print("Skipping: No transit duration\n")
        all_fits[toi] = None
        continue
    #elif toi != 177.01:#270.02: #406.01:#
        #continue
    else:
        # Bin lightcurve
        if bin_lightcurve:
            lightcurve = light_curves[tic].remove_nans().bin(binsize=binsize)
        else:
            lightcurve = light_curves[tic].remove_nans()

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
        toi_row["e_radius"]*e_radius_mult,
        )
    
    # Get mask for all transits with this system
    mask = transit.make_transit_mask_all_periods(
        lightcurve, 
        toi_info, 
        tic)

    # Fit light light curve, but catch and handle any errors with batman
    try:
        opt_res = transit.fit_light_curve(
            lightcurve, 
            toi_row["Epoch (BJD)"], 
            toi_row["Period (days)"], 
            toi_row["Duration (hours)"] / 24,  # convert to days 
            toi_row[ldc_cols].values,
            sma_rstar, 
            e_sma_rstar,
            mask,
            ld_model="nonlinear",
            t_min=t_min,
            verbose=True,
            n_trans_dur=n_trans_dur,
            binsize=binsize,
            bin_lightcurve=bin_lightcurve,)

    except transit.BatmanError:
        print("\nBatman failure, skipping\n")
        all_fits[toi] = None
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
    e_r_p = r_p * np.sqrt((e_radius_mult*toi_row["e_radius"]/toi_row["radius"])**2
                           + (e_rp_rstar_fit/rp_rstar_fit)**2) * r_e

    # Record fit dictionaries
    all_fits[toi] = opt_res

    # Save calculated params + uncertainties
    result_df.loc[toi][["sma", "e_sma"]] = [sma, e_sma]
    result_df.loc[toi][["sma_rstar_fit", "e_sma_rstar_fit"]] = [sma_rstar, e_sma_rstar]
    result_df.loc[toi][["rp_fit", "e_rp_fit"]] = [r_p, e_r_p]

    # Save fitted parameters
    param_cols = ["rp_rstar_fit", "sma_rstar_fit", "inclination_fit"]
    result_df.loc[toi][param_cols] = opt_res["x"]

    e_param_cols = ["e_rp_rstar_fit", "e_sma_rstar_fit", "e_inclination_fit"]
    result_df.loc[toi][e_param_cols] = opt_res["std"]

    # Save details of flattening
    result_df.loc[toi]["window_length"] = opt_res["window_length"]
    result_df.loc[toi]["niters_flat"] = opt_res["niters_flat"]

    print("\n---Result---")
    print("Rp/R* = {:0.5f} +/- {:0.5f},".format(
            result_df.loc[toi]["rp_rstar_fit"], 
            result_df.loc[toi]["e_rp_rstar_fit"]), 
          "\na = {:0.2f} +/- {:0.2f},".format(
            result_df.loc[toi]["sma_rstar_fit"], 
            result_df.loc[toi]["e_sma_rstar_fit"]), 
          "\ni = {:0.2f} +/- {:0.2f}\n".format(
            result_df.loc[toi]["inclination_fit"], 
            result_df.loc[toi]["e_inclination_fit"]),)

# Concatenate our two dataframes
toi_info = pd.concat((toi_info, result_df), axis=1)

nan_mask = np.isnan(toi_info["rp_fit"])
small_mask = toi_info["rp_fit"] < 0.1

print("{} TOI fits were nan".format(len(toi_info[nan_mask])))
print("TOIs: {}\n".format(str(list(toi_info.index[nan_mask]))))

print("{} TOI fits unfeasibly small".format(len(toi_info[small_mask])))
print("TOIs: {}\n".format(str(list(toi_info.index[small_mask]))))

# Plotting
pplt.plot_all_lightcurve_fits(
    light_curves,
    toi_info,
    tic_info,
    observations,
    binsize=binsize,
    bin_lightcurve=bin_lightcurve,
    break_tolerance=break_tolerance,)

pplt.merge_spectra_pdfs(
    "plots/lc_diagnostics/*.pdf", 
    "plots/lc_diagnostics.pdf",) 

# Save results
toi_info.index.set_names(["TOI"], inplace=True)
utils.save_fits_table("TRANSIT_FITS", toi_info, "tess")
