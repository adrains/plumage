"""Script to fit saved TESS light curves using previously determined values for
stellar radius, mass, limb-darkening (from stellar params), and transit params
per values from NASA ExoFOP
"""
import numpy as np
import pandas as pd
import plumage.utils as utils
import plumage.transits as transit
import plumage.parameters as params
import astropy.constants as const

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
label = "TESS"
spec_path =  "spectra"

# Load in literature info for our stars
tess_info = utils.load_info_cat(remove_fp=True, only_observed=True)

# Load NASA ExoFOP info on TOIs
toi_info = utils.load_exofop_toi_cat()

# Load in info on observations and fitting results
observations = utils.load_fits_obs_table(label, path=spec_path) 

# Load in lightcurves
light_curves = transit.load_all_light_curves(tess_info["TIC"].values)  

# Determine limb darkening coefficients for all stars and save
# TODO: current implementation does not take into account [Fe/H]
ldc_ak = params.get_claret17_limb_darkening_coeff(
    observations["teff_synth"], 
    observations["logg_synth"], 
    observations["feh_synth"])

ldd_cols = ["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]

for ldc_i, ldd_col in enumerate(ldd_cols):
    observations[ldd_col] = ldc_ak[:,ldc_i]

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
# Initialise array to hold dictionaries from fit
all_fits = {}

# Initialise new dataframe to hold results, which will be appended to 
# toi_info and saved once we're done
result_cols = ["sma", "e_sma", "rp_rstar_fit", "e_rp_rstar_fit", 
               "sma_rstar_fit", "e_sma_rstar_fit", "inclination_fit", 
               "e_inclination_fit", "radius_fit", "e_radius_fit", ]

result_df = pd.DataFrame(
    data=np.full((len(toi_info), len(result_cols)), np.nan), 
    index=toi_info.index, 
    columns=result_cols)

# Given stars may have multiple planets, we have to loop over TOI ID
for toi_i, (toi, toi_row) in enumerate(toi_info.iterrows()):
    # Look up TIC ID in tess_info
    tic = toi_row["TIC"]

    print("{}\n{}/{} - TOI {} (TIC {})\n{}".format(
        "-"*40, toi_i+1, len(toi_info), toi, tic, "-"*40))

    # Get the literature info
    tic_info = tess_info[tess_info["TIC"]==tic].iloc[0]

    source_id = tic_info["source_id"]

    obs_info = observations[observations["uid"]==source_id]

    if len(obs_info) == 0:
        print("No source ID")
        all_fits[toi] = None
        continue
    else:
        obs_info = obs_info.iloc[0]
    
    if light_curves[tic] is None:
        print("No light curve")
        all_fits[toi] = None
        continue

    # Calculate semi-major axis
    sma, e_sma = params.compute_semi_major_axis(
        tic_info["radii_m19"], 
        toi_row["Period (days)"])

    # Fit light light curve, but catch and handle any errors with batman
    try:
        opt_res = transit.fit_light_curve(
            light_curves[tic], 
            toi_row["Epoch (BJD)"], 
            toi_row["Period (days)"], 
            toi_row["Duration (hours)"] / 24,  # convert to days 
            obs_info[ldd_cols].values,
            ld_model="nonlinear")

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
    radius = (opt_res["x"][0] * tic_info["radii_m19"]
              * const.R_sun.si.value / const.R_earth.si.value)
    e_radius = np.nan

    # Record fit dictionaries
    all_fits[toi] = opt_res

    # Save calculated params + uncertainties
    result_df.loc[toi][["sma", "e_sma"]] = [sma, e_sma]
    result_df.loc[toi][["radius_fit", "e_radius_fit"]] = [radius, e_radius]

    # Save fitted parameters
    param_cols = ["rp_rstar_fit", "sma_rstar_fit", "inclination_fit"]
    result_df.loc[toi][param_cols] = opt_res["x"]

    e_param_cols = ["e_rp_rstar_fit", "e_sma_rstar_fit", "e_inclination_fit"]
    result_df.loc[toi][e_param_cols] = opt_res["std"]

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

toi_info.to_csv("data/exofop_tess_tois2.csv", quoting=1)
