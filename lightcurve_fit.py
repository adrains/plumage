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
tic_info = utils.load_info_cat(remove_fp=True, only_observed=True).reset_index()
tic_info.set_index("TIC", inplace=True)

# Load NASA ExoFOP info on TOIs
toi_info = utils.load_exofop_toi_cat()

# Load in info on observations and fitting results
# TODO propagate this change through the code
observations = utils.load_fits_obs_table(label, path=spec_path)
observations.rename(columns={"uid":"source_id"}, inplace=True)
observations.set_index("source_id", inplace=True)

# Load in lightcurves
light_curves = transit.load_all_light_curves(tic_info.index.values)

logg_col = "logg_synth"
logg_col = "logg_m19"

# Temporary join to get combined info in single datastructure
info = toi_info.join(tic_info, on="TIC", how="inner", lsuffix="", rsuffix="_2")
#info.reset_index(inplace=True)
comb_info = info.join(observations, on="source_id", lsuffix="", rsuffix="_2", how="inner")

# Determine limb darkening coefficients for all stars and save
# TODO: current implementation does not take into account [Fe/H]
ldc_ak = params.get_claret17_limb_darkening_coeff(
    comb_info["teff_synth"], 
    comb_info[logg_col], 
    comb_info["feh_synth"])

ldd_cols = ["ldc_a1", "ldc_a2", "ldc_a3", "ldc_a4"]

for ldc_i, ldd_col in enumerate(ldd_cols):
    comb_info[ldd_col] = ldc_ak[:,ldc_i]

# -----------------------------------------------------------------------------
# Do fitting
# -----------------------------------------------------------------------------
# Initialise array to hold dictionaries from fit
all_fits = {}

# Initialise new dataframe to hold results, which will be appended to 
# toi_info and saved once we're done
result_cols = ["sma", "e_sma", "rp_rstar_fit", "e_rp_rstar_fit", 
               "sma_rstar_fit", "e_sma_rstar_fit", "inclination_fit", 
               "e_inclination_fit", "rp_fit", "e_rp_fit", ]

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
        print("No light curve")
        all_fits[toi] = None
        continue

    # Calculate semi-major axis and scaled semimajor axis
    sma, e_sma, sma_rstar, e_sma_rstar = params.compute_semi_major_axis(
        toi_row["mass_m19"], 
        toi_row["e_mass_m19"],
        toi_row["Period (days)"],
        toi_row["Period error"],
        toi_row["radius"],
        toi_row["e_radius"],
        )

    # Get mask for all transits with this system
    mask = transit.make_transit_mask_all_periods(
        light_curves[tic].remove_nans(), 
        toi_info, 
        tic)

    # Fit light light curve, but catch and handle any errors with batman
    try:
        opt_res = transit.fit_light_curve(
            light_curves[tic].remove_nans(), 
            toi_row["Epoch (BJD)"], 
            toi_row["Period (days)"], 
            toi_row["Duration (hours)"] / 24,  # convert to days 
            toi_row[ldd_cols].values,
            sma_rstar, 
            e_sma_rstar,
            mask,
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
    radius = (opt_res["x"][0] * toi_row["radius"]
              * const.R_sun.si.value / const.R_earth.si.value)
    e_radius = (opt_res["std"][0] * toi_row["radius"]
              * const.R_sun.si.value / const.R_earth.si.value)

    # Record fit dictionaries
    all_fits[toi] = opt_res

    # Save calculated params + uncertainties
    result_df.loc[toi][["sma", "e_sma"]] = [sma, e_sma]
    result_df.loc[toi][["rp_fit", "e_rp_fit"]] = [radius, e_radius]

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

# Save results
utils.save_fits_table("TRANSIT_FITS", toi_info, "tess")
