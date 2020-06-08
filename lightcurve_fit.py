"""Script to fit saved TESS light curves using previously determined values for
stellar radius, mass, limb-darkening (from stellar params), and transit params
per values from NASA ExoFOP
"""
import numpy as np
import plumage.utils as utils
import plumage.transits as transit
import plumage.parameters as params

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
label = "TESS"
spec_path =  "spectra"

# Imports
tess_info = utils.load_info_cat(remove_fp=True, only_observed=True)

# Load ExoFOP info
exofop_info = utils.load_exofop_toi_cat()

observations = utils.load_fits_obs_table(label, path=spec_path) 

# Load in lightcurves
light_curves = transit.load_all_light_curves(tess_info["TIC"].values)  

# Determine limb darkening coefficients for all stars
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
# Initialise arrays to hold results
all_fits = []
fitted_sma = []
fitted_sma = []

# Given stars may have multiple planets, we have to loop over TOI ID
for toi_i, toi_info in exofop_info.iterrows():
    # Look up TIC ID in tess_info
    tic = toi_info["TIC"]

    ln = "-"*40
    print("{}\n{} - {}\n{}".format(ln, toi_i, tic,ln))

    # Get the literature info
    tic_info = tess_info[tess_info["TIC"]==tic].iloc[0]

    source_id = tic_info["source_id"]

    obs_info = observations[observations["uid"]==source_id]

    if len(obs_info) == 0:
        print("No source ID")
        continue
    else:
        obs_info = obs_info.iloc[0]
    
    if light_curves[tic] is None:
        print("No light curve")
        continue

    # Calculate semi-major axis
    sma = params.compute_semi_major_axis(
        tic_info["radii_m19"], 
        toi_info["Period (days)"])

    # Fit light light curve
    opt_res = transit.fit_light_curve(
        light_curves[tic], 
        toi_info["Epoch (BJD)"], 
        toi_info["Period (days)"], 
        toi_info["Duration (hours)"] / 24,  # convert to days 
        obs_info[ldd_cols].values,
        ld_model="nonlinear")

    all_fits.append(opt_res)
# Calculate final planet parameters
