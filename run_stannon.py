"""Script to train and cross validate a Cannon model
"""
import numpy as np
import pandas as pd
import plumage.utils as utils
import plumage.spectra as spec
import plumage.synthetic as synth
import stannon.stannon as stannon
import stannon.plotting as splt
from numpy.polynomial.polynomial import Polynomial

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
suppress_output = True
use_br_combined = False
normalise_spectra = True
add_photometry = False
do_cross_validation = True

# Whether to do sigma clipping using trained Cannon model
do_iterative_bad_px_masking = True
flux_sigma_to_clip = 5

# Don't consider wavelengths below this when normalising spectra
wl_min_normalisation = 4000

wl_min_model = 4000
wl_max_model = 7000

poly_order = 4
#model_type = "basic"

model_type = "label_uncertainties"
use_label_uniform_variances = False

model_save_path = "spectra"
std_label = "cannon"

# Whether to fit for abundances from Montes+18. Not recommended to fit > 1-2.
# Available options: Na, Mg, Al, Si, Ca, Sc, Ti, V, Cr, Mn, Co, Ni
# Select as e.g.["X_H",..] or leave empty to not use abundances.
abundance_labels = []

label_names = ["teff", "logg", "feh"] + abundance_labels
n_labels = len(label_names)

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
# Import literature info for both standards and TESS targets
std_info = utils.load_info_cat(
    "data/std_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)
tess_info = utils.load_info_cat(
    "data/tess_info.tsv",
    use_mann_code_for_masses=False,
    in_paper=True,
    only_observed=True,
    do_extinction_correction=False,)

# Load results tables for both standards and TESS targets
obs_std = utils.load_fits_table("OBS_TAB", std_label, path="spectra")
obs_tess = utils.load_fits_table("OBS_TAB", "tess", path="spectra")

# Load in RV corrected standard spectra
wave_std_br = utils.load_fits_image_hdu("rest_frame_wave", std_label, arm="br")
spec_std_br = utils.load_fits_image_hdu("rest_frame_spec", std_label, arm="br")
e_spec_std_br = utils.load_fits_image_hdu("rest_frame_sigma", std_label, arm="br")

spec_std_br, e_spec_std_br = spec.normalise_spectra(
    wave_std_br,
    spec_std_br,
    e_spec_std_br,
    poly_order=poly_order,
    wl_min=wl_min_normalisation,)

# Load in RV corrected TESS spectra
wave_tess_br = utils.load_fits_image_hdu("rest_frame_wave", "tess", arm="br")
spec_tess_br = utils.load_fits_image_hdu("rest_frame_spec", "tess", arm="br")
e_spec_tess_br = utils.load_fits_image_hdu("rest_frame_sigma", "tess", arm="br")

spec_tess_br, e_spec_tess_br = spec.normalise_spectra(
    wave_tess_br,
    spec_tess_br,
    e_spec_tess_br,
    poly_order=poly_order,
    wl_min=wl_min_normalisation,)

# Table joins
obs_join = obs_std.join(std_info, "source_id", rsuffix="_info")

obs_join_tess = obs_tess.join(tess_info, "source_id", rsuffix="_info")

# And crossmatch this with our set of CPM abundances
montes18_abund = pd.read_csv(
    "data/montes18_primaries.tsv",
    delimiter="\t",
    dtype={"source_id":str},
    na_filter=False)
montes18_abund.set_index("source_id", inplace=True)
obs_join = obs_join.join(montes18_abund, "source_id", rsuffix="_m18")

# Import Montes+18 abundance trends
montes18_abund_trends = pd.read_csv("data/montes18_abundance_trends.csv") 

#------------------------------------------------------------------------------
# Setup training set
#------------------------------------------------------------------------------
def prepare_labels(
    obs_join,
    n_labels=3,
    e_teff_quad=60,
    max_teff=4200,
    abundance_labels=[],
    abundance_trends=montes18_abund_trends,):
    """Prepare our set of training labels using our hierarchy of parameter 
    source preferences.

    Teff: Prefer interferometric measurements, otherwise take the uniform Teff
    scale from Rains+21 which has been benchmarked to the interferometric Teff
    scale. Add Rains+21 uncertainties in quadrature with standard M+15 
    uncertainties to ensure that interferometric benchmarks are weighted more
    highly. Enforce a max Teff limit to avoid warmer stars.

    Logg: uniform Logg from Rains+21 (Mann+15 intial guess, updated from fit)

    [Fe/H]: Prefer CPM binary benchmarks, then M+15, then RA+12, then [Fe/H]
    from other NIR relations (e.g. T+15, G+14), then just default for Solar 
    Neighbourhood with large uncertainties.
    """
    # Intialise mask
    std_mask = np.full(len(obs_join), True)

    # Initialise label vector
    label_values = np.full( (len(obs_join), n_labels), np.nan)
    label_sigma = np.full( (len(obs_join), n_labels), np.nan)

    # Go through one star at a time and select labels
    for star_i, (source_id, star_info) in enumerate(obs_join.iterrows()):
        # Only accept properly vetted stars with consistent Teffs
        if not star_info["in_paper"]:
            std_mask[star_i] = False
            continue

        # Only accept interferometric, M+15, RA+12, NIR other, & CPM standards
        elif not (~np.isnan(star_info["teff_int"]) 
            or ~np.isnan(star_info["teff_m15"])
            or ~np.isnan(star_info["teff_ra12"])
            or ~np.isnan(star_info["feh_nir"])
            or ~np.isnan(star_info["feh_cpm"])):
            std_mask[star_i] = False
            continue
        
        # Enforce our max temperature for interferometric standards
        elif star_info["teff_int"] > max_teff:
            std_mask[star_i] = False
            continue

        # Teff: interferometric > Rains+21
        if not np.isnan(star_info["teff_int"]):
            label_values[star_i, 0] = star_info["teff_int"]
            label_sigma[star_i, 0] = star_info["e_teff_int"]

        else:
            label_values[star_i, 0] = star_info["teff_synth"]
            label_sigma[star_i, 0] = (
                star_info["e_teff_synth"]**2 + e_teff_quad**2)**0.5

        # logg: Rains+21
        label_values[star_i, 1] = star_info["logg_synth"]
        label_sigma[star_i, 1] = star_info["e_logg_synth"]

        # [Fe/H]: CPM > M+15 > RA+12 > NIR other > Rains+21 > default
        if not np.isnan(star_info["feh_cpm"]):
            label_values[star_i, 2] = star_info["feh_cpm"]
            label_sigma[star_i, 2] = star_info["e_feh_cpm"]

        elif not np.isnan(star_info["feh_m15"]):
            label_values[star_i, 2] = star_info["feh_m15"]
            label_sigma[star_i, 2] = star_info["e_feh_m15"]

        elif not np.isnan(star_info["feh_ra12"]):
            label_values[star_i, 2] = star_info["feh_ra12"]
            label_sigma[star_i, 2] = star_info["e_feh_ra12"]

        elif not np.isnan(star_info["feh_nir"]):
            label_values[star_i, 2] = star_info["feh_nir"]
            label_sigma[star_i, 2] = star_info["e_feh_nir"]

        elif not np.isnan(star_info["phot_feh"]):
            label_values[star_i, 2] = star_info["phot_feh"]
            label_sigma[star_i, 2] = star_info["e_phot_feh"]

        else:
            label_values[star_i, 2] = -0.14 # Mean for Solar Neighbourhood
            label_sigma[star_i, 2] = 2.0    # Default uncertainty

        # Note the adopted [Fe/H]
        feh_adopted = label_values[star_i, 2]

        # Other abundances
        for abundance_i, abundance in enumerate(abundance_labels):
            label_i = 3 + abundance_i

            # Use the abundance if we have it
            if not np.isnan(star_info[abundance]):
                label_values[star_i, label_i] = star_info[abundance]
                label_sigma[star_i, label_i] = star_info["e{}".format(abundance)]
            
            # Otherwise default to the solar neighbourhood abundance trend
            else:
                poly = Polynomial(abundance_trends[abundance].values)
                X_H = poly(feh_adopted)
                label_values[star_i, label_i] = X_H
                label_sigma[star_i, label_i] = 2.0

    return label_values, label_sigma, std_mask

# Prepare our labels
label_values_all, label_sigma_all, std_mask = prepare_labels(
    obs_join=obs_join,
    n_labels=n_labels,
    abundance_labels=abundance_labels,
    abundance_trends=montes18_abund_trends)

label_values = label_values_all[std_mask]
label_var = label_sigma_all[std_mask]**0.5

# Test with uniform variances
if use_label_uniform_variances:
    label_var = 1e-3 * np.ones_like(label_values)

# Prepare fluxes
training_set_flux, training_set_ivar, bad_px_mask = stannon.prepare_fluxes(
    spec_std_br[std_mask],
    e_spec_std_br[std_mask],)

wls = wave_std_br

# Construct standard mask to mask emission regions
adopted_wl_mask = spec.make_wavelength_mask(
    wls,
    mask_emission=True,
    mask_sky_emission=False,
    mask_edges=True,)

# Enforce minimum and maximum wavelengths
adopted_wl_mask = adopted_wl_mask * (wls > wl_min_model) * (wls < wl_max_model)

#------------------------------------------------------------------------------
# Photometry (optional)
#------------------------------------------------------------------------------
if add_photometry:
    phot_wls = np.array([
        4500,
        7500,
        6000,
        12350,
        16620,
        21590,
    ])

    abs_mags = np.array([
        "Bp_mag_abs",
        "Rp_mag_abs",
        "G_mag_abs",
        "J_mag_abs",
        "H_mag_abs",
        "K_mag_abs",
        ], dtype=object)

    e_abs_mags = np.array([
        "e_Bp_mag_abs",
        "e_Rp_mag_abs",
        "G_mag_abs",
        "e_J_mag_abs",
        "e_H_mag_abs",
        "e_K_mag_abs",
        ], dtype=object)

    # Add photometry
    wls = np.concatenate([wls, phot_wls])
    training_set_flux = np.concatenate(
        [training_set_flux, obs_join[std_mask][abs_mags].values], axis=1)
    training_set_ivar = np.concatenate(
        [training_set_ivar, 1/obs_join[std_mask][e_abs_mags].values**2], axis=1)

#------------------------------------------------------------------------------
# Make and Train model
#------------------------------------------------------------------------------
print("\n\n", "%"*80, "\n", sep="")
print("\tRunning Cannon model:\n\t", "-"*21, sep="")
print("\tmodel: \t\t\t = {}".format(model_type))
print("\tn px: \t\t\t = {:0.0f}".format(np.sum(adopted_wl_mask)))
print("\tn labels: \t\t = {:0.0f}".format(len(label_names)))
print("\tlabels: \t\t = {}".format(label_names))
print("\tcross validation: \t = {}".format(do_cross_validation))
print("\titerative masking: \t = {}".format(do_iterative_bad_px_masking))
print("\n", "%"*80, "\n\n", sep="")

# Make model
sm = stannon.Stannon(
    training_data=training_set_flux,
    training_data_ivar=training_set_ivar,
    training_labels=label_values, 
    label_names=label_names,
    wavelengths=wls,
    model_type=model_type,
    training_variances=label_var,
    adopted_wl_mask=adopted_wl_mask,
    bad_px_mask=bad_px_mask,)

# Train model
print("\nRunning initial training with {} benchmarks...".format(len(label_values)))
sm.train_cannon_model(suppress_output=suppress_output)

# If we run the iterative bad px masking, train again afterwards
if do_iterative_bad_px_masking:
    print("\nRunning iterative sigma clipping for bad px...")
    sm.make_sigma_clipped_bad_px_mask(flux_sigma_to_clip=flux_sigma_to_clip)
    sm.train_cannon_model(suppress_output=suppress_output)

# Predict and plot
if do_cross_validation:
    sm.run_cross_validation()

    labels_pred = sm.cross_val_labels

# Just test on training set (to give a quick idea of performance)
else:
    labels_pred, errs_all, chi2_all = sm.infer_labels(
        sm.masked_data, sm.masked_data_ivar)

# Save model
sm.save_model(model_save_path)

#------------------------------------------------------------------------------
# Diagnostic Plotting
#------------------------------------------------------------------------------
# Work out uncertainties
label_pred_std = np.nanstd(label_values - labels_pred, axis=0)
std_text = "sigma_teff = {:0.2f}, sigma_logg = {:0.2f}, sigma_feh = {:0.2f}"
print(std_text.format(*label_pred_std))

fn_label = "_{}_{}_label_{}".format(
    model_type, len(label_names), "_".join(label_names))

# Label recovery for Teff, logg, and [Fe/H]
splt.plot_label_recovery(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    obs_join=obs_join[std_mask],
    fn_suffix=fn_label,)

# Plot recovery for interferometric Teff, M+15 [Fe/H], RA+12 [Fe/H], CPM [Fe/H]
splt.plot_label_recovery_per_source( 
    label_values=sm.training_labels, 
    e_label_values=sm.training_variances**2, 
    label_pred=labels_pred, 
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L), 
    obs_join=obs_join[std_mask],
    fn_suffix=fn_label,)

# And finally plot the label recovery for any abundances we might be using
splt.plot_label_recovery_abundances(
    label_values=sm.training_labels,
    e_label_values=sm.training_variances**2,
    label_pred=labels_pred,
    e_label_pred=np.tile(label_pred_std, sm.S).reshape(sm.S, sm.L),
    obs_join=obs_join[std_mask],
    fn_suffix=fn_label,
    abundance_labels=abundance_labels)

# Save theta coefficients - one for each WiFeS arm
splt.plot_theta_coefficients(
    sm,
    x_lims=(wl_min_model,5400),
    y_theta_lims=(-0.06,0.06),
    y_s2_lims=(-0.001, 0.006),
    x_ticks=(200,100),
    label="b",
    fn_suffix=fn_label,) 

splt.plot_theta_coefficients(
    sm,
    x_lims=(5400,wl_max_model),
    y_s2_lims=(-0.0005, 0.005),
    x_ticks=(200,100),
    label="r",
    fn_suffix=fn_label,)

# Plot comparison of observed vs model spectra. Here we've picked a set of 
# spectral types at approximately low, high, and solar [Fe/H] for testing.
source_ids = [
    #"2595284016771502080",      # M5, +, LHS 3799
    "5853498713160606720",      # M5.5 +, GJ 551
    "2467732906559754496",      # M5.1, 0, GJ 3119
    #"2640434056928150400",      # M5, 0, GJ 1286
    "2358524597030794112",      # M5, -, PM J01125-1659

    "2603090003484152064",      # M3, +, GJ 876
    "4508377078422114944",      # M4, 0, GJ 4065
    "4472832130942575872",      # M4, -, Gl 699

    "2910909931633597312",      # M3, +, LP 837-53
    "3184351876391975808",      # M2, 0, Gl 173
    "2979590513145784192",      # M2, -, Gl 180

    "145421309108301184",       # K8, +, Gl 169
    "2533723464155234176",      # K8, 0, Gl 56.3 B
    "1244644727396803584",      # K8, -, Gl 525
]
    #"3796072592206250624",      # M4, GJ 447
    #"3101920046552857728",      # M2, Gl 250 B
    #"3738099879558957952",      # M1, Gl 514
    #"3339921875389105152",      # K8, Gl 208
    #"3057712188691831936",      # K7, Gl 282B


splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    source_ids=source_ids,
    x_lims=(wl_min_model,5400),
    fn_label="b",)

splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    source_ids=source_ids,
    x_lims=(5400,wl_max_model),
    fn_label="r",)

# Plot diagnostic model fit for all stars
bp_rp_order = np.argsort(obs_join[std_mask]["Bp-Rp"])
splt.plot_spectra_comparison(
    sm=sm,
    obs_join=obs_join[std_mask],
    source_ids=obs_join[std_mask].index[bp_rp_order],
    x_lims=(wl_min_model,wl_max_model),
    fn_label="d",)

#------------------------------------------------------------------------------
# Predict labels for TESS targets
#------------------------------------------------------------------------------
tess_flux, tess_ivar, tess_bad_px_mask = stannon.prepare_fluxes(
    spec_tess_br,
    e_spec_tess_br,)

tess_wls = wave_tess_br

# Add photometry
if add_photometry:
    tess_wls = np.concatenate([tess_wls, phot_wls])
    tess_flux = np.concatenate(
        [tess_flux, obs_join_tess[abs_mags].values], axis=1)
    tess_ivar = np.concatenate(
        [tess_ivar, 1/obs_join_tess[e_abs_mags].values**2], axis=1)

# Predict
tess_labels_pred, tess_errs_all, tess_chi2_all = sm.infer_labels(
    tess_flux[:,sm.adopted_wl_mask],
    tess_ivar[:,sm.adopted_wl_mask])

# Plot CMD
splt.plot_cannon_cmd(
    benchmark_colour=obs_join[std_mask]["Bp-Rp"],
    benchmark_mag=obs_join[std_mask]["K_mag_abs"],
    benchmark_feh=label_values[:,2],
    science_colour=obs_join_tess["Bp-Rp"],
    science_mag=obs_join_tess["K_mag_abs"],)

# Plot Kiel Diagram for results
splt.plot_kiel_diagram(
    teffs=tess_labels_pred[:,0],
    e_teffs=np.repeat(label_pred_std[0], len(tess_labels_pred)),
    loggs=tess_labels_pred[:,1],
    e_loggs=np.repeat(label_pred_std[1], len(tess_labels_pred)),
    fehs=tess_labels_pred[:,2],
    label="science",)

# And one for the benchmarks
splt.plot_kiel_diagram(
    teffs=label_values[:,0],
    e_teffs=label_var[:,0]**2,
    loggs=label_values[:,1],
    e_loggs=label_var[:,1]**2,
    fehs=label_values[:,2],
    label="benchmark",)
