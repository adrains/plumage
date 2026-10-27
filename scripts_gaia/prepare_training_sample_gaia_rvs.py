"""Script to prepare the training dataset for use when training a Cannon model
based on Gaia RVS spectra and APOGEE stellar parameters. The resulting training
labels are saved to our fits file. Note that stars prepared in this way
represent *potential* training stars that might be used as part of the training
sample for a given Cannon model--we need not use all possible stars.

This is part of a series of Cannon scripts in scripts_gaia/:
  1) collate_spectra_gaia_rvs.py
  2) prepare_training_sample_gaia_rvs.py
  3) train_stannon_gaia_rvs.py
  4) make_stannon_diagnostics_gaia_rvs.py
  4) run_stannon_gaia_rvs.py

The parameters for this script can be found in label_settings_gaia_rvs.yml.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import plumage.utils as pu
import stannon.utils as su
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
from astropy.stats import sigma_clip

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def prepare_labels(
    obs_tab,
    base_labels,
    abundance_labels,):
    """Prepare our set of training labels (currently only from APOGEE) and add
    new columns to obs_tab in the Stannon-expected format. The columns added
    are: ['label_adopt_<>', 'label_adopt_sigma_<>', 'label_adopt_var_<>',
    'label_source_<>']  per label, as well as two global masks:
    ['has_base_label_set', 'has_complete_label_set']. The DataFrame is updated
    in place.

    Parameters
    ----------
    obs_tab: pandas DataFrame
        Pandas dataframe crossmatch containing observation Gaia and APOGEE
        information.

    base_labels, abundance_labels: str list
        Base (e.g. ['teff', 'logg', 'Fe_H']) and abundance labels (e.g. 
        ['Ti_Fe', 'Ca_Fe', 'Si_Fe']) to collate for each star.

    Updated
    -------
    obs_tab: DataFrame
        New columns added to DataFrame corresponding to adopted labels.
    """
    # Initialise for convenience
    labels = base_labels + abundance_labels
    n_labels = len(labels)
    n_stars = len(obs_tab)

    # Initialise label vector
    label_values = np.full( (n_stars, n_labels), np.nan)
    label_sigmas = np.full( (n_stars, n_labels), np.nan)
    label_variances = np.full( (n_stars, n_labels), np.nan)

    # Initialise record of label source/s
    label_sources = np.full( (n_stars, n_labels), "").astype(object)

    # By default everything has APOGEE labels, so we can just assign everything
    # accordingly before working with any exceptions
    for label_i, label in enumerate(labels):
        APOGEE_value_col = label.upper()
        APOGEE_sigma_col = "{}_ERR".format(label.upper())

        label_values[:, label_i] = obs_tab[APOGEE_value_col]
        label_sigmas[:, label_i] = obs_tab[APOGEE_sigma_col]
        label_variances[:, label_i] = obs_tab[APOGEE_sigma_col]**2
        label_sources[:, label_i] = "APOGEE"

        # Add new columns, being careful to provide the indices of the new
        # columns so that pandas can do the table join properly.
        obs_tab["label_adopt_{}".format(label)] = \
            pd.Series(label_values[:,label_i], index=obs_tab.index)
        obs_tab["label_adopt_sigma_{}".format(label)] = \
            pd.Series(label_sigmas[:,label_i], index=obs_tab.index)
        obs_tab["label_adopt_var_{}".format(label)] = \
            pd.Series(label_variances[:,label_i], index=obs_tab.index)
        obs_tab["label_source_{}".format(label)] = \
            pd.Series(label_sources[:,label_i], index=obs_tab.index)

    # Assign masks
    has_base_label_set = np.any(
        ~np.logical_or(np.isnan(label_values[:,:3]), 
                       np.isnan(label_sigmas[:,:3])), axis=1)

    has_complete_label_set = np.any(
            ~np.logical_or(np.isnan(label_values), 
                           np.isnan(label_sigmas)), axis=1)

    # Add corresponding mask to the dataframe
    obs_tab["has_base_label_set"] = \
        pd.Series(has_base_label_set, index=obs_tab.index)
    obs_tab["has_complete_label_set"] = \
        pd.Series(has_complete_label_set, index=obs_tab.index)

    # Compute the final mask
    icb = np.logical_and(
        obs_tab["is_candidate_benchmark"].values,
        obs_tab["has_complete_label_set"].values,)

    obs_tab["is_cannon_benchmark"] = pd.Series(icb, index=obs_tab.index)


def flag_Ca_triplet_emission(
    wave_1D,
    spectra_2D,
    Ca_triplet_windows_nm,
    sigma_threshold,
    wing_width_nm=0.05,):
    """Flags those stars with Ca triplet emission by comparing line cores and
    wings. Can run on a list of lines, or a single line.

    Parameters
    ----------
    wave_1D: 1D float array
        Wavelength array of shape [n_px].

    spectra_2D: 2D float array
        Gaia RVS spectra array of shape [n_star, n_px].

    Ca_triplet_windows_nm: 2D float array
        List of [min,max] wavelength line core ranges to inspect for emission.

    sigma_threshold: float
        Sigma clipping threshold for assessing emission.

    wing_width_nm: float
        The width either side of the line core to treat as the wing.

    Returns
    -------
    has_Ca_triplet_emission: 1D bool array
        Boolean array where True indicates line emission, of shape [n_star].
    """
    # Create a Ca triplet mask
    is_Ca_triplet_core = np.full(spectra_2D.shape[1], False)
    is_Ca_triplet_wing = np.full(spectra_2D.shape[1], False)

    for line in Ca_triplet_windows_nm:
        # Compute a mask for the line cores
        (core_min, core_max) = line
        
        is_Ca_triplet_core = np.logical_or(
            is_Ca_triplet_core,
            np.logical_and(wave_1D >= core_min, wave_1D <= core_max))

        # Compute a mask for the line wings
        wing_min = core_min - wing_width_nm
        wing_max = core_max + wing_width_nm

        is_Ca_triplet_wing = np.logical_or(
            is_Ca_triplet_wing,
            np.logical_or(
                np.logical_and(wave_1D >= wing_min, wave_1D <= core_min),
                np.logical_and(wave_1D >= core_max, wave_1D <= wing_max)))
    
    # Compute difference between maximum core flux and minimum wing flux
    Ca_triplet_core_max = np.nanmax(spectra_2D[:,is_Ca_triplet_core], axis=1)
    Ca_triplet_wing_min = np.nanmin(spectra_2D[:,is_Ca_triplet_wing], axis=1)
    Ca_triplet_delta = Ca_triplet_core_max - Ca_triplet_wing_min

    # Sigma clip based on this difference
    is_sc = sigma_clip(Ca_triplet_delta, sigma=sigma_threshold).mask

    # Also enforce that the core has more flux than the line
    Ca_triplet_wing_med = np.nanmedian(spectra_2D[:,is_Ca_triplet_wing],axis=1)
    core_gt_wing = Ca_triplet_core_max > Ca_triplet_wing_med

    has_Ca_triplet_emission = np.logical_and(is_sc, core_gt_wing)

    return has_Ca_triplet_emission

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
label_settings = "scripts_gaia/gaia_rvs_label_settings.yml"
ls = su.load_yaml_settings(label_settings)

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
obs_tab = pu.load_fits_table(
    extension="OBS_TAB",
    label=ls.fits_label,
    fn_base=ls.fits_fn_base,
    path=ls.fits_folder,)

wave = pu.load_fits_image_hdu(
    extension="wave",
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,
    label=ls.fits_label,
    arm="RVS")

spectra_2D = pu.load_fits_image_hdu(
    extension="spec",
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,
    label=ls.fits_label,
    arm="RVS")

sigmas_2D = pu.load_fits_image_hdu(
    extension="sigma",
    path=ls.fits_folder,
    fn_base=ls.fits_fn_base,
    label=ls.fits_label,
    arm="RVS",)

#------------------------------------------------------------------------------
# Computing extra columns
#------------------------------------------------------------------------------
# Compute distance and M_G, TODO: use Bailer-Jones distances
dist = 1000 / obs_tab["parallax"].values
e_dist = obs_tab["parallax_error"].values
M_G = obs_tab["phot_g_mean_mag"].values - 5*np.log10(dist/10)

# Save back to DataFrame
obs_tab["dist"] = pd.Series(dist, index=obs_tab.index)
obs_tab["e_dist"] = pd.Series(e_dist, index=obs_tab.index)
obs_tab["M_G"] = pd.Series(M_G, index=obs_tab.index)

#------------------------------------------------------------------------------
# Quality cuts
#------------------------------------------------------------------------------
# APOGEE bitmasks
apogee_target_bitmask = obs_tab["APOGEE_TARGET1"].values.astype(np.int32)
apogee_star_bitmask = obs_tab["STARFLAG"].values.astype(np.int32)
apogee_snr = obs_tab["SNR"].values

# Gaia flags
ruwe = obs_tab["ruwe"].values

# Compute Ca triplet emission
has_Ca_triplet_emission = flag_Ca_triplet_emission(
    wave_1D=wave,
    spectra_2D=spectra_2D,
    Ca_triplet_windows_nm=ls.Ca_triplet_windows,
    sigma_threshold=ls.emission_sigma_to_mask,
    wing_width_nm=ls.wing_width_nm,)

obs_tab["has_Ca_triplet_emission"] = \
    pd.Series(has_Ca_triplet_emission, index=obs_tab.index)

# Create mask ensuring that all (relevant) APOGEE parameters are unflagged. The
# parameter flags are: ['TEFF_FLAG', 'LOGG_FLAG', 'LOGVMICRO_FLAG', 'M_H_FLAG',
# 'C_M_FLAG', 'N_M_FLAG', 'ALPHA_M_FLAG', 'LGVSINI_FLAG', 'PARAM_O_FLAG'].
has_reliable_params = np.full(obs_tab.shape[0], True)
for param in ls.param_flags_to_check:
    has_reliable_params = np.logical_and(
        has_reliable_params, 
        obs_tab[param].values == 0)

# Create mask ensuring that all [X/Fe] are unflagged
has_reliable_X_Fe = np.full(obs_tab.shape[0], True)
for X_Fe in ls.abundance_labels:
    has_reliable_X_Fe = np.logical_and(
        has_reliable_X_Fe, 
        obs_tab["{}_FLAG".format(X_Fe.upper())].values == 0)

# TODO: passed_quality_cuts column
pass

# Create a mask for what we'll consider a candidate training star
is_candidate_benchmark = np.all([
    apogee_target_bitmask == 0,
    apogee_star_bitmask == 0,
    has_reliable_params,
    has_reliable_X_Fe,
    apogee_snr >= ls.min_apogee_snr,
    ~has_Ca_triplet_emission,
    ruwe < ls.ruwe_threshold,], axis=0)

# Save this mask back to table
obs_tab["is_candidate_benchmark"] = \
    pd.Series(is_candidate_benchmark, index=obs_tab.index)

# Format labels
prepare_labels(obs_tab, ls.base_labels, ls.abundance_labels)

pu.save_fits_table(
    extension="OBS_TAB",
    dataframe=obs_tab,
    label=ls.fits_label,
    fn_base=ls.fits_fn_base,
    path=ls.fits_folder,)

#------------------------------------------------------------------------------
# Diagnostic Plots
#------------------------------------------------------------------------------
# Gaia RVS wavelength scale settings
N_RVS_PX = 2401
WAVE_MIN = 846
WAVE_MAX = 870
WAVE_DELTA = 0.01

N_BM = np.sum(is_candidate_benchmark)
obs_tab_bm = obs_tab[is_candidate_benchmark].copy()
spectra_2D_bm = spectra_2D[is_candidate_benchmark]
sigmas_2D_bm = sigmas_2D[is_candidate_benchmark]

bp_rp_all = obs_tab_bm["bp_rp"].values
bp_rp_min = np.nanmin(bp_rp_all)
bp_rp_max = np.nanmax(bp_rp_all)
cmap = cm.get_cmap("magma")

#=========================================
# Plot #1: spectra colour-coded by BP-RP
#=========================================
plt.close("all")

fig_spec, axis_spec = plt.subplots(1, figsize=(18,8))

for star_i in tqdm(range(N_BM), desc="Plotting spectra", leave=False):
    if obs_tab_bm.iloc[star_i]["rvs_snr"] > 30:
        bp_rp = bp_rp_all[star_i]
        colour = cmap.reversed()((bp_rp-bp_rp_min)/(bp_rp_max))
        axis_spec.plot(wave, spectra_2D_bm[star_i], linewidth=0.1, c=colour)

        # HACK so that we have something to pass to the colourbar
        sc = axis_spec.scatter(
            0,1,c=bp_rp, cmap=cmap.reversed(), vmin=bp_rp_min, vmax=bp_rp_max)

axis_spec.set_xlabel("Wavelength (nm)")
axis_spec.set_xlim(WAVE_MIN-20*WAVE_DELTA,WAVE_MAX+20*WAVE_DELTA)

cb = fig_spec.colorbar(sc, ax=axis_spec, fraction=0.05)
cb.ax.set_title(r"$BP-RP$")

fig_spec.subplots_adjust(left=0.05, bottom=0.075, right=1.0, top=0.95,)

# Plot line list
line_list = pd.read_csv(
    "data/contursi21_gspec_linelist.tsv", delimiter="\t", comment="#")

# Get the colour map for our lines
cmap = cm.get_cmap("cubehelix")

# Count how many unique species are in the line list
unique_species = list(set(line_list["El"].values))
unique_species.sort()
n_unique_species = len(unique_species)
colour_i = (np.arange(n_unique_species+2)/(n_unique_species+2))[:-1]
species_mapping_dict = OrderedDict(zip(unique_species, colour_i))

species_plotted = []

for line_i, line_data in line_list.iterrows():
    species = line_data["El"]
    # Offset
    min_y = 1.07; max_y = 1.25
    height = (max_y - min_y) / n_unique_species
    species_i = np.argwhere(np.array(unique_species) == species)
    offset = species_i.item() * height

    # Label lines on spectral plot
    axis_spec.vlines(
        x=line_data["lambdaVac"],
        ymin=min_y+offset,
        ymax=min_y+height+offset,
        linewidth=1,
        colors=cmap(species_mapping_dict[species]),
        label=species,
        alpha=0.6,)

    if species not in species_plotted:
        species_plotted.append(species)

        axis_spec.text(
            x=WAVE_MIN-10*WAVE_DELTA,
            y =min_y+height/2+offset,
            s=species,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize="xx-small",
            color=cmap(species_mapping_dict[species]),)

# Legend
handles, labels = axis_spec.get_legend_handles_labels()
ii = np.argsort(labels)
by_label = OrderedDict(zip(np.array(labels)[ii], np.array(handles)[ii]))
leg = axis_spec.legend(
    handles=by_label.values(),
    labels=by_label.keys(),
    loc="lower center",
    ncol=n_unique_species,
    fontsize="medium",)

for legobj in leg.legendHandles:
    legobj.set_linewidth(1.5)

fig_spec.savefig("plots/gaia_rvs_selected_spectra.pdf")
fig_spec.savefig("plots/gaia_rvs_selected_spectra.png", dpi=400)

#=========================================
# Plot #2: CMD colour-coded by [Fe/H]
#=========================================
M_G = obs_tab_bm["M_G"].values
Fe_Hs = obs_tab_bm["FE_H"].values

fig_cmd, axis_cmd = plt.subplots(1, figsize=(8,6))
sc = axis_cmd.scatter(x=bp_rp_all, y=M_G, c=Fe_Hs, cmap="cividis")

cb = fig_cmd.colorbar(sc, ax=axis_cmd,)
cb.ax.set_title("[Fe/H]")

axis_cmd.set_xlabel(r"$BP-RP$")
axis_cmd.set_ylabel(r"$M_G$")

axis_cmd.set_title(r"$N_{{\rm stars}}={:0.0f}$".format(N_BM))

axis_cmd.set_ylim(13, -5)
fig_cmd.tight_layout()
fig_cmd.savefig("plots/gaia_rvs_selected_cmd.pdf")
fig_cmd.savefig("plots/gaia_rvs_selected_cmd.png", dpi=400)