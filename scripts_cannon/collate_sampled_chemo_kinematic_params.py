"""Script to save mean and standard deviation of GALAH+Gaia sampled parameters,
including predicted [Ti/Fe] abundances.
"""
import pandas as pd
import numpy as np
import glob
import plumage.utils as pu
import stannon.plotting as splt

# -----------------------------------------------------------------------------
# Setup + Settings
# -----------------------------------------------------------------------------
path_wc = "/Users/arains/Dropbox/AdamTiDists/Results/*PullsSymmetricErrors"
sample_files = glob.glob(path_wc)

# Setup mean and sigma columns for dataframe and interleave
cols_mean = ["ra_monty", "dec_monty", "dist_monty", "pm_ra_monty", 
            "pm_dec_monty",  "rv_monty", "feh_monty", "vphi_monty", 
            "Ti_Fe_monty"]

cols_sigma = ["e_{}".format(col) for col in cols_mean]

cols_all = [*sum(zip(cols_mean, cols_sigma),())]

# Dimensions for reference
n_cols = len(cols_mean)
n_stars = len(sample_files)
n_samples = 1000

# -----------------------------------------------------------------------------
# Load in samples and save mean+std
# -----------------------------------------------------------------------------
# Initialise
source_ids = []
samples = np.full((n_stars, n_cols, n_samples), np.nan)

# Loop over all stars and load in sampled parameters, extract source_id from fn
for sf_i, sf in enumerate(sample_files):
    samples[sf_i] = np.loadtxt(sf).T
    source_ids.append(sf.split("/")[-1].split("_")[0])

# Compute means and standard deviations
sample_means = np.nanmean(samples, axis=2)
sample_sigmas = np.nanstd(samples, axis=2)

# Construct dataframe
df = pd.DataFrame(
    columns=cols_mean+cols_sigma, 
    data=np.hstack((sample_means, sample_sigmas)), 
    index=source_ids)

# Reorder columns, assign name to index
df = df.reindex(columns=cols_all)
df.index.name = "source_id_dr3"

# Save
df.to_csv("data/monty_sampled_params_n{:0.0f}.csv".format(n_stars))

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
# Import existing dataframe, table join
obs_join = pu.load_fits_table("CANNON_INFO", "cannon")
obs_join = obs_join.join(df, "source_id_dr3",)

# Mask
is_cannon_benchmark = obs_join["is_cannon_benchmark"].values
obs_join = obs_join[is_cannon_benchmark]

# Grab masks
is_cpm = obs_join["is_cpm"]

labels = ["teff", "logg", "feh", "Ti_Fe"]
cols = ["label_adopt_{}".format(lbl) for lbl in labels]
e_cols = ["label_adopt_sigma_{}".format(lbl) for lbl in labels]

labels_pred = np.hstack(
    (np.full((103, 3), np.nan),
     np.atleast_2d(obs_join["Ti_Fe_monty"].values).T))
e_labels_pred = np.hstack(
    (np.full((103, 3), np.nan),
     np.atleast_2d(obs_join["e_Ti_Fe_monty"].values).T))

splt.plot_label_recovery_abundances(
    label_values=obs_join[cols].values[is_cpm],
    e_label_values=obs_join[e_cols].values[is_cpm],
    label_pred=labels_pred[is_cpm],
    e_label_pred=e_labels_pred[is_cpm],
    obs_join=obs_join[is_cpm],
    fn_suffix="_Ti_Fe_monty_comp",
    abundance_labels=["Ti_Fe"],
    feh_lims=(-0.15,0.4),
    feh_ticks=(0.4,0.2,0.2,0.1),)