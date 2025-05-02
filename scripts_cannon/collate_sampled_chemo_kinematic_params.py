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
path_wc = "/Users/adamrains/Dropbox/AdamTiDists/2025/Results/*PullsSymmetricErrors"
sample_files = glob.glob(path_wc)

# Setup mean and sigma columns for dataframe and interleave
cols_mean = ["ra", "dec", "dist", "pm_ra", "pm_dec",  "rv", "[Fe/H]", "vphi", 
             "[Ti/Fe]"]

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

# Add dummy BP-RP column
df["bp_rp"] = np.nan

# Save
df.to_csv("data/monty_sampled_params_n{:0.0f}.csv".format(n_stars), sep="\t")