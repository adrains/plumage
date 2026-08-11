"""Script to save mean and standard deviation of GALAH+Gaia sampled parameters,
including predicted [X/Fe] abundances.
"""
import os
import pandas as pd
import numpy as np
import glob

# -----------------------------------------------------------------------------
# Setup + Settings
# -----------------------------------------------------------------------------
# The individual elements we want to import
abund = ["Ti", "Mg", "Ca"]

# Formatted abundance labels with [X/Fe] notation (for column names)
X_Fe = ["[{}/Fe]_pred".format(xh) for xh in abund]

# Formatted abundance labels with XFe notation (for file names)
X_Fe_label = ["{}Fe".format(xh) for xh in abund]

# Import all files via glob
path_wc = "/Users/arains/Dropbox/AdamTiDists/2026_PaperII/Results/Benchmark_Errors/*PullsSymmetricErrors"
sample_files = glob.glob(path_wc)

if len(sample_files) == 0:
    raise Exception("No files found!")
else:
    print("{} files found.".format(len(sample_files)))

# Details for our ouput filename, <path>/<label>_<abundances>_Pred.csv
run_label = "260811_KM"
out_path = "/Users/arains/Dropbox/code/plumage/data/cd_samples/"

# Setup mean and sigma columns for dataframe and interleave
cols_mean = \
    ["ra", "dec", "dist", "pm_ra", "pm_dec",  "rv", "[Fe/H]", "vphi",] + X_Fe

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
#df["bp_rp"] = np.nan

# Save
out_fn = os.path.join(
    out_path, 
    "{}_{}_Pred.csv".format(run_label, "_".join(X_Fe_label)))
df.to_csv(out_fn, sep="\t")