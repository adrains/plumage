"""Script to compile separately sampled chemodynamic [X/Fe] from into a single
file for use when running assess_literature_systematics.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Reference sample. Set this to 'KM' when running the benchmark sample, and to
# e.g. 'A12' when running the reference sample.
ref = "KM"

ref_tsv = "data/std_info.tsv"       # For benchmarks
#ref_tsv = "data/A12_gaia_all.tsv"  # For reference sample

# Abundances to consider
abund_X = ["Ti", "Mg", "Ca",]

# Whether the abundances are in different files (False) or the same file (True)
using_single_sampled_file = True

# Abundances are in the same file due to sampling
if using_single_sampled_file:
    xh_label = "_".join(["{}Fe".format(xh) for xh in abund_X])
    X_Fe_map_fn = "data/cd_samples/260811_{}_{}_Pred.csv".format(ref, xh_label)

# Or abundances are in separate files
else:
    # columns: [GaiaID [X/Fe]_true [X/Fe]_pred]
    base_fn = "data/cd_samples/260803_{}_{}Fe_Pred.dat"
    X_Fe_map_fns = [base_fn.format(ref, xfe) for xfe in abund_X]

sampled_cols = ["e_ra", "e_dec", "dist", "e_dist", "pm_ra", "e_pm_ra", 
    "pm_dec", "e_pm_dec", "rv", "e_rv", "[Fe/H]", "e_[Fe/H]", "vphi", "e_vphi"]

# -----------------------------------------------------------------------------
# Combine our reference sample
# -----------------------------------------------------------------------------
# In the case of the stars that have been MC sampled, their abundances have all
# been simultaneously sampled and are in the same file.
if using_single_sampled_file:
    ref_CD_df = pd.read_csv(
        X_Fe_map_fn, delim_whitespace=True, dtype={"source_id_dr3":str})
    ref_CD_df.set_index("source_id_dr3", inplace=True)
    ref_CD_df.sort_index(inplace=True)

# Otherwise we need to load in separate files and combine.
else:
    ref_dfs = []

    # Import all [X/Fe] samples
    for fn, xfe in zip(X_Fe_map_fns, abund_X):
        df = pd.read_csv(fn, delim_whitespace=True, dtype={"GaiaID":str})
        df.rename(columns={"GaiaID":"source_id_dr3"}, inplace=True)
        df.set_index("source_id_dr3", inplace=True)
        df.sort_index(inplace=True)

        # Add in dummy columns that the reference sample doesn't have, but the
        # MCed sampled does have.
        dummy_cols = sampled_cols + ["e_[{}/Fe]_pred".format(xfe)]

        for col in dummy_cols:
            df[col] = np.nan

        ref_dfs.append(df)

    # Collate all reference samples into a single dataframe.
    assert len(set(tuple(df.index.values) for df in ref_dfs)) == 1

    # Use first DataFrame as a base, to which we'll insert additional [X/Fe].
    # This has the problem that we disconnect [X/Fe] from the other sampled
    # parameters since SM looks to have run each [X/Fe] separately, but we're
    # not using the other parameters so this should be fine.
    ref_CD_df = ref_dfs[0].copy()

    for df in ref_dfs[1:]:
        # [X/Fe] columns are the 1st and 2nd columns
        cols = df.columns.values
        X_Fe = cols[0]
        e_X_Fe = cols[1]

        ref_CD_df.insert(
            loc=len(ref_CD_df.columns), column=X_Fe, value=df[X_Fe].values)
        ref_CD_df.insert(
            loc=len(ref_CD_df.columns), column=e_X_Fe, value=df[e_X_Fe].values)

# Import original reference catalogue
ref_all_df = pd.read_csv(
    ref_tsv,
    delimiter="\t",
    dtype={"source_id":str, "source_id_dr3":str},
    comment="#",)
ref_all_df.rename(
    columns={"source_id":"source_id_dr3",
             "BP-RP_dr3":"bp_rp",
             "ra_dr3":"ra",
             "dec_dr3":"dec"},
    inplace=True,)
ref_all_df.set_index("source_id_dr3", inplace=True)

# Crossmatch this so we can grab RA, DEC, BP-RP
ref_comb = ref_CD_df.join(ref_all_df, "source_id_dr3", rsuffix="_obs")

# Construct abundance value and error columns
X_Fe_cols = ["[{}/Fe]_pred".format(xfe) for xfe in abund_X]
X_Fe_cols_new = ["[{}/Fe]".format(xfe) for xfe in abund_X]

e_X_Fe_cols = ["e_[{}/Fe]_pred".format(xfe) for xfe in abund_X]
e_X_Fe_cols_new = ["e_[{}/Fe]".format(xfe) for xfe in abund_X]

# Interleave
X_Fe_cols_all = \
    [xx for sublist in zip(X_Fe_cols, e_X_Fe_cols) for xx in sublist]
X_Fe_cols_new_all = \
    [xx for sublist in zip(X_Fe_cols_new, e_X_Fe_cols_new) for xx in sublist]

# Now grab just the relevant columns to append to the bottom of the KM DF
columns = ["ra", "dec",] + sampled_cols + ["bp_rp"] + X_Fe_cols_all

ref_selected = ref_comb[columns].copy()

# Rename to remove '_pred'
ref_selected.rename(
    columns={key:value for key, value in zip(X_Fe_cols_all, X_Fe_cols_new_all)},
    inplace=True,)

# -----------------------------------------------------------------------------
# Diagnostic plot
# -----------------------------------------------------------------------------
# Only plot the recovery plot for the samples where we know the ground truth.
if not using_single_sampled_file:
    plt.close("all")
    fig, axes = plt.subplots(nrows=len(abund_X), sharex=True, figsize=(10,6))

    for i, x in enumerate(abund_X):
        resid = ref_comb["[{}/Fe]_true".format(x)].values \
            - ref_comb["[{}/Fe]_pred".format(x)].values
        bp_rp = ref_comb["bp_rp"].values
        mn = np.nanmedian(resid)
        std = np.nanstd(resid)
        axes[i].plot(bp_rp, resid, ".")
        axes[i].hlines(
            y=0,
            xmin=np.nanmin(bp_rp),
            xmax=np.nanmax(bp_rp),
            linestyles="dashed",
            color="k")
        axes[i].text(
            x=0.8,
            y=0.2,
            s=r"${:0.2f} \pm {:0.2f}$ dex".format(mn, std),
            transform=axes[i].transAxes,
            horizontalalignment="center")
        axes[i].set_title(x)
        axes[i].set_ylabel(r"$\Delta$[{}/Fe]".format(x))
    axes[i].set_xlabel(r"$BP-RP$")
    plt.tight_layout()
    plt.savefig("paper/CD_vs_{}_X_Fe.pdf".format(ref))

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
save_fn = "data/chemodynamic_X_Fe_{}_{}.tsv".format(ref, "_".join(abund_X))

ref_selected.to_csv(save_fn, sep="\t")

print("Saved to: {}".format(save_fn))