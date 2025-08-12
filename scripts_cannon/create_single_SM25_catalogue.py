"""Script to compile separately sampled chemodynamic trends from the script
collate_sampled_chemo_kinematic_params.py into a single file for use when
running assess_literature_systematics.py.
"""
import numpy as np
import pandas as pd
import plumage.utils as pu
import matplotlib.pyplot as plt

# MK samples
# [source_id_dr3, ra, dec, dist, pm_ra, pm_dec, rv, [Fe/H], vphi, X_Fe]
X_Fe_KM_fns = [
    "data/monty_sampled_params_Al_Fe_n224.csv",
    "data/monty_sampled_params_Ca_Fe_n224.csv",
    "data/monty_sampled_params_Mg_Fe_n224.csv",
    "data/monty_sampled_params_Na_Fe_n224.csv",
    "data/monty_sampled_params_Ti_Fe_n224.csv",]

# Brewer+16 samples
# [GaiaID [X/Fe]_true [X/Fe]_pred]
X_Fe_B16_fns = [
    "data/SM25_B16_pred_Al_Fe.dat",
    "data/SM25_B16_pred_Ca_Fe.dat",
    "data/SM25_B16_pred_Mg_Fe.dat",
    "data/SM25_B16_pred_Na_Fe.dat",
    "data/SM25_B16_pred_Ti_Fe.dat",]

b16_tsv = "data/B16_dr3_all.tsv"

# -----------------------------------------------------------------------------
# Combine K/M
# -----------------------------------------------------------------------------
MK_dfs = []

# Import all KM samples
for fn in X_Fe_KM_fns:
    df = pd.read_csv(fn, delimiter="\t", dtype={"source_id_dr3":str})
    df.set_index("source_id_dr3", inplace=True)
    df.sort_index(inplace=True)
    MK_dfs.append(df)

# Collate all KM samples into a single dataframe.
assert len(set(tuple(df.index.values) for df in MK_dfs)) == 1

# Use first DataFrame as a base, to which we'll insert additional [X/Fe]. This
# has the problem that we disconnect [X/Fe] from the other sampled parameters
# since SM looks to have run each [X/Fe] separately, but we're not using the
# other parameters so this should be fine.
MK_df = MK_dfs[0].copy()

for df in MK_dfs[1:]:
    # [X/Fe] columns are the 2nd and 3rd last
    cols = df.columns.values
    X_Fe = cols[-3]
    e_X_Fe = cols[-2]

    MK_df.insert(
        loc=len(MK_df.columns)-1, column=X_Fe, value=df[X_Fe].values)
    MK_df.insert(
        loc=len(MK_df.columns)-1, column=e_X_Fe, value=df[e_X_Fe].values)

# Drop the existing (assumed entirely NaN) BP-RP column
assert np.sum(~np.isnan( MK_df["bp_rp"].values)) == 0
MK_df.drop(columns=["bp_rp"], inplace=True)

# Crossmatch this back to our obs_join DataFrame to specifically get BP-RP. We
# do this so that we can self-consistently correct for systematics later, even
# if for the chemodynamic sample we only ever correct for a scalar value.
obs_join = pu.load_fits_table("CANNON_INFO", "cannon_mk")
obs_join_subset =  obs_join[["BP_RP_dr3"]].copy()
obs_join_subset.rename(columns={"BP_RP_dr3":"bp_rp"}, inplace=True)

MK_df = MK_df.join(obs_join_subset, "source_id_dr3",).copy()

# -----------------------------------------------------------------------------
# Combine B16
# -----------------------------------------------------------------------------
B16_dfs = []

# Import all KM samples
for fn in X_Fe_B16_fns:
    df = pd.read_csv(fn, delim_whitespace=True, dtype={"GaiaID":str})
    df.rename(columns={"GaiaID":"source_id_dr3"}, inplace=True)
    df.set_index("source_id_dr3", inplace=True)
    df.sort_index(inplace=True)
    B16_dfs.append(df)

# Collate all B16 samples into a single dataframe.
assert len(set(tuple(df.index.values) for df in B16_dfs)) == 1

# Use first DataFrame as a base, to which we'll insert additional [X/Fe]. This
# has the problem that we disconnect [X/Fe] from the other sampled parameters
# since SM looks to have run each [X/Fe] separately, but we're not using the
# other parameters so this should be fine.
B16_SM25_df = B16_dfs[0].copy()

for df in B16_dfs[1:]:
    # [X/Fe] columns are the 1st and 2nd columns
    cols = df.columns.values
    X_Fe = cols[0]
    e_X_Fe = cols[1]

    B16_SM25_df.insert(
        loc=len(B16_SM25_df.columns), column=X_Fe, value=df[X_Fe].values)
    B16_SM25_df.insert(
        loc=len(B16_SM25_df.columns), column=e_X_Fe, value=df[e_X_Fe].values)

# Impot original B16 catalogue
B16_all_df = pd.read_csv(
    b16_tsv,
    delimiter="\t",
    dtype={"source_id":str, "source_id_dr3":str},)
B16_all_df.rename(columns={"source_id":"source_id_dr3"}, inplace=True)
B16_all_df.set_index("source_id_dr3", inplace=True)

# Crossmatch this so we can grab RA, DEC, BP-RP
B16_comb = B16_SM25_df.join(B16_all_df, "source_id_dr3",)

# Now grab just the relevant columns to append to the bottom of the KM DF
columns = ["ra", "dec", "[Al/Fe]_pred", "[Ca/Fe]_pred", "[Mg/Fe]_pred",
           "[Na/Fe]_pred", "[Ti/Fe]_pred", "bp_rp"]
B16_selected = B16_comb[columns].copy()

X_Fe_cols = ["[Al/Fe]_pred", "[Ca/Fe]_pred", "[Mg/Fe]_pred",
    "[Na/Fe]_pred", "[Ti/Fe]_pred"]
X_Fe_cols_new = [xfe[:-5] for xfe in X_Fe_cols]

B16_selected.rename(
    columns={key:value for key, value in zip(X_Fe_cols, X_Fe_cols_new)},
    inplace=True,)

# Add in dummy columns
dummy_cols = ["e_ra", "e_dec", "dist", "e_dist", "pm_ra", "e_pm_ra", "pm_dec",
    "e_pm_dec", "rv", "e_rv", "[Fe/H]", "e_[Fe/H]", "vphi", "e_vphi",
    'e_[Al/Fe]', 'e_[Ca/Fe]', 'e_[Mg/Fe]', 'e_[Na/Fe]', 'e_[Ti/Fe]']

for col in dummy_cols:
    B16_selected[col] = np.nan

# Finally, re-order
B16_selected = B16_selected[MK_df.columns.values].copy()

# -----------------------------------------------------------------------------
# Diagnostic plot
# -----------------------------------------------------------------------------
X_Fe = ["Ti", "Ca", "Na", "Al", "Mg"]

plt.close("all")
fig, axes = plt.subplots(nrows=len(X_Fe), sharex=True, figsize=(10,6))

for i, x in enumerate(X_Fe):
    resid = B16_comb["[{}/Fe]_true".format(x)].values \
        - B16_comb["[{}/Fe]_pred".format(x)].values
    bp_rp = B16_comb["bp_rp"].values
    mn = np.nanmedian(resid)
    std = np.nanstd(resid)
    axes[i].plot(bp_rp, resid, ".")
    axes[i].hlines(
        0, np.nanmin(bp_rp), np.nanmax(bp_rp), linestyles="dashed", color="k")
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
plt.savefig("paper/SM25_vs_B16_X_Fe.pdf")

# -----------------------------------------------------------------------------
# Combine MMK and B16 DataFrames
# -----------------------------------------------------------------------------
# We need to drop the duplicates from B16, as some targets will be in common.
# However, since we'll be comparing with the B16 catalogue imported to the
# assess_literature_systematics.py script, we don't need to keep the original
# values here and thus can preferentially take those properly MCMC sampled.
B16_ids = B16_selected.index.values
MK_ids = MK_df.index.values

dup_ids = np.array([id in MK_ids for id in B16_ids])

SM25_all = pd.concat([MK_df, B16_selected[~dup_ids]], axis=0)

save_fn = "data/SM25_X_Fe_chemodynamic_{}.tsv".format("_".join(X_Fe))

SM25_all.to_csv(save_fn, sep="\t")

