"""Script to plan observations of new binaries given a the set of targets we've
already observed, plus candidate binaries with preferred candidates flagged. We
then plot a BP-RP vs [Fe/H] plot for binaries, interferometric, candidates, and
WD secondaries.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# List of WD secondary BP-RP values
wd_bp_rp = np.array([
    1.759798,
    1.7129335,
    4.3793364,
    1.8374443,
    1.0452166,
    0.51247644,
    1.6572514,
    1.6732607,
    2.803464,
    2.5311613,
    2.6131277,
    1.2755804,
    1.4490776,
    0.63952446,
    1.888464,])

BP_RP_lims = (1.0, 3.9)

# -----------------------------------------------------------------------------
# Import list of *observed* targets
# -----------------------------------------------------------------------------
mike_info_fn = "data/mike_info.tsv"

mike_info = pd.read_csv(
    filepath_or_buffer=mike_info_fn,
    sep="\t",
    comment="#",
    dtype={"source_id":str},)

mike_info.set_index("source_id", inplace=True)

# Exclude rejected targets and calibrators
exclude_mask = np.logical_or(
    mike_info["rejected"].values, mike_info["observed"].values == "-")

mike_info = mike_info[~exclude_mask].copy()

mike_info["Fe_H"] = mike_info["Fe_H"].astype(float)

# -----------------------------------------------------------------------------
# Import list of *candidate* targets
# -----------------------------------------------------------------------------
binary_info_fn = "data/mike_FGK_KM_target_list.tsv"

binary_info = pd.read_csv(
    filepath_or_buffer=binary_info_fn,
    sep="\t",
    comment="#",
    dtype={"source_id":str},)

binary_info.set_index("source_id", inplace=True)

# Mask
is_candidate = np.all([
    np.logical_or(
        binary_info["kind"].values == "secondary",
        binary_info["kind"].values == "interferometric",),
    binary_info["vis_1103"].values == "TRUE",
    [type(sts) == str for sts in binary_info["status_2026"].values],], axis=0)

binary_info = binary_info[is_candidate].copy()

binary_info["Fe_H"] = binary_info["Fe_H"].astype(float)

is_int_candidate = binary_info["kind"].values == "interferometric"

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")
fig, axis = plt.subplots(figsize=(7,4))

# # # # # # # # # # # # # # # # # # #
# Plot binary secondaries
# # # # # # # # # # # # # # # # # # #
is_sec = mike_info["component"] == "sec"

sc = axis.scatter(
    mike_info[is_sec]["bp_rp"].values,
    mike_info[is_sec]["Fe_H"].values,
    edgecolor="b",
    facecolor="None",
    #c=mike_info[is_sec]["phot_bp_mean_mag"].values,
    marker="o",
    label="Observed Binary ({:0.0f})".format(np.sum(is_sec)),)

# # # # # # # # # # # # # # # # # # #
# Plot interferometric benchmarks
# # # # # # # # # # # # # # # # # # #
is_int = mike_info["is_int"].values

sc = axis.scatter(
    mike_info[is_int]["bp_rp"].values,
    mike_info[is_int]["Fe_H"].values,
    edgecolor="k",
    facecolor="None",
    #c=mike_info[is_sec]["phot_bp_mean_mag"].values,
    marker="*",
    label="Observed Interferometric ({:0.0f})".format(np.sum(is_int)),)

# # # # # # # # # # # # # # # # # # #
# Plot WD secondaries
# # # # # # # # # # # # # # # # # # #
for bp_rp_i, bp_rp in enumerate(wd_bp_rp):
    axis.vlines(
        x=bp_rp,
        ymin=0.5,
        ymax=0.55,
        colors="r",
        alpha=0.7,
        linewidth=0.5,
        label="WD secondary" if bp_rp_i == 0 else None,)

# # # # # # # # # # # # # # # # # # #
# Plot candidates
# # # # # # # # # # # # # # # # # # #
axis.scatter(
    binary_info["BP-RP_dr3"].values[~is_int_candidate],
    binary_info["Fe_H"].values[~is_int_candidate],
    edgecolor="g",
    facecolor="None",
    marker="^",
    label="Binary Candidate ({:0.0f})".format(np.sum(is_candidate)),)

axis.scatter(
    binary_info["BP-RP_dr3"].values[is_int_candidate],
    binary_info["Fe_H"].values[is_int_candidate],
    edgecolor="r",
    facecolor="None",
    marker="*",
    label="Interferometric Candidate ({:0.0f})".format(np.sum(is_int_candidate)),)


# # # # # # # # # # # # # # # # # # #
# Plot setup
# # # # # # # # # # # # # # # # # # #

axis.set_xlim(BP_RP_lims[0], BP_RP_lims[1])
axis.xaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
axis.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))

axis.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
axis.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))

axis.grid(True, which="major", linestyle=":")

axis.set_xlabel(r"$BP-RP$")
axis.set_ylabel("[Fe/H]")

axis.legend(
    loc="upper center",
    fontsize="x-small",
    ncol=3,
    bbox_to_anchor=(0.5, 1.15),)
    
plt.tight_layout()