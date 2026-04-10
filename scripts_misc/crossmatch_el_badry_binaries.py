import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import stannon.parameters as params
import plumage.utils as pu
import matplotlib.ticker as plticker

lit_chem =  "data/lit_chemistry_corrected_Fe_Ti_Mg_Ca_Na.tsv"

ff = fits.open("/Users/adamrains/Dropbox/code/plumage/data/El-Badry_all_columns_catalog.fits.gz")
data = ff[1].data

# -----------------------------------------------------------------------------
# Import table
# -----------------------------------------------------------------------------
binary_tab = Table(data)
binary_df = binary_tab.to_pandas()

binary_df["source_id_dr3"] = binary_df["source_id1"].values.astype(str)
binary_df.set_index("source_id_dr3", inplace=True)

# Compute M_G
dist2 = 1000 / binary_df["parallax2"].values
binary_df["M_G_2"] = \
    binary_df["phot_g_mean_mag2"].values - 5*np.log10(dist2/10)

# -----------------------------------------------------------------------------
# Masks
# -----------------------------------------------------------------------------
# BP-RP limits to ensure F/G/K primaries
prim_mask = data["bp_rp1"] <= 1.2
#np.logical_and(data["bp_rp1"] > 1.2, data["bp_rp1"] < 2.5)

# BP-RP limits to ensure K/M secondaries
sec_mask = np.logical_and(data["bp_rp2"] > 1.2, data["bp_rp2"] < 5)

# Declination limit to target southern systems we don't already have
dec_mask = np.logical_and(data["dec1"] < 90, data["dec1"] >= -90)

# Parallax mask to target nearby stars
plx_mask = data["parallax1"] > 1.0

# Magnitude mask to ensure the stars are observable
mag_mask = data["phot_rp_mean_mag1"] > 0
#np.logical_and(
#   data["phot_rp_mean_mag1"] < 15, data["phot_rp_mean_mag2"] < 16)

# RUWE cut for primaries
ruwe_mask_prim = data["ruwe1"] <= 1.4

# RUWE cut for secondaries
ruwe_mask_sec = data["ruwe2"] <= 1.4

# Secondary Parallax precision
parallax_error = data["parallax_error2"] / data["parallax2"]
unreliable_plx = parallax_error > 0.01

# Separation cut to ensure they're widely enough separated to observe. This
# column is in degrees, so we multiply by 3600 to get to arcseconds.
sep_mask = data["pairdistance"]*3600 > 3

# 'Probability' of chance alignment per Eqn 8 of paper.
chance_align_mask = data["R_chance_align"] < 0.01

# Exclude binaries that we already know about
cpm_prim = pu.load_info_cat(
    path="data/cpm_primaries_dr3.tsv",
    make_observed_col_bool_on_yes=False,
    use_mann_code_for_masses=False,
    gdr="dr3",
    has_2mass=False,)

combined_mask = np.all((prim_mask, sec_mask, dec_mask, plx_mask, mag_mask,
                        ruwe_mask_prim, ruwe_mask_sec, sep_mask,
                        chance_align_mask, ~unreliable_plx), axis=0)

binary_df = binary_df[combined_mask].copy()

is_known_binary = np.array(
    [sid in cpm_prim.index.values for sid in binary_df.index.values])

# -----------------------------------------------------------------------------
# Crossmatching with [Fe/H] and [X/Fe] catalogues
# -----------------------------------------------------------------------------
binary_tab = Table(data[combined_mask])
binary_df = binary_tab.to_pandas()

binary_df["source_id_dr3"] = binary_df["source_id1"].values.astype(str)
binary_df.set_index("source_id_dr3", inplace=True)

# Calculate pair distance in arcsec instead
binary_df["pairdistance_arcsec"] = binary_df["pairdistance"]*3600

# Compute secondary absolute magnitudes
dist_sec = 1000 / binary_df["parallax2"].values
binary_df["dist2"] = dist_sec
binary_df["M_G_sec"] = binary_df["phot_g_mean_mag2"] - 5*np.log10(dist_sec/10)

lit_chem_df = pd.read_csv(
    lit_chem,
    sep="\t",
    dtype={"source_id_dr3":str},)
lit_chem_df.set_index("source_id_dr3", inplace=True)

lit_chem_df["has_chem"] = True

binary_df_join = binary_df.join(lit_chem_df, "source_id_dr3", rsuffix="_chem", how="inner")

#------------------------------------------------------------------------------
# Collate [Fe/H]
#------------------------------------------------------------------------------
# Select adopted [Fe/H] values which are needed for empirical Teff relations
feh_info_all = []

mid_K_BP_RP_bound = 1.7
mid_K_MKs_bound = 5

#ABUND_ORDER_K = ["VF05", "B16", "M13", "A12", "L18", "M18", "R07", "D19", "RB20", "M14"]
ABUND_ORDER_K = ["M15", "G14b", "G14a", "D19", "RA12", "VF05", "B16", "M13", "A12", "L18", "M18", "R07", "D19", "RB20", "M14"]

# Add dummy magnitude information to treat stars as mid-K
binary_df_join["K_mag_abs"] = 4
binary_df_join["BP-RP_dr3"] = 1.5
binary_df_join["is_cpm"] = False

for star_i, (source_id, star_info) in enumerate(binary_df_join.iterrows()): 
    feh_info = params.select_abund_label(
        star_info=star_info,
        abund="Fe_H",
        mid_K_BP_RP_bound=mid_K_BP_RP_bound,
        mid_K_MKs_bound=mid_K_MKs_bound,
        abund_order_k=ABUND_ORDER_K,
        abund_order_m=None,
        abund_order_binary=None,
        mid_K_BP_RP_trustworthy_X_Fe=1.2,)
    feh_info_all.append(feh_info)

feh_info_all = np.vstack(feh_info_all)

feh_values = feh_info_all[:,0].astype(float)
e_feh_values = feh_info_all[:,1].astype(float)
feh_source = feh_info_all[:,2]

# TEMPORARY archive of chosen [Fe/H]
binary_df_join["Fe_H_adopt"] = feh_values.copy()
binary_df_join["e_Fe_H_adopt"] = e_feh_values.copy()
binary_df_join["Fe_H_adopt_source"] = feh_source

binary_df_join.sort_values(by="ra1", inplace=True)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
"""
fig, axes = plt.subplots(nrows=2)
axes[0].hist(dd["phot_bp_mean_mag1"], bins=100, alpha=0.5, label="Primary")
axes[0].hist(dd["phot_bp_mean_mag2"], bins=100, alpha=0.5, label="Secondary")
axes[1].hist(dd["bp_rp1"], bins=100, alpha=0.5, label="Primary")
axes[1].hist(dd["bp_rp2"], bins=100, alpha=0.5, label="Secondary")
axes[0].set_title("Dec < -25, plx > 1.5, 0.5 < BP-RP (Prim) < 1.2, 1.5 < BP-RP (Sec) < 5, BP < 15", fontsize="small")
axes[0].set_xlabel("BPmag")
axes[1].set_xlabel("BP-RP")
plt.tight_layout()
axes[1].legend()
"""

is_known_binary = np.array(
    [sid in cpm_prim.index.values for sid in binary_df_join.index.values])

plt.close("all")
fig, axes = plt.subplots(ncols=5, figsize=(16,5))

# Panel 1: [Fe/H] histogram
axes[0].hist(
    binary_df_join["Fe_H_adopt"].values,
    bins=20,
    label="All",
    alpha=0.6)

axes[0].hist(
    binary_df_join["Fe_H_adopt"].values[~is_known_binary],
    bins=20,
    label="New",
    alpha=0.6)

axes[0].set_xlabel("[Fe/H]")
axes[0].legend(fontsize="xx-small")

# Panel 2: secondary BP-RP histogram
axes[1].hist(
    binary_df_join["bp_rp2"].values,
    bins=15,
    label="All ({})".format(is_known_binary.size),
    alpha=0.6)

axes[1].hist(
    binary_df_join["bp_rp2"].values[~is_known_binary],
    bins=15,
    label="New ({})".format(np.sum(~is_known_binary)),
    alpha=0.6)

axes[1].set_xlabel("BP-RP")

# Panel 3: secondary BP-RP vs [Fe/H]
axes[2].plot(
    binary_df_join["bp_rp2"].values,
    binary_df_join["Fe_H_adopt"].values,
    linestyle="",
    marker="o",
    ms="2",
    #markeredgecolor="k",
    label="All ({})".format(is_known_binary.size),)

axes[2].plot(
    binary_df_join["bp_rp2"].values[~is_known_binary],
    binary_df_join["Fe_H_adopt"].values[~is_known_binary],
    linestyle="",
    marker="o",
    ms="2",
    #markeredgecolor="k",
    label="New ({})".format(np.sum([~is_known_binary])),)

axes[2].set_xlabel("BP-RP")
axes[2].set_ylabel("[Fe/H]")

# Panel 3: distances
axes[3].hist(
    binary_df_join["dist2"].values,
    bins=15,
    label="All ({})".format(is_known_binary.size),
    alpha=0.6)

axes[3].hist(
    binary_df_join["dist2"].values[~is_known_binary],
    bins=15,
    label="New ({})".format(np.sum(~is_known_binary)),
    alpha=0.6)

axes[3].set_xlabel("Distance (pc)")

# Panel 4: CMD
scatter = axes[4].scatter(
    binary_df_join["bp_rp2"].values,
    binary_df_join["M_G_sec"].values, 
    zorder=1,
    c=binary_df_join["Fe_H_adopt"].values,
    alpha=0.9,
    label="All ({})".format(is_known_binary.size),
    cmap="viridis",)

_ = axes[4].scatter(
    binary_df_join["bp_rp2"].values[is_known_binary],
    binary_df_join["M_G_sec"].values[is_known_binary],
    marker="o",
    zorder=2,
    edgecolor="black",
    facecolors="none",
    alpha=0.9,
    label="Known ({})".format(np.sum(is_known_binary)),
    cmap="viridis",)

axes[4].legend(fontsize="xx-small")
cb = fig.colorbar(scatter, ax=axes[4])
cb.set_label("[Fe/H]")
axes[4].set_xlabel(r"$BP-RP$", fontsize="large")
axes[4].set_ylabel(r"$M_G$", fontsize="large")

ymin, ymax = axes[4].get_ylim()
axes[4].set_ylim((ymax, ymin))

axes[1].xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
axes[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))
axes[2].xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
axes[2].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))
axes[4].xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
axes[4].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))


plt.tight_layout()

#------------------------------------------------------------------------------
# Make CPM list for MIKE
#------------------------------------------------------------------------------
source_ids = []
names = []
kinds = []
companions = []
ras = []
decs = []
pm_ras = []
pm_decs = []
G_mags = []
BP_mags = []
RP_mags = []
bp_rps = []
fehs = []
separation = []

columns = ["source_id_dr3", "simbad_name", "kind", "companion", "ra_dr3",
           "dec_dr3", "pmra_dr3", "pmdec_dr3", "G_mag_dr3", "BP_mag_dr3",
           "RP_mag_dr3", "BP-RP_dr3", "Fe_H", "sep_arcsec"]

for star_i, (source_id, star_info) in enumerate(binary_df_join.iterrows()):
    # Secondary
    source_ids.append(star_info["source_id2"])
    names.append("")
    kinds.append("secondary")
    companions.append(star_info["source_id1"])
    ras.append(star_info["ra2"])
    decs.append(star_info["dec2"])
    pm_ras.append(star_info["pmra2"])
    pm_decs.append(star_info["pmdec2"])
    G_mags.append(np.round(star_info["phot_g_mean_mag2"], 2))
    BP_mags.append(np.round(star_info["phot_bp_mean_mag2"], 2))
    RP_mags.append(np.round(star_info["phot_rp_mean_mag2"], 2))
    bp_rps.append(np.round(star_info["bp_rp2"], 2))
    fehs.append(np.round(star_info["Fe_H_adopt"], 2))
    separation.append(np.round(star_info["pairdistance_arcsec"], 2))

    # Primary
    source_ids.append(star_info["source_id1"])
    names.append("")
    kinds.append("primary")
    companions.append(star_info["source_id2"])
    ras.append(star_info["ra1"])
    decs.append(star_info["dec1"])
    pm_ras.append(star_info["pmra1"])
    pm_decs.append(star_info["pmdec1"])
    G_mags.append(np.round(star_info["phot_g_mean_mag1"], 2))
    BP_mags.append(np.round(star_info["phot_bp_mean_mag1"], 2))
    RP_mags.append(np.round(star_info["phot_rp_mean_mag1"], 2))
    bp_rps.append(np.round(star_info["bp_rp1"], 2))
    fehs.append(np.round(star_info["Fe_H_adopt"], 2))
    separation.append(np.round(star_info["pairdistance_arcsec"], 2))

data = [source_ids, names, kinds, companions, ras, decs, pm_ras, pm_decs, 
        G_mags, BP_mags, RP_mags, bp_rps, fehs, separation]

cpm_df_unified = pd.DataFrame(data=np.stack(data).T, columns=columns)
cpm_df_unified.set_index("source_id_dr3", inplace=True)

cpm_df_unified.to_csv("data/el-badry_mike_unified.tsv", sep="\t")
