"""Script to develop a M_K-(Bp-K) based [Fe/H] relation using Gaia DR3 and 
2MASS data, and literature [Fe/H] values for binary systems.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plumage.utils as putils
import plumage.parameters as params
from scipy.optimize import least_squares
import matplotlib.ticker as plticker
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
# Mask settings
apply_ruwe_cut = True
use_only_aaa_2mass = True
rule_out_gaia_dups = True

enforce_system_useful = True

# Primary data quality + suitability
enforce_primary_ruwe = False
enforce_primary_bp_rp_colour = True

# Secondary data quality
enforce_secondary_ruwe = True
enforce_secondary_2mass_unblended = True
enforce_secondary_2mass_aaa_quality = True
enforce_secondary_bp_rp_colour = True

# Distance and velocity consistency
enforce_parallax_consistency = True
enforce_pm_consistency = True
enforce_rv_consistency = True

allow_exceptions = False

DEFAULT_FEH_UNCERTAINTY = 0.2
DELTA_PARALLAX = 0.2
DELTA_NORM_PM = 5
DELTA_RV = 5

RUWE_THRESHOLD = 1.4
BP_RP_BOUNDS_PRIM = (-100, 1.5)
BP_RP_BOUNDS_SEC = (2.5, 4.5)

# Which colour relation to use
c_rel = "BP-K"
e_c_rel = "e_{}".format(c_rel)
c_label = "BP-K_S"

# Whether to use only Mann+15 for the main sequence fit, or both M15 and CPM
only_use_m15_ms = False

# Polynnomial order when fitting main sequence
main_seq_poly_deg = 4

# Whether the offset from the MS is M_Ks per JA09, or offset per SL10
use_bp_k_offset = True

# Whether to do the [Fe/H] fit with just the CPM sample, or just the M15 sample
fit_feh_with_cpm = True

# Or both samples
fit_feh_with_both = False

# Polynomial order of the fitted offset function
offset_poly_order = 2

# Whether to apply an additional linear correction to the residuals
apply_correction = False

# Whether to plot remaining residual trend
plot_resid_trend = False

# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
# Files
tsv_primaries = "data/cpm_primaries_dr3.tsv"
tsv_secondaries = "data/cpm_secondaries_dr3.tsv"
tsv_mann15 = "data/mann15_all_dr3.tsv"

# Import primary info
cpm_prim = putils.load_info_cat(
    tsv_primaries,
    clean=False,
    allow_alt_plx=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="_dr3",
    has_2mass=False,)

cpm_prim.reset_index(inplace=True)
cpm_prim.rename(columns={"source_id_dr3":"source_id_dr3_prim"}, inplace=True)
cpm_prim.set_index("prim_name", inplace=True)

# Import secondary info
cpm_sec = putils.load_info_cat(
    tsv_secondaries,
    clean=False,
    allow_alt_plx=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="_dr3",
    has_2mass=True,)

# Merge on prim_name
cpm_join = cpm_sec.join(cpm_prim, "prim_name", rsuffix="_prim")

# Import Mann+15 standards
m15_data = putils.load_info_cat(
    tsv_mann15,
    clean=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="_dr3",)

# Remove any entries without gaia photometry or parallaxes
nan_mask = np.logical_and(
    ~np.isnan(m15_data["BP_mag_dr3"]),
    ~np.isnan(m15_data["plx_dr3"]))
ruwe_mask = m15_data["ruwe_dr3"] <= RUWE_THRESHOLD
m15_data = m15_data[np.logical_and(nan_mask, ruwe_mask)]

# -----------------------------------------------------------------------------
# Masking
# -----------------------------------------------------------------------------
# Now only keep pairs matching our criteria
keep_mask = np.full(len(cpm_join), True)

# And also create a mask for use when fitting the main sequence. This mask
# doesn't need to be as strict, as we don't require the system to be associated
# or to have good [Fe/H] values, only that we have good photometry and reliable
# astronomy
ms_keep_mask = np.full(len(cpm_join), True)

# Enforce the system has not been marked as not useful
if enforce_system_useful:
    keep_mask = np.logical_and(
        keep_mask,
        cpm_join["useful"] != "no")

# Primary RUWE <= 1.4
if enforce_primary_ruwe:
    keep_mask = np.logical_and(
        keep_mask,
        cpm_join["ruwe_dr3_prim"] <= RUWE_THRESHOLD)

# Primary BP-RP
if enforce_primary_bp_rp_colour:
    bp_rp_mask = np.logical_and(
        cpm_join["BP-RP_dr3_prim"] > BP_RP_BOUNDS_PRIM[0],
        cpm_join["BP-RP_dr3_prim"] < BP_RP_BOUNDS_PRIM[1])
    keep_mask = np.logical_and(keep_mask, bp_rp_mask)

# Secondary RUWE <= 1.4
if enforce_secondary_ruwe:
    keep_mask = np.logical_and(
        keep_mask,
        cpm_join["ruwe_dr3"] <= RUWE_THRESHOLD)

    ms_keep_mask = np.logical_and(
        ms_keep_mask,
        cpm_join["ruwe_dr3"] <= RUWE_THRESHOLD)

# Secondary 2MASS quality
if enforce_secondary_2mass_aaa_quality:
    keep_mask = np.logical_and(
        keep_mask,
        cpm_join["Qflg"] == "AAA")
    
    ms_keep_mask = np.logical_and(
        ms_keep_mask,
        cpm_join["Qflg"] == "AAA")

# Secondary 2MASS unblended
if enforce_secondary_2mass_aaa_quality:
    keep_mask = np.logical_and(
        keep_mask,
        cpm_join["blended_2mass"] != "yes")

    ms_keep_mask = np.logical_and(
        ms_keep_mask,
        cpm_join["blended_2mass"] != "yes")

# Secondary BP-RP colour
if enforce_secondary_bp_rp_colour:
    bp_rp_mask = np.logical_and(
        cpm_join["BP-RP_dr3"] > BP_RP_BOUNDS_SEC[0],
        cpm_join["BP-RP_dr3"] < BP_RP_BOUNDS_SEC[1])
    keep_mask = np.logical_and(keep_mask, bp_rp_mask)

    # Enforce upper bound for the main sequence mask, but not the lower one
    ms_keep_mask = np.logical_and(
        ms_keep_mask,
        cpm_join["BP-RP_dr3"] > BP_RP_BOUNDS_SEC[0])

# Parallaxes are consistent
if enforce_parallax_consistency:
    keep_mask = np.logical_and(
        keep_mask,
        np.abs(cpm_join["plx_dr3"]-cpm_join["plx_dr3_prim"]) < DELTA_PARALLAX)

# Proper motions are consistent
if enforce_parallax_consistency:
    total_pm_prim = np.sqrt(
        cpm_join["pmra_dr3_prim"]**2+cpm_join["pmdec_dr3_prim"]**2)
    total_pm_sec = np.sqrt(cpm_join["pmra_dr3"]**2+cpm_join["pmdec_dr3"]**2)
    pm_norm_diff = \
        total_pm_prim/cpm_join["dist_prim"] - total_pm_sec/cpm_join["dist"]

    keep_mask = np.logical_and(
        keep_mask,
        pm_norm_diff < DELTA_NORM_PM)

# RVs are consistent
if enforce_rv_consistency:
    keep_mask = np.logical_and(
        keep_mask,
        np.abs(cpm_join["rv_dr3_prim"]-cpm_join["rv_dr3"]) < DELTA_RV)

#  Allow exceptions:
if allow_exceptions:
    # Keep VB12, our most metal poor star
    star_i = int(np.argwhere(cpm_join.index.values == "2411728182287010816"))
    keep_mask[star_i] = True

# Apply mask
cpm_selected = cpm_join[keep_mask].copy()

cpm_ms_selected = cpm_selected.copy()
#cpm_ms_selected = cpm_join[ms_keep_mask].copy()

# -----------------------------------------------------------------------------
# [Fe/H] Systematics
# -----------------------------------------------------------------------------
# Correct [Fe/H] systematics
feh_corr_all = []
e_feh_corr_all = []

# [Fe/H] offsets from Mann+13
feh_offsets = {
    "TW":0.0,
    "VF05":0.0,
    "CFHT":0.0,
    "C01":0.02,
    "M04" :0.04,
    "LH05":0.02,
    "T05":0.01,
    "B06":0.07,
    "Ra07":0.08,
    "Ro07":0.00,
    "F08":0.03,
    "C11":0.00,
    "S11":0.03,
    "N12":0.00,
    "M18":0.0,      # TODO Confirm this
    "Sou06":0.0,    # TODO Confirm this
    "Soz09":0.0,    # TODO Confirm this
    "M14":0.0,      # Probably safe to assume this is zero?
}

# Citations
citations = []

feh_citations = {
    "TW":"mann_spectro-thermometry_2013",
    "VF05":"valenti_spectroscopic_2005",
    "CFHT":"",
    "C01":"",
    "M04" :"mishenina_correlation_2004",
    "LH05":"luck_stars_2005",
    "T05":"",
    "B06":"bean_accurate_2006",
    "Ra07":"ramirez_oxygen_2007",
    "Ro07":"robinson_n2k_2007",
    "F08":"fuhrmann_nearby_2008",
    "C11":"casagrande_new_2011",
    "S11":"da_silva_homogeneous_2011",
    "N12":"neves_metallicity_2012",
    "M18":"montes_calibrating_2018",
    "Sou06":"",
    "Soz09":"",
    "M14":"",
}

ignored = []

for source_id, cpm_row in cpm_selected.iterrows():
    # More stars from Mann+13, check that first
    if ~np.isnan(cpm_row["feh_prim_m13"]):
        ref = cpm_row["ref_feh_prim_m13"]
        feh_corr = cpm_row["feh_prim_m13"] + feh_offsets[ref]
        e_feh_corr = cpm_row["e_feh_prim_m13"]
        citation = feh_citations[ref]

    # Then check those from Newton+14
    elif ~np.isnan(cpm_row["feh_prim_n14"]):
        feh_corr = cpm_row["feh_prim_n14"] + feh_offsets["VF05"]
        e_feh_corr = cpm_row["e_feh_prim_n14"]
        citation = feh_citations[cpm_row["feh_prim_ref_n14"]]
    
    # Now grab from Montes+2018
    elif ~np.isnan(cpm_row["Fe_H_m18"]):
        feh_corr = cpm_row["Fe_H_m18"] + feh_offsets["M18"]
        e_feh_corr = cpm_row["eFe_H_m18"]
        citation = feh_citations["M18"]

    # Then check Mann+2014 for VLM dwarfs
    elif ~np.isnan(cpm_row["feh_m14"]):
        feh_corr = cpm_row["feh_m14"] + feh_offsets["M14"]
        e_feh_corr = cpm_row["e_feh_m14"]
        citation = feh_citations["M18"]

    # Then check other entries
    elif cpm_row["feh_prim_ref_other"] == "Ra07":
        feh_corr = cpm_row["feh_prim_other"] + feh_offsets["Ra07"]
        e_feh_corr = cpm_row["e_feh_prim_other"]
        citation = feh_citations["Ra07"]

    # Don't know systematics, ignore
    else:
        feh_corr = np.nan
        e_feh_corr = np.nan
        #ignored.append(source_id)
        #citation = "--"
    
    # If the uncertainty is a nan, then set a reasonable default
    #if ((not np.isnan(feh_corr) and np.isnan(e_feh_corr)) or e_feh_corr == 0):
    if np.isnan(feh_corr) or np.isnan(e_feh_corr) or e_feh_corr == 0:
        print("Setting default uncertainty for {}".format(source_id))
        feh_corr = np.nan
        e_feh_corr = np.nan
        ignored.append(source_id)
        citation = "--"

    feh_corr_all.append(feh_corr)
    e_feh_corr_all.append(e_feh_corr)
    citations.append(citation)

# All done
print("Ignored {} stars: {}".format(len(ignored), str(ignored)))
cpm_selected["feh_corr"] = feh_corr_all
cpm_selected["e_feh_corr"] = e_feh_corr_all
cpm_selected["citation"] = citations

# Now mask out stars which we've excluded
cpm_info_feh_corr = cpm_selected[~np.isnan(feh_corr_all)].copy()
print("{} stars in total".format(len(cpm_info_feh_corr)))

# -----------------------------------------------------------------------------
# Fit polynomial to mean MS
# -----------------------------------------------------------------------------
if only_use_m15_ms:
    bp_k = m15_data[c_rel]
    e_bp_k = m15_data[e_c_rel]
    k_mag_abs = m15_data["K_mag_abs"]
    e_k_mag_abs = m15_data["e_K_mag_abs"]
    ms_fehs = m15_data["[Fe/H]"]
    print("Using only Mann+15 sample for mean-MS, {} stars".format(len(bp_k)))

else:
    bp_k = np.concatenate((m15_data[c_rel], cpm_ms_selected[c_rel]))
    e_bp_k = np.concatenate((m15_data[e_c_rel], cpm_ms_selected[e_c_rel]))
    k_mag_abs = np.concatenate((m15_data["K_mag_abs"], cpm_ms_selected["K_mag_abs"]))
    e_k_mag_abs = np.concatenate((m15_data["e_K_mag_abs"], cpm_ms_selected["e_K_mag_abs"]))
    ms_fehs = np.concatenate((m15_data["[Fe/H]"], cpm_selected["feh_corr"]))
    print("Using combined sample for mean-MS, {} stars".format(len(bp_k)))

# Use absolute K band magnitude as offset
if not use_bp_k_offset:
    print("M_Ks offset per JA09")
    ms_coeff = np.polynomial.polynomial.polyfit(
        x=bp_k,
        y=k_mag_abs,
        deg=main_seq_poly_deg,
        w=1/e_k_mag_abs,)

# Instead use the Bp-K colour as the offset (per Schlaufman & Laughlin 2010)
else:
    print("BP-Ks offset per SL10")
    ms_coeff = np.polynomial.polynomial.polyfit(
        x=k_mag_abs,
        y=bp_k,
        deg=main_seq_poly_deg,
        w=1/e_bp_k,)

main_seq = np.polynomial.polynomial.Polynomial(ms_coeff)

# Now strip away the Mannn+15 targets who fall outside the Bp-Rp range
bp_rp_mask = np.logical_and(
    m15_data["BP-RP_dr3"] > BP_RP_BOUNDS_SEC[0],
    m15_data["BP-RP_dr3"] < BP_RP_BOUNDS_SEC[1])
m15_data = m15_data[bp_rp_mask]

# -----------------------------------------------------------------------------
# Fit [Fe/H] offset
# -----------------------------------------------------------------------------
if fit_feh_with_cpm and not fit_feh_with_both:
    print("[Fe/H] fit to only CPM sample")
    opt_res = params.fit_feh_model(
        bp_k=cpm_info_feh_corr[c_rel],
        k_mag_abs=cpm_info_feh_corr["K_mag_abs"],
        feh=cpm_info_feh_corr["feh_corr"],
        e_feh=cpm_info_feh_corr["e_feh_corr"],
        main_seq=main_seq,
        offset_poly_order=offset_poly_order,
        use_bp_k_offset=use_bp_k_offset,)

elif not fit_feh_with_both:
    print("[Fe/H] fit to only Mann+15 sample")
    opt_res = params.fit_feh_model(
        bp_k=m15_data[c_rel],
        k_mag_abs=m15_data["K_mag_abs"],
        feh=m15_data["[Fe/H]"],
        e_feh=m15_data["e_[Fe/H]"],
        main_seq=main_seq,
        offset_poly_order=offset_poly_order,
        use_bp_k_offset=use_bp_k_offset,)

else:
    print("[Fe/H] fit to both CPM and Mann+15 sample")
    opt_res = params.fit_feh_model(
        bp_k=np.concatenate((m15_data[c_rel], cpm_info_feh_corr[c_rel])),
        k_mag_abs=np.concatenate((m15_data["K_mag_abs"], cpm_info_feh_corr["K_mag_abs"])),
        feh=np.concatenate((m15_data["[Fe/H]"], cpm_info_feh_corr["feh_corr"])),
        e_feh=np.concatenate((m15_data["e_[Fe/H]"], cpm_info_feh_corr["e_feh_corr"])),
        main_seq=main_seq,
        offset_poly_order=offset_poly_order,
        use_bp_k_offset=use_bp_k_offset,)

# Get coeffs and function
offset_coeff = opt_res["x"]
feh_poly = np.polynomial.polynomial.Polynomial(offset_coeff)

# Calculate rchi2
rchi2 = np.sum(opt_res["fun"]**2) / (len(opt_res["fun"])-offset_poly_order)
print("rchi^2 = {:0.2f}".format(rchi2))

# Compute [Fe/H] from relation for CPM and M15 sample
feh_pred_cpm = params.calc_photometric_feh(
    offset_coeff,
    cpm_info_feh_corr[c_rel],
    cpm_info_feh_corr["K_mag_abs"],
    main_seq,
    use_bp_k_offset,)

feh_pred_m15 = params.calc_photometric_feh(
    offset_coeff,
    m15_data[c_rel],
    m15_data["K_mag_abs"],
    main_seq,
    use_bp_k_offset,)

# Calculate residuals
feh_cpm_resid = cpm_info_feh_corr["feh_corr"] - feh_pred_cpm
feh_m15_resid = m15_data["[Fe/H]"] - feh_pred_m15

# Fit linear line to residuals
resid_cpm_coeff = np.polynomial.polynomial.polyfit(
    x=cpm_info_feh_corr["feh_corr"],
    y=feh_cpm_resid,
    deg=1,
    w=1/cpm_info_feh_corr["e_feh_corr"],)

resid_m15_coeff = np.polynomial.polynomial.polyfit(
    x=m15_data["[Fe/H]"],
    y=feh_m15_resid,
    deg=1,
    w=1/m15_data["e_[Fe/H]"],)

# Calculate trend in residuals, update coefficients
params_corr = [offset_coeff[0]*(1+resid_cpm_coeff[1]) + resid_cpm_coeff[0],
               offset_coeff[1] * (1+resid_cpm_coeff[1])]

feh_poly_corr = np.polynomial.polynomial.Polynomial(params_corr)

# Correct for trend in residuals
if apply_correction:
    feh_pred_cpm = feh_pred_cpm + (resid_cpm_coeff[1]*feh_pred_cpm + resid_cpm_coeff[0])
    feh_pred_m15 = feh_pred_m15 + (resid_m15_coeff[1]*feh_pred_m15 + resid_m15_coeff[0])

    # Recalculate residuals
    feh_cpm_resid = cpm_info_feh_corr["feh_corr"] - feh_pred_cpm
    cpm_info_feh_corr["feh_resid"] = feh_cpm_resid
    feh_m15_resid = m15_data["[Fe/H]"] - feh_pred_m15
    m15_data["feh_resid"] = feh_m15_resid

    # Recalculate resid fit
    resid_cpm_coeff = np.polynomial.polynomial.polyfit(
        x=cpm_info_feh_corr["feh_corr"],
        y=feh_cpm_resid,
        deg=1,
        w=1/cpm_info_feh_corr["e_feh_corr"],)

    resid_m15_coeff = np.polynomial.polynomial.polyfit(
        x=m15_data["[Fe/H]"],
        y=feh_m15_resid,
        deg=1,
        w=1/m15_data["e_[Fe/H]"],)

# Calculate mean, std
feh_offset_cpm_mean = np.mean(feh_cpm_resid)
feh_offset_cpm_std = np.std(feh_cpm_resid)

feh_offset_m15_mean = np.mean(feh_m15_resid)
feh_offset_m15_std = np.std(feh_m15_resid)

# Dump coefficients
np.savetxt("data/phot_feh_rel_ms_coeff_dr3.csv", ms_coeff)
np.savetxt("data/phot_feh_rel_offset_coeff_dr3.csv", offset_coeff)

# Format for paper
print("MS Coeff:",
    "$a_3={:0.5f}$, $a_2={:0.5f}$, $a_1={:0.5f}$, and $a_0={:0.5f}$".format(
    *ms_coeff[::-1]))
print("Offset Coeff:", 
    "$b_1={:0.5f}$, and $b_0={:0.5f}$".format(*offset_coeff[::-1]))

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")

# Plot #1 - fit to mean main sequence
fig_ms, ms_ax = plt.subplots(1)

# Plot #2 - 3 panel results
fig = plt.figure()
gs = fig.add_gridspec(nrows=2, ncols=3)
cpm_ax = fig.add_subplot(gs[0, 2])
offset_fit_ax = fig.add_subplot(gs[1, 2])
cpm_feh_ax = fig.add_subplot(gs[:, :2])

# Plot #3 - testing relation on Mann+15 stars
fig_m15, m15_ax = plt.subplots(1)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot 1 - Mean main sequence fit
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = ms_ax.scatter(
    bp_k,
    k_mag_abs,
    c=ms_fehs,
    zorder=2,)
ms_ax.errorbar(
    bp_k,
    k_mag_abs,
    xerr=e_bp_k,
    yerr=e_k_mag_abs,
    fmt=".",
    zorder=1,)

cb = fig.colorbar(sc, ax=ms_ax)
cb.set_label(r"[Fe/H]")
ms_ax.set_xlabel(r"$({})$".format(c_label))
ms_ax.set_ylabel(r"$M_{K_S}$")

ms_ax.set_title(
    "Main Sequence Fit ({:0.0f}th order poly, {:0.0f} stars)".format(
        main_seq_poly_deg, len(bp_k)))

# Set limits, invert
xlims = ms_ax.get_xlim()
ylims = ms_ax.get_ylim()

ms_ax.set_ylim(ylims[1], ylims[0])

# Plot main sequence fit
# Use absolute K band offset from MS (per Johnson & Apps 2009)
if not use_bp_k_offset:
    xx = np.arange(xlims[0], xlims[1], 0.1)
    cpm_ax.plot(xx, main_seq(xx), "r--")
    ms_ax.plot(xx, main_seq(xx), "r--")

# Use (Bp-K) colour offset from MS (per Schlaufman & Laughlin 2010)
else:
    xx = np.arange(ylims[0], ylims[1], 0.1)
    cpm_ax.plot(main_seq(xx), xx, "r--")
    ms_ax.plot(main_seq(xx), xx, "r--")

# Wrap up
#fig_ms.set_size_inches(9, 4.5)
fig_ms.tight_layout() 
fig_ms.savefig("paper/phot_feh_rel_mean_ms.pdf")
fig_ms.savefig("paper/phot_feh_rel_mean_ms.png", dpi=500)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot 2, Panel #1 - Scatter about adopted mean main sequence
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = cpm_ax.scatter(
    cpm_info_feh_corr[c_rel], 
    cpm_info_feh_corr["K_mag_abs"], 
    c=cpm_info_feh_corr["feh_corr"],
    zorder=2,)

msk = ~np.isnan(cpm_selected["plx_alt"])

cpm_ax.scatter(
    cpm_info_feh_corr[c_rel][msk], 
    cpm_info_feh_corr["K_mag_abs"][msk], 
    #c=cpm_info_feh_corr["feh_corr"][msk],
    marker="*",
    facecolors='none',
    edgecolors="black",
    linewidths=0.2,
    zorder=2,)

cpm_ax.errorbar(
    cpm_info_feh_corr[c_rel],
    cpm_info_feh_corr["K_mag_abs"],
    xerr=cpm_info_feh_corr[e_c_rel],
    yerr=cpm_info_feh_corr["e_K_mag_abs"],
    fmt=".",
    zorder=1,)

# Set limits, invert
xlims = cpm_ax.get_xlim()
ylims = cpm_ax.get_ylim()

cpm_ax.set_xlim(xlims[0], xlims[1])
cpm_ax.set_ylim(ylims[1], ylims[0])

cb = fig.colorbar(sc, ax=cpm_ax)
cb.set_label(r"[Fe/H]", fontsize="large")
#cpm_ax.set_title("CPM Sample")
cpm_ax.set_xlabel(r"$({})$".format(c_label), fontsize="large")
cpm_ax.set_ylabel(r"$M_{K_S}$", fontsize="large")

cpm_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
cpm_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))

cpm_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Plot 2, Panel #2 - BP-Ks fit
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if not use_bp_k_offset:
    offset = cpm_info_feh_corr["K_mag_abs"] - main_seq(cpm_info_feh_corr[c_rel])

    offset_fit_ax.set_xlabel(r"$\Delta M_{K_S}$")

if use_bp_k_offset:
    offset = cpm_info_feh_corr[c_rel] - main_seq(cpm_info_feh_corr["K_mag_abs"])
    
    offset_fit_ax.set_xlabel(r"$\Delta ({})$".format(c_label))

sc = offset_fit_ax.scatter(
    offset,
    cpm_info_feh_corr["feh_corr"],
    c=cpm_info_feh_corr["BP-RP_dr3"],
    cmap="plasma",
    zorder=2,
    label=None,)
    
offset_fit_ax.errorbar(
    offset,
    cpm_info_feh_corr["feh_corr"],
    yerr=cpm_info_feh_corr["e_feh_corr"],
    fmt=".",
    zorder=1,
    label=None,) 

# Plot fits
xx = np.arange(offset.min(), offset.max(), 0.01)
offset_fit_ax.plot(xx, feh_poly(xx), "r--", label="Fit (uncorrected)")

if apply_correction:
    offset_fit_ax.plot(xx, feh_poly_corr(xx), "-.", c="cornflowerblue", 
        label="Fit (adopted)")
cb = fig.colorbar(sc, ax=offset_fit_ax)
cb.set_label(r"$(BP-RP)$", fontsize="large")
offset_fit_ax.set_ylabel("[Fe/H] (Primary)", fontsize="large")
offset_fit_ax.legend(fontsize="x-small")

offset_fit_ax.set_ylim([-1.5,0.75])

plt.setp(offset_fit_ax.get_xticklabels(), rotation="vertical")

offset_fit_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
offset_fit_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))
offset_fit_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
offset_fit_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Plot 2, Panel #3 - [Fe/H] recovery
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = cpm_feh_ax.scatter(
    cpm_info_feh_corr["feh_corr"],
    feh_pred_cpm,
    c=cpm_info_feh_corr["BP-RP_dr3"],
    cmap="plasma",
    zorder=2,)

cpm_feh_ax.errorbar(
    cpm_info_feh_corr["feh_corr"],
    feh_pred_cpm,
    xerr=cpm_info_feh_corr["e_feh_corr"],
    yerr=feh_offset_cpm_std,
    fmt=".",
    zorder=1,)

xx = np.arange(-1.5,0.5,0.01)
cpm_feh_ax.plot(xx,xx,"k--")
cb = fig.colorbar(sc, ax=cpm_feh_ax)
cb.set_label(r"$(BP-RP)$", fontsize="large")
#cpm_feh_ax.set_xlabel("[Fe/H] (CPM)")
cpm_feh_ax.set_ylabel("[Fe/H] (fit)", fontsize="large")

cpm_feh_ax.text(
    x=-0.6, 
    y=0.9, 
    s=r"$\Delta{{\rm[Fe/H]}}={:0.2f}\pm {:0.2f}$".format(
        feh_offset_cpm_mean, feh_offset_cpm_std),
    horizontalalignment="center",
    fontsize="large")

cpm_feh_ax.text(
    x=-0.6, 
    y=0.6, 
    s=r"{:0.0f} stars".format(len(cpm_info_feh_corr)),
    horizontalalignment="center",
    fontsize="large")

# Residuals
divider = make_axes_locatable(cpm_feh_ax)
cpm_resid_ax = divider.append_axes("bottom", size="30%", pad=0)
cpm_feh_ax.figure.add_axes(cpm_resid_ax, sharex=cpm_feh_ax)
sc = cpm_resid_ax.scatter(
    cpm_info_feh_corr["feh_corr"],
    feh_cpm_resid,
    c=cpm_info_feh_corr["BP-RP_dr3"],
    cmap="plasma",
    zorder=2,)

cpm_resid_ax.errorbar(
    cpm_info_feh_corr["feh_corr"],
    feh_cpm_resid,
    xerr=cpm_info_feh_corr["e_feh_corr"],
    yerr=feh_offset_cpm_std,
    fmt=".",
    zorder=1,)

# Plot resid trend
if plot_resid_trend:
    fehs = np.arange(-1.5, 0.5, 0.01)
    cpm_resid_poly = np.polynomial.polynomial.Polynomial(resid_cpm_coeff)
    cpm_resid_ax.plot(fehs, cpm_resid_poly(fehs), "r--")

cpm_resid_ax.hlines(0, -1.5, 0.5, colors="k", linestyles="--")
cpm_resid_ax.set_xlabel("[Fe/H] (Primary)", fontsize="large")
cpm_resid_ax.set_ylabel("[Fe/H] Resid", fontsize="large")

cpm_feh_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_feh_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_feh_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

cpm_resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_resid_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

cpm_resid_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_resid_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))

# Wrap up
fig.set_size_inches(9, 4.5)
fig.tight_layout() 
fig.savefig("paper/phot_feh_rel.pdf")
fig.savefig("paper/phot_feh_rel.png", dpi=500)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Plot 3, Panel #1 - Mann+15 comp
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make residual axis
divider = make_axes_locatable(m15_ax)
m15_resid_ax = divider.append_axes("bottom", size="30%", pad=0)
m15_ax.figure.add_axes(m15_resid_ax, sharex=m15_ax)

sc = m15_ax.scatter(
    m15_data["[Fe/H]"],
    feh_pred_m15,
    c=m15_data["BP-RP_dr3"],
    cmap="plasma",
    zorder=2)

m15_ax.errorbar(
    m15_data["[Fe/H]"],
    feh_pred_m15,
    xerr=m15_data["e_[Fe/H]"],
    yerr=feh_offset_m15_std,
    fmt=".",
    zorder=1,)

# Plot resid trend
fehs = np.arange(-1.5, 0.5, 0.01)
m15_resid_poly = np.polynomial.polynomial.Polynomial(resid_m15_coeff)
m15_resid_ax.plot(fehs, m15_resid_poly(fehs), "r--")

xx = np.arange(-1.5,0.5,0.01)
m15_ax.plot(xx, xx, "k--")
cb = fig.colorbar(sc, ax=m15_ax)
cb.set_label(r"$(BP-RP)$")
#m15_feh_ax.set_xlabel("[Fe/H] (Mann+15)")
m15_ax.set_ylabel("[Fe/H] (fit)")

m15_ax.text(
    x=-0.25, 
    y=-1.0, 
    s=r"$\Delta[Fe/H]={:0.2f}\pm {:0.2f}$".format(
        feh_offset_m15_mean, feh_offset_m15_std),
    horizontalalignment="center")

# Resid
sc = m15_resid_ax.scatter(
    m15_data["[Fe/H]"],
    feh_m15_resid,
    c=m15_data["BP-RP_dr3"],
    cmap="plasma",
    zorder=2,)

m15_resid_ax.errorbar(
    m15_data["[Fe/H]"],
    feh_m15_resid,
    xerr=m15_data["e_[Fe/H]"],
    yerr=feh_offset_m15_std,
    fmt=".",
    zorder=1,)

m15_resid_ax.hlines(0, -1, 0.5, colors="k", linestyles="--")
m15_resid_ax.set_xlabel("[Fe/H] (Mann+15)")
m15_resid_ax.set_ylabel("[Fe/H] Resid")

fig_m15.tight_layout() 
fig_m15.savefig("paper/phot_feh_rel_m15.pdf")
fig_m15.savefig("paper/phot_feh_rel_m15.png", dpi=500)

# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------
break_row = 60

cpm_selected.sort_values("BP-RP_dr3", inplace=True)

cols = OrderedDict([
        ("Gaia DR2 ID (s)", ""),
        (r"$BP-RP$", ""),
        #(r"$a_4$", ""),
        ("Gaia DR2 ID (p)", ""),
        ("[Fe/H] ref", ""),
        ("[Fe/H] adopted", ""),
        #(r"$a_3$", ""),
        #(r"$a_4$", ""),
        
])
                        
header = []
header_1 = []
header_2 = []
table_rows = []
footer = []
notes = []

# Construct the header of the table
header.append("\\begin{table*}")
header.append("\\centering")


header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
header.append("\hline")
header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
header.append("\hline")

# Now add the separate info for the two tables
header_1 = header.copy()
caption = "Stellar pairs and primary [Fe/H] used for photometric [Fe/H] relation"
header_1.insert(2, "\\caption{{{}}}".format(caption))
header_1.insert(3, "\\label{tab:cpm_feh}")

header_2 = header.copy()
header_2.insert(2, "\\contcaption{{{}}}".format(caption))

# Populate the table for every science target
for star_i, star in cpm_info_feh_corr.iterrows():
    
    table_row = ""
    
    # Secondary source ID
    table_row += "{} &".format(star.name)

    # Bp-Rp
    table_row += r"{:0.2f} & ".format(star["BP-RP_dr3"])

    # Primary source ID
    table_row += "{} &".format(star["source_id_dr3_prim"])

    # [Fe/H] source
    table_row += r"\citealt{{{}}} &".format(star["citation"])

    # [Fe/H] source
    table_row += r"${:0.2f}\pm{:0.2f}$".format(star["feh_corr"], star["e_feh_corr"])
    #table_row += r"{:0.3f} & ".format(star["ldc_a2"])
    #table_row += r"{:0.3f} & ".format(star["ldc_a3"])
    #table_row += r"{:0.3f} ".format(star["ldc_a4"])

    table_rows.append(table_row + r"\\")

# Finish the table
footer.append("\hline")
footer.append("\end{tabular}")
footer.append("\\end{table*}")

# Write the table/s
break_rows = np.arange(break_row, len(cpm_info_feh_corr), break_row)
low_row = 0

for table_i, break_row in enumerate(break_rows):
    if table_i == 0:
        header = header_1
    else:
        header = header_2
    table_x = header_1 + table_rows[low_row:break_row] + footer + notes
    np.savetxt("paper/table_cpm_feh_dr3_{:0.0f}.tex".format(
        table_i), table_x, fmt="%s")
    low_row = break_row

# Do final part table
if low_row < len(cpm_info_feh_corr):
    table_i += 1
    table_x = header_2 + table_rows[low_row:] + footer + notes
    np.savetxt("paper/table_cpm_feh_dr3_{:0.0f}.tex".format(
        table_i), table_x, fmt="%s")