"""Script to develop a M_K-(Bp-K) to [Fe/H] relation using literature CPM stars 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plumage.utils as utils
import plumage.parameters as params
from scipy.optimize import least_squares
import matplotlib.ticker as plticker
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
# Import settings
apply_ruwe_cut = True
use_only_aaa_2mass = True
rule_out_gaia_dups = True

# Which colour relation to use
c_rel = "Bp-K"
e_c_rel = "e_{}".format(c_rel)
c_label = "B_P-K_S"

# Whether to use only Mann+15 for the main sequence fit, or both M15 and CPM
only_use_m15_ms = True

# Polynnomial order when fitting main sequence
main_seq_poly_deg = 3

# Whether the offset from the MS is M_Ks per JA09, or offset per SL10
use_bp_k_offset = True

# Whether to do the [Fe/H] fit with just the CPM sample, or just the M15 sample
fit_feh_with_cpm = True

# Or both samples
fit_feh_with_both = False

# Polynomial order of the fitted offset function
offset_poly_order = 1

# Whether to apply an additional linear correction to the residuals
apply_correction = True

# Whether to plot remaining residual trend
plot_resid_trend = False

# -----------------------------------------------------------------------------
# Import and setup
# -----------------------------------------------------------------------------
# Import CPM info
cpm_info = utils.load_info_cat(
    "data/cpm_info.tsv",
    clean=False,
    allow_alt_plx=True,
    use_mann_code_for_masses=False) 

# Only keep reliable pairs
cpm_info = cpm_info[cpm_info["included"] == "yes"]

if use_only_aaa_2mass:
    cpm_info = cpm_info[cpm_info["Qflg"] == "AAA"]

if rule_out_gaia_dups:
    cpm_info = cpm_info[cpm_info["dup"] == 0]

if apply_ruwe_cut:
    cpm_info = cpm_info[cpm_info["ruwe"] < 1.4]

# Exclude those with parallaxes from the primary
#cpm_info = cpm_info[np.isnan(cpm_info["plx_alt"])]

# Only keep stars with 1.5 < Bp-Rp < 3.5
bp_rp_mask = np.logical_and(cpm_info["Bp-Rp"] > 1.5, cpm_info["Bp-Rp"] < 3.5)
cpm_info = cpm_info[bp_rp_mask]

# Import Mann+15 standards
m15_data = utils.load_info_cat(
    "data/mann15_all.tsv",
    clean=False,
    use_mann_code_for_masses=False)

# Remove any entries without gaia photometry or parallaxes
nan_mask = np.logical_and(~np.isnan(m15_data["Bp_mag"]), ~np.isnan(m15_data["plx"]))
m15_data = m15_data[nan_mask]

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
}

ignored = []

for source_id, cpm_row in cpm_info.iterrows():
    # More stars from Mann+13, check that first
    if ~np.isnan(cpm_row["feh_prim_m13"]):
        ref = cpm_row["ref_feh_prim_m13"]
        feh_corr = cpm_row["feh_prim_m13"] + feh_offsets[ref]
        e_feh_corr = cpm_row["e_feh_prim_m13"]

    # Then check those from Newton+14 that are from VF05
    elif cpm_row["feh_prim_ref_n14"] == "VF05":
        feh_corr = cpm_row["feh_prim_n14"] + feh_offsets["VF05"]
        e_feh_corr = cpm_row["e_feh_prim_n14"]
    
    # Then check other entries
    elif cpm_row["feh_prim_ref_other"] == "Ra07":
        feh_corr = cpm_row["feh_prim_other"] + feh_offsets["Ra07"]
        e_feh_corr = cpm_row["e_feh_prim_other"]

    # Don't know systematics, ignore
    else:
        feh_corr = np.nan
        e_feh_corr = np.nan
        ignored.append(source_id)
    
    feh_corr_all.append(feh_corr)
    e_feh_corr_all.append(e_feh_corr)

# All done
print("Ignored {} stars: {}".format(len(ignored), str(ignored)))
cpm_info["feh_corr"] = feh_corr_all
cpm_info["e_feh_corr"] = e_feh_corr_all

# Now mask out stars which we've excluded
cpm_info_feh_corr = cpm_info[~np.isnan(feh_corr_all)]

# -----------------------------------------------------------------------------
# Fit polynomial to mean MS
# -----------------------------------------------------------------------------
if only_use_m15_ms:
    bp_k = m15_data[c_rel]
    e_bp_k = m15_data[e_c_rel]
    k_mag_abs = m15_data["K_mag_abs"]
    e_k_mag_abs = m15_data["e_K_mag_abs"]

else:
    bp_k = np.concatenate((m15_data[c_rel], cpm_info_feh_corr[c_rel]))
    e_bp_k = np.concatenate((m15_data[e_c_rel], cpm_info_feh_corr[e_c_rel]))
    k_mag_abs = np.concatenate((m15_data["K_mag_abs"], cpm_info_feh_corr["K_mag_abs"]))
    e_k_mag_abs = np.concatenate((m15_data["e_K_mag_abs"], cpm_info_feh_corr["e_K_mag_abs"]))

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
    print("Bp-Ks offset per SL10")
    ms_coeff = np.polynomial.polynomial.polyfit(
        x=k_mag_abs,
        y=bp_k,
        deg=main_seq_poly_deg,
        w=1/e_bp_k,)

main_seq = np.polynomial.polynomial.Polynomial(ms_coeff)

# Now strip away the Mannn+15 targets who fall outside the Bp-Rp range
bp_rp_mask = np.logical_and(m15_data["Bp-Rp"] > 1.5, m15_data["Bp-Rp"] < 3.5)
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
        e_feh=cnp.concatenate((m15_data["e_[Fe/H]"], cpm_info_feh_corr["e_feh_corr"])),
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
np.savetxt("data/phot_feh_rel_ms_coeff.csv", ms_coeff)
np.savetxt("data/phot_feh_rel_offset_coeff.csv", offset_coeff)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")
fig, (cpm_ax, offset_fit_ax, cpm_feh_ax,) = plt.subplots(1,3)
fig2, (m15_ax, m15_feh_ax, m15_resid_ax) = plt.subplots(3,1)

# Assign axis names
#(cpm_ax, m15_ax, offset_fit_ax, 
#cpm_feh_ax, m15_feh_ax, empty_ax_1, 
#cpm_resid_ax, m15_resid_ax, empty_ax_2) = ax.flatten()
#(cpm_ax, offset_fit_ax, cpm_feh_ax,) = ax.flatten()

#empty_ax_1.set_visible(False)
#empty_ax_2.set_visible(False)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CPM Scatter
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = cpm_ax.scatter(
    cpm_info_feh_corr[c_rel], 
    cpm_info_feh_corr["K_mag_abs"], 
    c=cpm_info_feh_corr["feh_corr"],
    zorder=2,)

msk = ~np.isnan(cpm_info["plx_alt"])

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

cb = fig.colorbar(sc, ax=cpm_ax)
cb.set_label(r"[Fe/H]")
#cpm_ax.set_title("CPM Sample")
cpm_ax.set_xlabel(r"$({})$".format(c_label))
cpm_ax.set_ylabel(r"$M_{K_S}$")

cpm_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mann+15 Scatter
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = m15_ax.scatter(
    m15_data[c_rel], 
    m15_data["K_mag_abs"], 
    c=m15_data["[Fe/H]"],
    zorder=2,)
m15_ax.errorbar(
    m15_data[c_rel],
    m15_data["K_mag_abs"],
    xerr=m15_data[e_c_rel],
    yerr=m15_data["e_K_mag_abs"],
    fmt=".",
    zorder=1,)

cb = fig.colorbar(sc, ax=m15_ax)
cb.set_label(r"[Fe/H]")
m15_ax.set_title("Mann+15")
m15_ax.set_xlabel(r"$({})$".format(c_label))
#m15_ax.set_ylabel(r"$M_{K_S}$")

# Ensure axes have same limits
xlims = np.array([m15_ax.get_xlim(),cpm_ax.get_xlim()])
ylims = np.array([m15_ax.get_ylim(),cpm_ax.get_ylim()])

m15_ax.set_xlim(xlims[:,0].min(), xlims[:,1].max())
cpm_ax.set_xlim(xlims[:,0].min(), xlims[:,1].max())
m15_ax.set_ylim(ylims[:,1].max(), ylims[:,0].min())
cpm_ax.set_ylim(ylims[:,1].max(), ylims[:,0].min())

# Plot main sequence fit
# Use absolute K band offset from MS (per Johnson & Apps 2009)
if not use_bp_k_offset:
    xx = np.arange(xlims[:,0].min(), xlims[:,1].max(), 0.1)
    cpm_ax.plot(xx, main_seq(xx), "r--")
    m15_ax.plot(xx, main_seq(xx), "r--")

# Use (Bp-K) colour offset from MS (per Schlaufman & Laughlin 2010)
else:
    xx = np.arange(ylims[:,0].min(), ylims[:,1].max(), 0.1)
    cpm_ax.plot(main_seq(xx), xx, "r--")
    m15_ax.plot(main_seq(xx), xx, "r--")

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Bp-Ks fit
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
    c=cpm_info_feh_corr["Bp-Rp"],
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
offset_fit_ax.plot(xx, feh_poly(xx), "r--", label="LS fit")
offset_fit_ax.plot(xx, feh_poly_corr(xx), "-.", c="cornflowerblue", label="LS fit (corr)")
cb = fig.colorbar(sc, ax=offset_fit_ax)
cb.set_label(r"$(B_P-R_P)$")
offset_fit_ax.set_ylabel("[Fe/H]")
offset_fit_ax.legend(fontsize="small")

offset_fit_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CPM comp
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = cpm_feh_ax.scatter(
    cpm_info_feh_corr["feh_corr"],
    feh_pred_cpm,
    c=cpm_info_feh_corr["Bp-Rp"],
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
cb.set_label(r"$(B_P-R_P)$")
#cpm_feh_ax.set_xlabel("[Fe/H] (CPM)")
cpm_feh_ax.set_ylabel("[Fe/H] (fit)")

cpm_feh_ax.text(
    x=-0.6, 
    y=0.9, 
    s=r"$\Delta[Fe/H]={:0.2f}\pm {:0.2f}$".format(feh_offset_cpm_mean, feh_offset_cpm_std),
    horizontalalignment="center",
    fontsize="small")

# Residuals
divider = make_axes_locatable(cpm_feh_ax)
cpm_resid_ax = divider.append_axes("bottom", size="30%", pad=0)
cpm_feh_ax.figure.add_axes(cpm_resid_ax, sharex=cpm_feh_ax)
sc = cpm_resid_ax.scatter(
    cpm_info_feh_corr["feh_corr"],
    feh_cpm_resid,
    c=cpm_info_feh_corr["Bp-Rp"],
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
#cb = fig.colorbar(sc, ax=cpm_resid_ax)
#cb.set_label(r"$(B_P-R_P)$")
cpm_resid_ax.set_xlabel("[Fe/H] (Primary)")
cpm_resid_ax.set_ylabel("[Fe/H] Resid")

cpm_feh_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_feh_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
cpm_resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mann+15 comp
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = m15_feh_ax.scatter(
    m15_data["[Fe/H]"],
    feh_pred_m15,
    c=m15_data["Bp-Rp"],
    cmap="plasma",
    zorder=2)

m15_feh_ax.errorbar(
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
m15_feh_ax.plot(xx, xx, "k--")
cb = fig.colorbar(sc, ax=m15_feh_ax)
cb.set_label(r"$(B_P-R_P)$")
#m15_feh_ax.set_xlabel("[Fe/H] (Mann+15)")
m15_feh_ax.set_ylabel("[Fe/H] (fit)")

m15_feh_ax.text(
    x=-0.25, 
    y=-1.0, 
    s=r"$\Delta[Fe/H]={:0.2f}\pm {:0.2f}$".format(feh_offset_m15_mean, feh_offset_m15_std),
    horizontalalignment="center")

# Resid
sc = m15_resid_ax.scatter(
    m15_data["[Fe/H]"],
    feh_m15_resid,
    c=m15_data["Bp-Rp"],
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
cb = fig.colorbar(sc, ax=m15_resid_ax)
cb.set_label(r"$(B_P-R_P)$")
m15_resid_ax.set_xlabel("[Fe/H] (Mann+15)")
m15_resid_ax.set_ylabel("[Fe/H] Resid")

# Wrap up
fig.set_size_inches(9, 2.5)
fig.tight_layout() 
fig.savefig("paper/phot_feh_rel.pdf")
fig.savefig("paper/phot_feh_rel.png")

# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------
break_row = 60

cpm_info.sort_values("Bp-Rp", inplace=True)

cols = OrderedDict([
        ("Gaia DR2 ID (s)", ""),
        (r"$B_P-R_P$", ""),
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
header.append("\\label{tab:cpm_feh}")

header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
header.append("\hline")
header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
header.append("\hline")

# Now add the separate info for the two tables
header_1 = header.copy()
caption = ("Stellar pairs and primary [Fe/H] used for photometric [Fe/H] relation")
header_1.insert(3, "\\caption{{{}}}".format(caption))

#header_2 = header.copy()
#header_2.insert(3, "\\contcaption{Limb darkening coefficients}")

# Populate the table for every science target
for star_i, star in cpm_info_feh_corr.iterrows():
    
    table_row = ""
    
    # Secondary source ID
    table_row += "{} &".format(star.name)

    # Bp-Rp
    table_row += r"{:0.2f} & ".format(star["Bp-Rp"])

    # Secondary source ID
    table_row += "{} &".format(star["HIP"])

    # [Fe/H] source
    table_row += "{} &".format(star["ref_feh_prim_m13"])

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
    table_x = header_1 + table_rows[low_row:break_row] + footer + notes
    np.savetxt("paper/table_cpm_feh_{:0.0f}.tex".format(
        table_i), table_x, fmt="%s")
    low_row = break_row

# Do final part table
if low_row < len(cpm_info_feh_corr):
    table_i += 1
    table_x = header_1 + table_rows[low_row:] + footer + notes
    np.savetxt("paper/table_cpm_feh_{:0.0f}.tex".format(
        table_i), table_x, fmt="%s")