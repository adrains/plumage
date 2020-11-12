"""Script to develop a M_K-(Bp-K) to [Fe/H] relation using literature CPM stars 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plumage.utils as utils
from scipy.optimize import least_squares

# Import CPM info
cpm_info = utils.load_info_cat("data/cpm_info.tsv", clean=False, allow_alt_plx=True) 

# Only keep reliable pairs
cpm_info = cpm_info[cpm_info["included"] == "yes"]

# Exclude those with parallaxes from the primary
cpm_info = cpm_info[np.isnan(cpm_info["plx_alt"])]

# Only keep stars with 1.5 < Bp-Rp < 3.5
bp_rp_mask = np.logical_and(cpm_info["Bp-Rp"] > 1.5, cpm_info["Bp-Rp"] < 3.5)
cpm_info = cpm_info[bp_rp_mask]

# Import Mann+15 standards
m15_data = utils.load_info_cat("data/mann15_all.tsv", clean=False)

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

c_rel = "Bp-K"
e_c_rel = "e_{}".format(c_rel)

# -----------------------------------------------------------------------------
# Fit polynomial to mean MS
# -----------------------------------------------------------------------------
only_use_m15_ms = True

if only_use_m15_ms:
    bp_k = m15_data[c_rel]
    k_mag_abs = m15_data["K_mag_abs"]
    e_k_mag_abs = m15_data["e_K_mag_abs"]

else:
    bp_k = np.concatenate((m15_data[c_rel], cpm_info_feh_corr[c_rel]))
    k_mag_abs = np.concatenate((m15_data["K_mag_abs"], cpm_info_feh_corr["K_mag_abs"]))
    e_k_mag_abs = np.concatenate((m15_data["e_K_mag_abs"], cpm_info_feh_corr["e_K_mag_abs"]))

coeff = np.polynomial.polynomial.polyfit(
    x=bp_k,
    y=k_mag_abs,
    deg=3,
    w=1/e_k_mag_abs,)

main_seq = np.polynomial.polynomial.Polynomial(coeff)

# -----------------------------------------------------------------------------
# Fit [Fe/H] offset
# -----------------------------------------------------------------------------
def calc_feh(params, bp_k, k_mag_abs, feh, main_seq,):
    """
    """
    delta_mk = k_mag_abs - main_seq(bp_k)

    feh_poly = np.polynomial.polynomial.Polynomial(params)
    feh_fit = feh_poly(delta_mk)

    return feh_fit

def calc_resid(params, bp_k, k_mag_abs, feh, e_feh, main_seq,):
    """
    """
    feh_fit = calc_feh(params, bp_k, k_mag_abs, feh, main_seq,)

    resid = (feh - feh_fit) / e_feh

    return resid

def fit_feh_model(bp_k, k_mag_abs, feh, e_feh, main_seq, poly_order):
    """
    """
    args = (bp_k, k_mag_abs, feh, e_feh, main_seq,)
    init_params = np.ones(poly_order+1)
    opt_res = least_squares(
        calc_resid,
        init_params,
        jac="3-point",
        args=args,
    )

    return opt_res

# Fit params
fit_feh_with_cpm = True
poly_order = 2

if fit_feh_with_cpm:
    opt_res = fit_feh_model(
        cpm_info_feh_corr[c_rel],
        cpm_info_feh_corr["K_mag_abs"],
        cpm_info_feh_corr["feh_corr"],
        cpm_info_feh_corr["e_feh_corr"],
        main_seq,
        poly_order)

else:
    opt_res = fit_feh_model(
        m15_data[c_rel],
        m15_data["K_mag_abs"],
        m15_data["[Fe/H]"],
        m15_data["e_[Fe/H]"],
        main_seq,
        poly_order,)

params = opt_res["x"]

# Compute [Fe/H] from relation for CPM and M15 sample
feh_pred_cpm = calc_feh(
    params,
    cpm_info_feh_corr[c_rel],
    cpm_info_feh_corr["K_mag_abs"],
    cpm_info_feh_corr["feh_corr"],
    main_seq,)

feh_pred_m15 = calc_feh(
    params,
    m15_data[c_rel],
    m15_data["K_mag_abs"],
    m15_data["[Fe/H]"],
    main_seq,)

# Calculate mean and std
feh_offset_cpm_mean = np.mean(cpm_info_feh_corr["feh_corr"] - feh_pred_cpm)
feh_offset_cpm_std = np.std(cpm_info_feh_corr["feh_corr"] - feh_pred_cpm)

feh_offset_m15_mean = np.mean(m15_data["[Fe/H]"] - feh_pred_m15)
feh_offset_m15_std = np.std(m15_data["e_[Fe/H]"] - feh_pred_m15)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")
fig, ax = plt.subplots(2,2)
(cpm_ax, m15_ax, cpm_feh_ax, m15_feh_ax) = ax.flatten()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CPM Scatter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
cpm_ax.set_xlabel(r"$(B_P-K_S)$")
cpm_ax.set_ylabel(r"$M_{K_S}$")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mann+15 Scatter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
m15_ax.set_xlabel(r"$(B_P-K_S)$")
#m15_ax.set_ylabel(r"$M_{K_S}$")

# Ensure axes have same limits
xlims = np.array([m15_ax.get_xlim(),cpm_ax.get_xlim()])
ylims = np.array([m15_ax.get_ylim(),cpm_ax.get_ylim()])

m15_ax.set_xlim(xlims[:,0].min(), xlims[:,1].max())
cpm_ax.set_xlim(xlims[:,0].min(), xlims[:,1].max())
m15_ax.set_ylim(ylims[:,1].max(), ylims[:,0].min())
cpm_ax.set_ylim(ylims[:,1].max(), ylims[:,0].min())

xx = np.arange(xlims[:,0].min(), xlims[:,1].max(), 0.1)
cpm_ax.plot(xx, main_seq(xx), "r--")
m15_ax.plot(xx, main_seq(xx), "r--")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CPM comp
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = cpm_feh_ax.scatter(
    feh_pred_cpm,
    cpm_info_feh_corr["feh_corr"],
    c=cpm_info_feh_corr["Bp-Rp"],
    cmap="plasma",
    zorder=2,)

cpm_feh_ax.errorbar(
    feh_pred_cpm,
    cpm_info_feh_corr["feh_corr"],
    #xerr=m15_data[e_c_rel],
    yerr=cpm_info_feh_corr["e_feh_corr"],
    fmt=".",
    zorder=1,)

xx = np.arange(-1,0.5,0.1)
cpm_feh_ax.plot(xx,xx,"k--")
cb = fig.colorbar(sc, ax=cpm_feh_ax)
cb.set_label(r"$B_P-R_P$")
cpm_feh_ax.set_xlabel("[Fe/H] (fit)")
cpm_feh_ax.set_ylabel("[Fe/H] (literature)")

cpm_feh_ax.text(
    x=-0.5, 
    y=0.4, 
    s=r"${:0.2f}\pm {:0.2f}$".format(feh_offset_cpm_mean, feh_offset_cpm_std),
    horizontalalignment="center")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mann+15 comp
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sc = m15_feh_ax.scatter(
    feh_pred_m15,
    m15_data["[Fe/H]"],
    c=m15_data["Bp-Rp"],
    cmap="plasma",
    zorder=2)

m15_feh_ax.errorbar(
    feh_pred_m15,
    m15_data["[Fe/H]"],
    #xerr=m15_data[e_c_rel],
    yerr=m15_data["e_[Fe/H]"],
    fmt=".",
    zorder=1,)

xx = np.arange(-1,0.5,0.1)
m15_feh_ax.plot(xx, xx, "k--")
cb = fig.colorbar(sc, ax=m15_feh_ax)
cb.set_label(r"$B_P-R_P$")
m15_feh_ax.set_xlabel("[Fe/H] (fit)")
m15_feh_ax.set_ylabel("[Fe/H] (literature)")

m15_feh_ax.text(
    x=-0.5, 
    y=0.4, 
    s=r"${:0.2f}\pm {:0.2f}$".format(feh_offset_m15_mean, feh_offset_m15_std),
    horizontalalignment="center")