"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Load in magnitude offsets
mag_offsets = pd.read_csv("data/mag_offset_results.csv", index_col="Bp-Rp")

# Fitting function
def compute_resid(params, bp_rp, bp_rp_offset):
    """
    """
    bp_rp_offset_pred = params[0] * bp_rp + params[1]

    resid = bp_rp_offset - bp_rp_offset_pred

    return resid

def fit_linear_offset_model(bp_rp, bp_rp_offset):
    """
    """
    args = (bp_rp, bp_rp_offset)
    init_params = np.array([0.025, 0])
    opt_res = least_squares(
        compute_resid,
        init_params,
        jac="3-point",
        args=args,
    )

    return opt_res

def plot_fit(params, bp_rp, bp_rp_offset, filter, fmt_1, fmt_2):
    """
    """
    offset_label = "$\Delta {}$ integrated".format(filter)

    plt.plot(bp_rp, bp_rp_offset, fmt_1, label=offset_label, zorder=2)

    fit = params[0] * bp_rp + params[1]
    resid = bp_rp_offset - fit
    std = np.std(resid)

    fit_label = r"$\Delta {} = {:0.3f} (B_P-R_P) {:0.3f}, [\sigma \Delta {} = {:0.2f}]$".format(
        filter, params[0], params[1], filter, std)

    plt.plot(bp_rp, fit, fmt_2, label=fit_label, zorder=1)

bp_rp = mag_offsets.index

opt_res_bp = fit_linear_offset_model(bp_rp, mag_offsets["BP"])
opt_res_r = fit_linear_offset_model(bp_rp, mag_offsets["r"])
opt_res_g = fit_linear_offset_model(bp_rp, mag_offsets["g"])
#opt_res_v = fit_linear_offset_model(bp_rp, mag_offsets["v"])

plt.close("all")
plot_fit(opt_res_r["x"], bp_rp, mag_offsets["r"], "r", "x", "k--")
plot_fit(opt_res_bp["x"], bp_rp, mag_offsets["BP"], "B_P", ".", "k--")
plot_fit(opt_res_g["x"], bp_rp, mag_offsets["g"], "g", "+", "k--")
#plot_fit(opt_res_v["x"], bp_rp, mag_offsets["v"], "v")

plt.legend(fontsize="medium")
plt.ylim(-0.05,0.5)
plt.xlabel(r"$B_P-R_P$", fontsize="large",)
plt.ylabel(r"$\Delta$m$_\zeta$", fontsize="large",)
plt.tight_layout()

plt.savefig("paper/mag_offsets.pdf", bbox_innches="tight")
plt.savefig("paper/mag_offsets.png", bbox_innches="tight", dpi=500)

