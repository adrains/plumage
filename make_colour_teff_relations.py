"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Import photometry
m15_data = pd.read_csv(
    "data/mann15_all.tsv",
    sep="\t",
    dtype={"source_id":str},
    header=0)

# Remove any entries without gaia photometry
m15_data = m15_data[~np.isnan(m15_data["BPmag"])]

def calc_relation_teff(coeff, colours, j_h):
    """
    """
    terms = [
        coeff[0],
        coeff[1]*colours,
        coeff[2]*colours**2,
        coeff[3]*colours**3,
        coeff[4]*colours**3,
        coeff[5]*j_h,
        coeff[6]*j_h**2,]

    teffs_pred = np.sum(terms) * 3500

    return teffs_pred

def calc_resid_colour_jh(coeff, colours, j_h, teffs_real, e_teffs_real):
    """
    """
    teffs_pred = calc_relation_teff(coeff, colours, j_h)

    resid = (teffs_real - teffs_pred) / e_teffs_real

    return resid

def fit_colour_teff_relation(colours, j_hs, teffs_real, e_teffs_real):
    """
    """
    # Setup fit settings
    args = (colours, j_hs, teffs_real, e_teffs_real,)

    coeff_init = np.ones(7)

    # Do fit
    opt_res = least_squares(
        calc_resid_colour_jh, 
        coeff_init, 
        jac="3-point",
        #bounds=bounds[:,param_mask],
        #x_scale=scale[param_mask],
        #diff_step=step[param_mask],
        args=args, 
    )

    return opt_res["x"]

# Running
#colour = m15_data["BPmag"] - m15_data["Gmag"]  # +/- 51 K
colour = m15_data["BP-RP"]                     # +/- 49 K
#colour = m15_data["BPmag"] - m15_data["Jmag"]  # +/- 52 K
#colour = m15_data["BPmag"] - m15_data["Hmag"]  # +/- 53 K
#colour = m15_data["BPmag"] - m15_data["Ksmag"] # +/- 53 K

#colour = m15_data["RPmag"] - m15_data["Jmag"]  # +/- 64 K
#colour = m15_data["RPmag"] - m15_data["Hmag"]  # +/- 67 K
#colour = m15_data["RPmag"] - m15_data["Ksmag"] # +/- 65 K

#colour = m15_data["Gmag"] - m15_data["RPmag"]  # +/- 53 K
#colour = m15_data["Gmag"] - m15_data["Jmag"]   # +/- 58 K
#colour = m15_data["Gmag"] - m15_data["Hmag"]   # +/- 59 K
#colour = m15_data["Gmag"] - m15_data["Ksmag"]  # +/- 58 K

#colour = m15_data["BP-RP"] - m15_data["Ksmag"] # +/- 242 K
#colour = m15_data["Jmag"] - m15_data["Hmag"]   # +/- 245 K
#colour = m15_data["Jmag"] - m15_data["Ksmag"]  # +/- 154 K

j_h = m15_data["Jmag"] - m15_data["Hmag"]
#j_h = m15_data["[Fe/H]"]

coeffs = fit_colour_teff_relation(
    colour,
    j_h,
    m15_data["Teff"],
    m15_data["e_Teff"])

teffs_pred = calc_relation_teff(coeffs, colour, j_h,)

plt.close("all")
fig, (comp_ax, res_ax) = plt.subplots(2,1)

xx = np.linspace(np.min(m15_data["Teff"]), np.max(m15_data["Teff"]), 50)
comp_ax.plot(xx, xx, "--", color="black", zorder=0)
comp_ax.errorbar(teffs_pred, m15_data["Teff"], yerr=m15_data["e_Teff"], zorder=0, fmt=".")
sc1 = comp_ax.scatter(teffs_pred, m15_data["Teff"], c=m15_data["[Fe/H]"], zorder=1)
cb1 = fig.colorbar(sc1, ax=comp_ax)
cb1.set_label("[Fe/H]")

res_ax.plot(xx, np.zeros(50), "--", color="black")
resid = m15_data["Teff"] - teffs_pred

res_ax.errorbar(teffs_pred, resid, yerr=m15_data["e_Teff"], zorder=0, fmt=".")
sc2 = res_ax.scatter(teffs_pred, resid, c=m15_data["[Fe/H]"], zorder=1)
cb2 = fig.colorbar(sc2, ax=res_ax)
cb2.set_label("[Fe/H]")
res_ax.text(np.median(xx), 150, r"$\pm{:0.0f} K$".format(np.std(resid)))



