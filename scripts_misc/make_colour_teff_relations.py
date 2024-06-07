"""Script to fit Mann+15 photometric relations using Gaia DR3 data.
"""
import numpy as np
import matplotlib.pyplot as plt
import plumage.utils as putils
from scipy.optimize import least_squares
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def calc_relation_teff_feh(coeff, colours, fehs,):
    """Calculate a colour relation of the form:

    Teff/3500 = a + bX + cX^2 + dX^3 + eX^4 + f*Y

    Where X is the adopted colour and Y is [Fe/H].

    Parameters
    ----------
    coeff: 1D float array
        Array of coefficients for polynomial fit.

    colours: 1D float array
        Array of stellar colours for the adopted photometric colour.

    fehs: 1D float array
        Array of stellar [Fe/H] values.

    Return
    ------
    teffs_pred: 1D float array
        Array of predicted temperatures.
    """
    terms = [
        coeff[0],               # a
        coeff[1]*colours,       # b
        coeff[2]*colours**2,    # c
        coeff[3]*colours**3,    # d
        coeff[4]*colours**4,    # e
        coeff[5]*fehs,]         # f

    teffs_pred = np.sum(terms) * 3500

    return teffs_pred


def calc_relation_teff_j_h(coeff, colours, j_h,):
    """Calculate a colour relation of the form:

    Teff/3500 = a + bX + cX^2 + dX^3 + eX^4 + f*Y + g*Y^2

    Where X is the adopted colour and Y is the 2MASS (J-H) colour.

   Parameters
    ----------
    coeff: 1D float array
        Array of coefficients for polynomial fit.

    colours: 1D float array
        Array of stellar colours for the adopted photometric colour.

    j_h: 1D float array
        Array of 2MASS (J-H) colours.

    Return
    ------
    teffs_pred: 1D float array
        Array of predicted temperatures.
    """
    terms = [
        coeff[0],               # a
        coeff[1]*colours,       # b
        coeff[2]*colours**2,    # c
        coeff[3]*colours**3,    # d
        coeff[4]*colours**4,    # e
        coeff[5]*j_h,           # f
        coeff[6]*j_h**2,]       # g

    teffs_pred = np.sum(terms) * 3500

    return teffs_pred


def calc_resid_colour_jh(coeff, colours, j_h, teffs_real, e_teffs_real):
    """Calculate the resisuals for the polynomial colour-(J-H) relation.

    Parameters
    ----------
    coeff: 1D float array
        Array of coefficients for polynomial fit.

    colours: 1D float array
        Array of stellar colours for the adopted photometric colour.

    j_h: 1D float array
        Array of 2MASS (J-H) colours.

    teffs_real, e_teffs_real: 1D float array
        Measured Teff values and associated uncertainties to fit to.

    Return
    ------
    resid: 1D float array
        Array of uncertainty-weighted residuals.
    """
    teffs_pred = calc_relation_teff_j_h(coeff, colours, j_h)

    resid = (teffs_real - teffs_pred) / e_teffs_real

    return resid


def calc_resid_colour_feh(coeff, colours, fehs, teffs_real, e_teffs_real):
    """Calculate the resisuals for the polynomial colour-[Fe/H]] relation.
    
    Parameters
    ----------
    coeff: 1D float array
        Array of coefficients for polynomial fit.

    colours: 1D float array
        Array of stellar colours for the adopted photometric colour.

    fehs: 1D float array
        Array of stellar [Fe/H] values.

    teffs_real, e_teffs_real: 1D float array
        Measured Teff values and associated uncertainties to fit to.

    Return
    ------
    resid: 1D float array
        Array of uncertainty-weighted residuals.
    """
    teffs_pred = calc_relation_teff_feh(coeff, colours, fehs)

    resid = (teffs_real - teffs_pred) / e_teffs_real

    return resid


def fit_colour_teff_relation_jh(colours, j_hs, teffs_real, e_teffs_real):
    """Fit a colour relation of the form:

    Teff/3500 = a + bX + cX^2 + dX^3 + eX^4 + f*Y + g*Y^2

    Where X is the adopted colour and Y is the 2MASS (J-H) colour.

    Parameters
    ----------
    colours: 1D float array
        Array of stellar colours for the adopted photometric colour.

    j_hs: 1D float array
        Array of 2MASS (J-H) colours.

    teffs_real, e_teffs_real: 1D float array
        Measured Teff values and associated uncertainties to fit to.

    Return
    ------
    coeff: 1D float array
        Array of coefficients for polynomial fit.
    """
    # Setup fit settings
    args = (colours, j_hs, teffs_real, e_teffs_real,)

    coeff_init = np.ones(7)

    # Do fit
    opt_res = least_squares(
        calc_resid_colour_jh, 
        coeff_init, 
        jac="3-point",
        args=args, 
    )

    coeff = opt_res["x"]

    return coeff

def fit_colour_teff_relation_feh(colours, fehs, teffs_real, e_teffs_real):
    """Fit a colour relation of the form:

    Teff/3500 = a + bX + cX^2 + dX^3 + eX^4 + f*Y

    Where X is the adopted colour and Y is [Fe/H].

    Parameters
    ----------
    colours: 1D float array
        Array of stellar colours for the adopted photometric colour.

    fehs: 1D float array
        Array of stellar [Fe/H] values.

    teffs_real, e_teffs_real: 1D float array
        Measured Teff values and associated uncertainties to fit to.

    Return
    ------
    coeff: 1D float array
        Array of coefficients for polynomial fit.
    """
    # Setup fit settings
    args = (colours, fehs, teffs_real, e_teffs_real,)

    coeff_init = np.ones(6)

    # Do fit
    opt_res = least_squares(
        calc_resid_colour_feh, 
        coeff_init, 
        jac="3-point",
        args=args, 
    )

    coeff = opt_res["x"]

    return coeff

# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------
# File to load
mann_tsv = "data/mann15_all_dr3.tsv"

# Import Mann+15 standards
m15_data = putils.load_info_cat(
    mann_tsv,
    clean=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="dr3",)

# Remove any entries without bad RUWE
m15_data = m15_data[m15_data["ruwe_dr3"] < 1.4].copy()

# Which relation to use
relation = "colour_feh"
#relation = "colour_J-H"

# Running
colour = m15_data["BP-RP_dr3"]
j_h = m15_data["J_mag"] - m15_data["H_mag"]

# Fit colour with [Fe/H] relation
if relation == "colour_feh":
    coeffs = fit_colour_teff_relation_feh(
        colour,
        m15_data["[Fe/H]"],
        m15_data["Teff"],
        m15_data["e_Teff"])

    teffs_pred = calc_relation_teff_feh(coeffs, colour, m15_data["[Fe/H]"],)

# Fit colour with J-H relation
elif relation == "colour_J-H":
    coeffs = fit_colour_teff_relation_jh(
        colour,
        j_h,
        m15_data["Teff"],
        m15_data["e_Teff"])

    teffs_pred = calc_relation_teff_j_h(coeffs, colour, j_h,)

else:
    raise Exception("Unknown relation")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")
fig, comp_ax = plt.subplots(1)

# Plot upper panel
xx = np.linspace(np.min(m15_data["Teff"]), np.max(m15_data["Teff"]), 50)
comp_ax.plot(xx, xx, "--", color="black", zorder=0)

comp_ax.errorbar(
    m15_data["Teff"], teffs_pred, yerr=m15_data["e_Teff"], zorder=0, fmt=".")
sc1 = comp_ax.scatter(
    m15_data["Teff"], teffs_pred, c=m15_data["[Fe/H]"], zorder=1)

cb1 = fig.colorbar(sc1, ax=comp_ax)
cb1.set_label("[Fe/H]")

# Compute residuals
resid = m15_data["Teff"] - teffs_pred
resid_offset = np.median(resid)
resid_std = np.std(resid)

comp_ax.text(
    x=3400,
    y=2900,
    s=r"$\sigma_{{T_{{\rm eff}}}}={:0.0f}\pm{:0.0f} K$".format(
        resid_offset, resid_std),
    horizontalalignment="center",)

# Plot residuals
divider = make_axes_locatable(comp_ax)
resid_ax = divider.append_axes("bottom", size="30%", pad=0)
comp_ax.figure.add_axes(resid_ax, sharex=comp_ax)

resid_ax.plot(xx, np.zeros(50), "--", color="black")

resid_ax.errorbar(
    x=m15_data["Teff"],
    y=resid,
    xerr=m15_data["e_Teff"],
    yerr=np.full(resid.shape, resid_std),
    zorder=0,
    fmt=".",)

sc2 = resid_ax.scatter(
    m15_data["Teff"],
    resid,
    c=m15_data["[Fe/H]"],
    zorder=1,)

# Other formatting
comp_ax.set_ylabel(r"$T_{\rm eff}$ (K, Fit)")
resid_ax.set_xlabel(r"$T_{\rm eff}$ (K, Mann+15)")
resid_ax.set_ylabel(r"Residual (K)")
plt.setp(comp_ax.get_xticklabels(), visible=False)

resid_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=100))
resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))

resid_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=50))
resid_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=100))

comp_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=100))
comp_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=200))

if relation == "colour_feh":
    comp_ax.set_title(r"$(BP-RP)-$[Fe/H]")
    fig_fn = "paper/mann_colour_relation_fit_feh"
elif relation == "colour_J-H":
    comp_ax.set_title(r"$(BP-RP)-(J-H)$")
    fig_fn = "paper/mann_colour_relation_fit_j_h"

plt.show()
plt.tight_layout()

# Save plot
plt.savefig("{}.pdf".format(fig_fn))
plt.savefig("{}.png".format(fig_fn), dpi=300)


