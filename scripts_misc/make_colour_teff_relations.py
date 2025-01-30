"""Script to fit Mann+15 photometric relations using Gaia DR3 data.
"""
import numpy as np
import matplotlib.pyplot as plt
import plumage.utils as pu
from scipy.optimize import least_squares
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# If true, we include the Kesseli+2019 sample of subdwarfs. These are gridded
# in 100 K intervals, however the impact of this is mitigated by them having
# larger Teff uncertainties (+/-100 K) than the Mann+2015 sample. For the
# (BP-RP)-[Fe/H] relation this does not decrease performance for stars with
# [Fe/H] > -0.5, but does reduce systematics and scatter for Teff recovery with
# the subdwarf sample.
include_K19_subdwarfs = True

# Whether to make a RUWE cut. Testing with a (BP-RP)-[Fe/H] relation indicates
# that the RUWE cut results in *worse* performance, so this is not recommended.
make_ruwe_cut = False
ruwe_threshold = 1.4

# Which relation to use (either 'colour_feh' or 'colour_J-H')
relation = "colour_feh"

# Inflation factor for K+19 uncertainties
K19x = 1.5

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
    c_unity = np.ones_like(colours)

    terms = [
        coeff[0]*c_unity,       # a
        coeff[1]*colours,       # b
        coeff[2]*colours**2,    # c
        coeff[3]*colours**3,    # d
        coeff[4]*colours**4,    # e
        coeff[5]*fehs,]         # f

    teffs_pred = np.sum(terms, axis=0) * 3500

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
    c_unity = np.ones_like(colours)

    terms = [
        coeff[0]*c_unity,       # a
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
# Import
# -----------------------------------------------------------------------------
# ---------------------------------------------
# Mann+2015
# ---------------------------------------------
mann_tsv = "data/mann15_all_dr3.tsv"

m15_data = pu.load_info_cat(
    mann_tsv,
    clean=False,
    use_mann_code_for_masses=False,
    do_extinction_correction=False,
    do_skymapper_crossmatch=False,
    gdr="dr3",)

# ---------------------------------------------
# Kiman+2019
# ---------------------------------------------
if include_K19_subdwarfs:
    k19_tsv = "data/K19_all.tsv"

    k19_data = pu.load_info_cat(
        k19_tsv,
        clean=False,
        use_mann_code_for_masses=False,
        do_extinction_correction=False,
        do_skymapper_crossmatch=False,
        gdr="dr3",)

    # ---------------------------------------------
    # Merge
    # ---------------------------------------------
    data_tab = m15_data.join(
        k19_data, "source_id_dr3", rsuffix="_k19", how="outer").copy()
    data_tab.set_index("source_id_dr3", inplace=True)

    # Drop nan rows
    keep = [type(aa) == str for aa in data_tab.index.values]
    data_tab = data_tab[keep].copy()

    has_m15 = ~np.isnan(data_tab["[Fe/H]"].values)
    has_k19 = ~np.isnan(data_tab["feh"].values)

    data_tab["has_m15"] = has_m15
    data_tab["has_k19"] = has_k19

    # Drop nan rows
    keep = [type(aa) == str for aa in data_tab.index.values]
    data_tab = data_tab[keep].copy()
    
    # ---------------------------------------------
    # Merge Gaia data
    # ---------------------------------------------
    data_tab.loc[has_k19, "BP-RP_dr3"] = data_tab.loc[has_k19, "BP-RP_dr3_k19"]
    data_tab.loc[has_k19, "ruwe_dr3"] = data_tab.loc[has_k19, "ruwe_dr3_k19"]

    # Remove any entries without bad RUWE
    if make_ruwe_cut:
        data_tab = data_tab[data_tab["ruwe_dr3"] < 1.4].copy()

    n_star = len(data_tab)
    has_m15 = data_tab["has_m15"]
    has_k19 = data_tab["has_k19"]

    # ---------------------------------------------
    # Select Params
    # ---------------------------------------------
    adopt_k19 = np.logical_and(~has_m15, has_k19)

    # [Fe/H]
    feh_adopt = np.full(n_star, np.nan)
    feh_adopt[has_m15] = data_tab["[Fe/H]"].values[has_m15]
    feh_adopt[adopt_k19] = data_tab["feh"].values[adopt_k19]

    # Teff
    teff_adopt = np.full(n_star, np.nan)
    e_teff_adopt = np.full(n_star, np.nan)

    teff_adopt[has_m15] = data_tab["Teff"].values[has_m15]
    e_teff_adopt[has_m15] = data_tab["e_Teff"].values[has_m15]

    teff_adopt[adopt_k19] = data_tab["teff"].values[adopt_k19]
    e_teff_adopt[adopt_k19] = data_tab["e_teff"].values[adopt_k19] * K19x

    data_tab["feh_adopt"] = feh_adopt
    data_tab["teff_adopt"] = teff_adopt
    data_tab["e_teff_adopt"] = e_teff_adopt

else:
    data_tab = m15_data
    data_tab.rename(
        columns={"[Fe/H]":"feh_adopt", 
                 "Teff":"teff_adopt",
                 "e_Teff":"e_teff_adopt"},
        inplace=True)

    # Remove any entries without bad RUWE
    if make_ruwe_cut:
        data_tab = data_tab[data_tab["ruwe_dr3"] < 1.4].copy()

# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------
# Running
colour = data_tab["BP-RP_dr3"].values
j_h = data_tab["J_mag"] - data_tab["H_mag"].values

# Fit colour with [Fe/H] relation
if relation == "colour_feh":
    coeffs = fit_colour_teff_relation_feh(
        colour,
        data_tab["feh_adopt"].values,
        data_tab["teff_adopt"].values,
        data_tab["e_teff_adopt"].values)

    teffs_pred = calc_relation_teff_feh(coeffs, colour, data_tab["feh_adopt"],)

# Fit colour with J-H relation
elif relation == "colour_J-H":
    coeffs = fit_colour_teff_relation_jh(
        colour,
        j_h,
        data_tab["teff_adopt"].values,
        data_tab["e_teff_adopt"].values)

    teffs_pred = calc_relation_teff_j_h(coeffs, colour, j_h,)

else:
    raise Exception("Unknown relation")

# Round coefficients
coeffs_orig = coeffs.copy()
coeffs = np.round(coeffs, 4)

print("Fitted Coefficients:")
print("\t".join(coeffs.astype(str)))

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")
fig, comp_ax = plt.subplots(1)

# ---------------------------------------------
# Combined (or just Mann+15) sample
# ---------------------------------------------
xx = np.linspace(
    np.min(data_tab["teff_adopt"]), np.max(data_tab["teff_adopt"]), 50)
comp_ax.plot(xx, xx, "--", color="black", zorder=0)

comp_ax.errorbar(
    data_tab["teff_adopt"],
    teffs_pred,
    yerr=data_tab["e_teff_adopt"],
    zorder=0,
    fmt=".")

sc1 = comp_ax.scatter(
    data_tab["teff_adopt"],
    teffs_pred,
    c=data_tab["feh_adopt"],
    zorder=1,
    label="K/M Benchmark ({} stars)".format(len(data_tab)))

# ---------------------------------------------
# Kiman+2019 highlighting + extra annotations
# ---------------------------------------------
if include_K19_subdwarfs:
    label = "Kesseli+2019 ({} stars)".format(int(np.sum(adopt_k19)))
        
    scatter = comp_ax.scatter(
        data_tab["teff_adopt"][adopt_k19],
        teffs_pred[adopt_k19],
        marker="o",
        c=data_tab["feh_adopt"][adopt_k19],
        #facecolors='none',
        edgecolor="k",
        linewidths=1.2,
        zorder=1,
        label=label,)
    
    plt.legend()

    # Performance
    teffs_pred_m15 = calc_relation_teff_feh(
        coeffs,
        m15_data["BP-RP_dr3"].values,
        m15_data["[Fe/H]"].values,)

    teffs_pred_k19 = calc_relation_teff_feh(
        coeffs,
        data_tab["BP-RP_dr3"].values[adopt_k19],
        data_tab["feh_adopt"].values[adopt_k19],)
    
    resid_m15 = m15_data["Teff"].values - teffs_pred_m15
    delta_m15 = np.nanmedian(resid_m15)
    sigma_m15 = np.nanstd(resid_m15)

    comp_ax.text(
        x=3200,
        y=2750,
        s=r"$\sigma_{{T_{{\rm eff}}}}={:+3.0f}\pm{:0.0f}\,$K (M15)".format(
            delta_m15, sigma_m15),
        horizontalalignment="left",)

    resid_k19 = data_tab["teff_adopt"].values[adopt_k19] - teffs_pred_k19
    delta_k19 = np.nanmedian(resid_k19)
    sigma_k19 = np.nanstd(resid_k19)

    comp_ax.text(
        x=3200,
        y=2575,
        s=r"$\sigma_{{T_{{\rm eff}}}}={:+3.0f}\pm{:0.0f}\,$K (K19)".format(
            delta_k19, sigma_k19),
        horizontalalignment="left",)

cb1 = fig.colorbar(sc1, ax=comp_ax)
cb1.set_label("[Fe/H]")

# ---------------------------------------------
# Residuals axis
# ---------------------------------------------
resid = data_tab["teff_adopt"] - teffs_pred
resid_offset = np.median(resid)
resid_std = np.std(resid)

comp_ax.text(
    x=3200,
    y=2925,
    s=r"$\sigma_{{T_{{\rm eff}}}}={:+3.0f}\pm{:0.0f}\,$K (All)".format(
        resid_offset, resid_std),
    horizontalalignment="left",)

# Plot residuals
divider = make_axes_locatable(comp_ax)
resid_ax = divider.append_axes("bottom", size="30%", pad=0)
comp_ax.figure.add_axes(resid_ax, sharex=comp_ax)

resid_ax.plot(xx, np.zeros(50), "--", color="black")

resid_ax.errorbar(
    x=data_tab["teff_adopt"],
    y=resid,
    xerr=data_tab["e_teff_adopt"],
    yerr=np.full(resid.shape, resid_std),
    zorder=0,
    fmt=".",)

sc2 = resid_ax.scatter(
    data_tab["teff_adopt"],
    resid,
    c=data_tab["feh_adopt"],
    zorder=1,)

if include_K19_subdwarfs:
    label = "K+19 ({})".format(int(np.sum(adopt_k19)))
        
    scatter = resid_ax.scatter(
        data_tab["teff_adopt"][adopt_k19],
        resid[adopt_k19],
        marker="o",
        c=data_tab["feh_adopt"][adopt_k19],
        edgecolor="k",
        linewidths=1.2,
        zorder=1,
        label=label,)

# Other formatting
comp_ax.set_ylabel(r"$T_{\rm eff}$ (K, Fit)")
resid_ax.set_xlabel(r"$T_{\rm eff}$ (K, Literature)")
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