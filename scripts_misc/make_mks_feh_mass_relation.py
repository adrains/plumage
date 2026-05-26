"""Script to fit MKs-[Fe/H]-R_star relations using Mann+15 and Kesseli+2019.
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

# Whether to force the use of coefficients from M15 or K19
force_M15_coeff = False
force_K19_coeff = False

# Whether to make a RUWE cut. Testing with a (BP-RP)-[Fe/H] relation indicates
# that the RUWE cut results in *worse* performance, so this is not recommended.
make_ruwe_cut = False
ruwe_threshold = 1.4

# Inflation factor for K+19 uncertainties
K19x = 1.0

N_COEFF = 4

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def calc_relation_MKs_Fe_H(coeff, MKs, Fe_H,):
    """Calculate a colour relation of the form:

    R_star = (a + b * Mks + c * MKs^2) * (1 + f * [Fe/H])

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
    rstar_pred = (
        coeff[0] + coeff[1]*MKs + coeff[2]*MKs**2) * (1 + coeff[3]*Fe_H)
    
    return rstar_pred


def calc_resid_MKs_Fe_H_rstar(coeff, MKs, Fe_H, rstar_real, e_rstar_real):
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
    rstar_pred = calc_relation_MKs_Fe_H(coeff, MKs, Fe_H)

    resid = (rstar_real - rstar_pred) / e_rstar_real

    return resid**2



def fit_MKs_Fe_H_radius_relation(MKs, Fe_H, rstar_real, e_rstar_real):
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
    args = (MKs, Fe_H, rstar_real, e_rstar_real,)

    coeff_init = np.ones(N_COEFF)

    # Do fit
    opt_res = least_squares(
        calc_resid_MKs_Fe_H_rstar, 
        coeff_init, 
        jac="3-point",
        args=args, 
    )

    coeff = opt_res["x"]

    return coeff, opt_res

# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
# ---------------------------------------------
# Mann+2015
# ---------------------------------------------
mann_tsv = "data/mann15_all_dr3.tsv"

m15_data = pu.load_info_cat(
    mann_tsv,
    make_observed_col_bool_on_yes=False,
    use_mann_code_for_masses=False,
    gdr="dr3",)

# ---------------------------------------------
# Kiman+2019
# ---------------------------------------------
if include_K19_subdwarfs:
    k19_tsv = "data/K19_all.tsv"

    k19_data = pu.load_info_cat(
        k19_tsv,
        make_observed_col_bool_on_yes=False,
        use_mann_code_for_masses=False,
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
    data_tab.loc[has_k19, "plx_dr3"] = data_tab.loc[has_k19, "plx_dr3_k19"]

    # Remove any entries without bad RUWE
    if make_ruwe_cut:
        data_tab = data_tab[data_tab["ruwe_dr3"] < 1.4].copy()

    n_star = len(data_tab)
    has_m15 = data_tab["has_m15"].values
    has_k19 = data_tab["has_k19"].values

    only_has_k19 = np.logical_and(has_k19, ~has_m15)

    # ---------------------------------------------
    # Merge 2MASS data
    # ---------------------------------------------
    data_tab.loc[has_m15, "Qflg_2MASS"] = "AAA"
    data_tab.loc[only_has_k19, "K_mag"] = data_tab.loc[only_has_k19, "K_mag_k19"]
    data_tab.loc[only_has_k19, "e_K_mag"] = data_tab.loc[only_has_k19, "e_K_mag_k19"]
    data_tab.loc[only_has_k19, "Qflg_2MASS"] = data_tab.loc[only_has_k19, "Qflg"]

    data_tab.loc[only_has_k19, "plx_dr3"] = data_tab.loc[has_k19, "plx_dr3_k19"]

    # Remove any entries with bad photometry
    if make_ruwe_cut:
        data_tab = data_tab[data_tab["Qflg_2MASS"] != "AAA"].copy()
    
    # Compute Absolute magnitudes
    dist = 1000 / data_tab["plx_dr3"].values
    data_tab["MKs"] = data_tab["K_mag"] - 5*np.log10(dist/10)

    # ---------------------------------------------
    # Select Params
    # ---------------------------------------------
    adopt_k19 = np.logical_and(~has_m15, has_k19)

    # [Fe/H]
    feh_adopt = np.full(n_star, np.nan)
    feh_adopt[has_m15] = data_tab["[Fe/H]"].values[has_m15]
    feh_adopt[adopt_k19] = data_tab["feh"].values[adopt_k19]

    # Radius
    rstar_adopt = np.full(n_star, np.nan)
    e_rstar_adopt = np.full(n_star, np.nan)

    rstar_adopt[has_m15] = data_tab["R"].values[has_m15]
    e_rstar_adopt[has_m15] = data_tab["e_R"].values[has_m15]

    rstar_adopt[adopt_k19] = data_tab["r_star"].values[adopt_k19]
    e_rstar_adopt[adopt_k19] = data_tab["e_r_star"].values[adopt_k19] * K19x

    data_tab["feh_adopt"] = feh_adopt
    data_tab["rstar_adopt"] = rstar_adopt
    data_tab["e_rstar_adopt"] = e_rstar_adopt

    # Mask out those stars without 2MASS
    data_tab = data_tab[~np.isnan(data_tab["MKs"])].copy()

    # RUWE cut
    data_tab = data_tab[data_tab["ruwe_dr3"].values <= 1.4].copy()

    # Update masks
    has_m15 = data_tab["has_m15"].values
    has_k19 = data_tab["has_k19"].values
    adopt_k19 = np.logical_and(~has_m15, has_k19)

else:
    data_tab = m15_data
    data_tab.rename(
        columns={"[Fe/H]":"feh_adopt", 
                 "R":"rstar_adopt",
                 "e_R":"e_rstar_adopt"},
        inplace=True)

    dist = 1000 / data_tab["plx_dr3"].values
    data_tab["MKs"] = data_tab["K_mag"] - 5*np.log10(dist/10)

    # RUWE cut
    data_tab = data_tab[data_tab["ruwe_dr3"].values <= 1.4].copy()

# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------
# Running
MKs = data_tab["MKs"].values

# Fit M_Ks with [Fe/H] relation
coeffs, opt_res = fit_MKs_Fe_H_radius_relation(
    data_tab["MKs"].values,
    data_tab["feh_adopt"].values,
    data_tab["rstar_adopt"].values,
    data_tab["e_rstar_adopt"].values)

# Force old coefficients for checking
if force_M15_coeff:
    coeffs = [1.930, -0.3466, 0.01647, -0.04458]
elif force_K19_coeff:
    coeffs = [1.875, -0.337, 0.0161, 0.079]

rstar_pred = calc_relation_MKs_Fe_H(coeffs, MKs, data_tab["feh_adopt"],)

# Round coefficients
coeffs_orig = coeffs.copy()
coeffs = np.round(coeffs, 4)

print("Fitted Coefficients:")
print("\t".join(coeffs.astype(str)))

# Number of degrees of freedom
ndf = len(MKs) - N_COEFF
rchi2 = opt_res["cost"] / ndf

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plt.close("all")
fig, comp_ax = plt.subplots(1, sharex=True)

# ---------------------------------------------
# Combined (or just Mann+15) sample
# ---------------------------------------------
xx = np.arange(0, 0.75, 0.01)
comp_ax.plot(xx, xx, "--", color="black", zorder=0)

comp_ax.errorbar(
    data_tab["rstar_adopt"],
    rstar_pred,
    xerr=data_tab["e_rstar_adopt"],
    zorder=0,
    elinewidth=0.5,
    fmt=".")

sc1 = comp_ax.scatter(
    data_tab["rstar_adopt"],
    rstar_pred,
    c=data_tab["feh_adopt"],
    zorder=1,
    label="K/M Benchmark ({} stars)".format(len(data_tab)))

# ---------------------------------------------
# Kesseli+2019 highlighting + extra annotations
# ---------------------------------------------
if include_K19_subdwarfs:
    label = "Kesseli+2019 ({} stars)".format(int(np.sum(adopt_k19)))
        
    scatter = comp_ax.scatter(
        data_tab["rstar_adopt"][adopt_k19],
        rstar_pred[adopt_k19],
        marker="o",
        c=data_tab["feh_adopt"][adopt_k19],
        #facecolors='none',
        edgecolor="k",
        linewidths=1.2,
        zorder=1,
        label=label,)
    
    plt.legend()

    # Performance
    rstar_pred_m15 = calc_relation_MKs_Fe_H(
        coeffs,
        data_tab["MKs"].values,
        data_tab["[Fe/H]"].values,)

    rstar_pred_k19 = calc_relation_MKs_Fe_H(
        coeffs,
        data_tab["MKs"].values[adopt_k19],
        data_tab["feh_adopt"].values[adopt_k19],)
    
    resid_m15 = data_tab["R"].values - rstar_pred_m15
    delta_m15 = np.nanmedian(resid_m15)
    sigma_m15 = np.nanstd(resid_m15)

    delta_pc_m15 = np.nanmedian(resid_m15 / data_tab["R"].values * 100)
    sigma_pc_m15 = np.nanstd(resid_m15 / data_tab["R"].values * 100)

    comp_ax.text(
        x=0.35,
        y=0.2,
        s=r"$\sigma_{{R_\star}}={:+3.2f}\pm{:0.2f}\,\%$ (M15)".format(
            delta_pc_m15, sigma_pc_m15),
        horizontalalignment="left",)

    resid_k19 = data_tab["r_star"].values[adopt_k19] - rstar_pred_k19
    delta_k19 = np.nanmedian(resid_k19)
    sigma_k19 = np.nanstd(resid_k19)

    delta_pc_k19 = np.nanmedian(resid_k19 / data_tab["r_star"].values[adopt_k19] * 100)
    sigma_pc_k19 = np.nanstd(resid_k19 / data_tab["r_star"].values[adopt_k19] * 100)

    comp_ax.text(
        x=0.35,
        y=0.15,
        s=r"$\sigma_{{R_\star}}={:+3.2f}\pm{:0.2f}\,\%$ (K19)".format(
            delta_pc_k19, sigma_pc_k19),
        horizontalalignment="left",)

cb1 = fig.colorbar(sc1, ax=comp_ax)
cb1.set_label("[Fe/H]")

# ---------------------------------------------
# Residuals axis
# ---------------------------------------------
resid = data_tab["rstar_adopt"] - rstar_pred
resid_offset = np.median(resid)
resid_std = np.std(resid)

delta_pc = np.nanmedian(resid / data_tab["rstar_adopt"].values * 100)
sigma_pc = np.nanstd(resid / data_tab["rstar_adopt"].values * 100)

e_resid = np.sqrt(
    np.full(resid.shape, resid_std)**2 + data_tab["e_rstar_adopt"].values**2)

comp_ax.text(
    x=0.35,
    y=0.1,
    s=r"$\sigma_{{R_\star}}={:+3.2f}\pm{:0.2f}\,\%$ (All)".format(
        delta_pc, sigma_pc),
    horizontalalignment="left",)

# Plot residuals
divider = make_axes_locatable(comp_ax)
resid_ax = divider.append_axes("bottom", size="30%", pad=0,)
comp_ax.figure.add_axes(resid_ax, sharex=resid_ax.axes.xaxis)
comp_ax.sharex(resid_ax)

resid_ax.hlines(y=0,xmin=0, xmax=0.75, linestyles="dashed", color="black")

resid_ax.errorbar(
    x=data_tab["rstar_adopt"],
    y=resid,
    xerr=data_tab["e_rstar_adopt"],
    yerr=e_resid,
    zorder=0,
    elinewidth=0.5,
    fmt=".",)

sc2 = resid_ax.scatter(
    data_tab["rstar_adopt"],
    resid,
    c=data_tab["feh_adopt"],
    zorder=1,)

if include_K19_subdwarfs:
    label = "K+19 ({})".format(int(np.sum(adopt_k19)))
        
    scatter = resid_ax.scatter(
        data_tab["rstar_adopt"][adopt_k19],
        resid[adopt_k19],
        marker="o",
        c=data_tab["feh_adopt"][adopt_k19],
        edgecolor="k",
        linewidths=1.2,
        zorder=1,
        label=label,)

# Other formatting
comp_ax.set_ylabel(r"$R_\star$ ($R_\odot$, Fit)")
resid_ax.set_xlabel(r"$R_\star$ ($R_\odot$, Literature)")
resid_ax.set_ylabel(r"Residual ($R_\odot$)")
plt.setp(comp_ax.get_xticklabels(), visible=False)

resid_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))
resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

resid_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.01))
resid_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.05))

comp_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))
comp_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

comp_ax.set_xlim(0.05, 0.75)
comp_ax.set_ylim(0.05, 0.75)

resid_ax.set_xlim(0.05, 0.75)

comp_ax.set_title(r"$M_{{K_S}}-$[Fe/H], $\chi_\nu^2={:0.2f}$".format(rchi2))
fig_fn = "paper/mann_kesseli_MKs_Fe_H_relation_fit"

plt.show()
plt.tight_layout()

# Save plot
plt.savefig("{}.pdf".format(fig_fn))
plt.savefig("{}.png".format(fig_fn), dpi=300)