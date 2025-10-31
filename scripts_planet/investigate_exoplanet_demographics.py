"""Script to investigate exoplanet demographics.

Files:
 - scripts_planet/planet_settings.yml
"""
import numpy as np
import pandas as pd
import plumage.utils as pu
import stannon.utils as su
import stannon.stannon as stannon
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from numpy.polynomial.polynomial import Polynomial, polyfit

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
# Import settings
planet_settings_yaml = "scripts_planet/planet_settings.yml"
ps = su.load_yaml_settings(planet_settings_yaml)

# Import literature Gaia + 2MASS info
host_info = pu.load_info_cat(
    path=ps.star_info,
    use_mann_code_for_masses=ps.use_mann_code_for_masses,
    in_paper=ps.in_paper,
    only_observed=ps.only_observed,
    do_extinction_correction=ps.do_extinction_correction,
    do_skymapper_crossmatch=ps.do_skymapper_crossmatch,
    gdr=ps.gdr,
    do_use_mann_15_JHK=ps.do_use_mann_15_JHK,)

# Import observations dataframe and crossmatch
obs_df = pu.load_fits_table(
    "OBS_TAB", ps.label, path=ps.path, do_use_dr3_id=ps.do_use_dr3_id)
obs_join = obs_df.join(host_info, "source_id_dr3", rsuffix="_info")

# Load in RV corrected science spectra
wave_sci_br = pu.load_fits_image_hdu(
"rest_frame_wave", ps.label, arm="br")
spec_sci_br = pu.load_fits_image_hdu(
    "rest_frame_spec", ps.label, arm="br")
e_spec_sci_br = pu.load_fits_image_hdu(
    "rest_frame_sigma", ps.label, arm="br")

# Exclude any overluminous stars
is_useful = ~obs_join["overluminous"].values
obs_join = obs_join[is_useful].copy()
spec_sci_br = spec_sci_br[is_useful]
e_spec_sci_br = e_spec_sci_br[is_useful]

# Also import benchmark sample for plotting later
fits_ext_label = "{}_{}L_{}P_{}S".format("MK", 4, 5024, 199)
cannon_df = pu.load_fits_table(
    extension="CANNON_MODEL",
    label=ps.benchmark_label,
    path=ps.path,
    ext_label=fits_ext_label)

adopted_benchmark = cannon_df["adopted_benchmark"].values

benchmark_df = pu.load_fits_table("CANNON_INFO", ps.benchmark_label)
benchmark_df = benchmark_df[adopted_benchmark]

#------------------------------------------------------------------------------
# Import and normalise spectra
#------------------------------------------------------------------------------
# Normalise
fluxes_norm, ivars_norm, bad_px_mask, continua, adopted_wl_mask = \
    stannon.prepare_cannon_spectra_normalisation(
        wls=wave_sci_br,
        spectra=spec_sci_br,
        e_spectra=e_spec_sci_br,
        wl_max_model=ps.wl_max_model,
        wl_min_normalisation=ps.wl_min_normalisation,
        wl_broadening=ps.wl_broadening,
        do_gaussian_spectra_normalisation=ps.do_gaussian_spectra_normalisation,
        poly_order=ps.poly_order)

# Apply bad pixel mask
fluxes_norm[bad_px_mask] = 1
ivars_norm[bad_px_mask] = 0

#------------------------------------------------------------------------------
# Predict Stellar Properties
#------------------------------------------------------------------------------
sm = stannon.load_model(ps.cannon_model_fn)

# Predict labels
pred_label_values, pred_label_sigmas_stat, chi2_all = sm.infer_labels(
    fluxes_norm[:,adopted_wl_mask],
    ivars_norm[:,adopted_wl_mask])

for lbl_i, lbl in enumerate(sm.label_names):
    obs_join["{}_value_cannon".format(lbl)] = pred_label_values[:,lbl_i]
    obs_join["{}_sigma_cannon".format(lbl)] = pred_label_sigmas_stat[:,lbl_i]

#------------------------------------------------------------------------------
# Candidate Planets
#------------------------------------------------------------------------------
# New table of planet parameters. Start with host ID, then query for all planet
# IDs. Once we have the list of IDs, crossmatch TIC IDs to ExoFOP info, and
# for confirmed planets using DR3 ID 
exofop_tois = pd.read_csv(ps.candidate_planet_info, delimiter=",")

observed_tics = obs_join["TIC"].values
all_tics = exofop_tois["TIC"].values

is_observed = np.isin(all_tics, observed_tics)

exofop_tois_observed = exofop_tois[is_observed].copy()
unique_tois = exofop_tois_observed["TOI"].values

# Add in a 'pl_name' column for crossmatching later
pl_names = ["TOI {}".format(toi) for toi in exofop_tois_observed["TOI"].values]
exofop_tois_observed["pl_name"] = pl_names

exofop_tois_observed.set_index("pl_name", inplace=True)

#------------------------------------------------------------------------------
# Confirmed Planets
#------------------------------------------------------------------------------
# Import confirmed planets
confirmed_planets = pd.read_csv(
    ps.confirmed_planet_info,
    comment="#",
    delimiter="\t",
    dtype={"source_id_dr3":str})

# Only keep rows associated with our 'adopted' reference, one per planet
adopt_ref = confirmed_planets["adopted"].values
confirmed_planets = confirmed_planets[adopt_ref].copy()

confirmed_planets.set_index("pl_name", inplace=True)

#------------------------------------------------------------------------------
# Collate planet info
#------------------------------------------------------------------------------
# For all TICs in obs_join, collate a list of planet parameters by checking a)
# confirmed planets and b) candidate planets, and constructing the dataframe
# as we go.


# For all TICS in obs_join, go through confirmed and candidate planet tables
# (crossmatching via TIC) to create a list of planet IDs. Then, now that we
# know the size of the table, create a master# dataframe with relevant info
# from all three tables, keeping information separate from confirmed/TOI, but
# also collating into 'master' columns for the parameters of interest.

# Get size of array
planet_ids = []
tic_ids = []
dispositions = []

for sid, star_info in obs_join.iterrows():
    tic = star_info["TIC"]

    # Check if confirmed
    if np.isin(tic, confirmed_planets["TIC"].values):
        # Grab all matches (multi-systems will appear several times)
        planet_row_ii = np.atleast_1d(np.squeeze(
            np.argwhere(tic == confirmed_planets["TIC"].values)))
        
        for row_i in planet_row_ii:
            confirmed_info = confirmed_planets.iloc[row_i]

            planet_ids.append(confirmed_info.name)
            tic_ids.append(tic)
            dispositions.append("confirmed")

    # Must be candidate
    else:
        # Grab all matches (multi-systems will appear several times)
        planet_row_ii = np.atleast_1d(np.squeeze(
            np.argwhere(tic == exofop_tois_observed["TIC"].values)))
        
        for row_i in planet_row_ii:
            exofop_info = exofop_tois_observed.iloc[row_i]

            planet_ids.append(exofop_info.name)
            tic_ids.append(tic)
            dispositions.append(exofop_info["TFOPWG Disposition"])

planet_info = pd.DataFrame(
    data=np.array([planet_ids, tic_ids, dispositions]).T,
    columns=["pl_name", "TIC", "disposition"],)

planet_info = planet_info.astype({"TIC": int,})
planet_info.set_index("pl_name", inplace=True)

# Now we can crossmatch!
planet_info = planet_info.join(confirmed_planets, rsuffix="_2")     # known
planet_info = planet_info.join(exofop_tois_observed, rsuffix="_3")  # candidate
planet_info = planet_info.join(                                     # host info
    obs_join.reset_index().set_index("TIC"),
    on="TIC",
    rsuffix="_4")

is_confirmed = planet_info["disposition"] == "confirmed"

#------------------------------------------------------------------------------
# Adopting Parameters
#------------------------------------------------------------------------------
# Adopted period
periods = []
radii = []

for sid, row_info in planet_info.iterrows():
    #-------------
    # Period
    # ------------
    # Confirmed
    if not np.isnan(row_info["pl_orbper"]):
        periods.append(row_info["pl_orbper"])

    # Candidate
    elif not np.isnan(row_info["Period (days)"]):
        periods.append(row_info["Period (days)"])

    # Missing
    else:
        periods.append(np.nan)

    #-------------
    # Radii
    # ------------
    # Confirmed
    if not np.isnan(row_info["pl_rade"]):
        radii.append(row_info["pl_rade"])

    # Candidate
    elif not np.isnan(row_info["Planet Radius (R_Earth)"]):
        radii.append(row_info["Planet Radius (R_Earth)"])

    # Missing
    else:
        radii.append(np.nan)

periods = np.array(periods)
planet_info["period_adopt"] = periods

radii = np.array(radii)
planet_info["radius_adopt"] = radii

# Also divide the planet sample into classes (rocky, sub-neptune, giant)
planet_class = np.full(len(radii), "").astype(object)
planet_class[radii <= 1.8] = "rocky"
planet_class[np.logical_and(radii > 1.8, radii <= 3.5)] = "sub-Neptune"
planet_class[radii > 3.5] = "giant"

planet_info["class"] = planet_class

#------------------------------------------------------------------------------
# Plot #0: CMD
#------------------------------------------------------------------------------
plt.close("all")
fig, ax_cmd = plt.subplots(ncols=1, nrows=1)

fehs_all = np.concatenate([planet_info["Fe_H_value_cannon"].values, 
                           benchmark_df["label_adopt_Fe_H"].values])

# Determine min/max radii
vmin = np.nanmin(fehs_all)
vmax = np.nanmax(fehs_all)

# Confirmed
sc = ax_cmd.scatter(
    x=benchmark_df["BP_RP_dr3"].values,
    y=benchmark_df["K_mag_abs"].values,
    c=benchmark_df["label_adopt_Fe_H"].values,
    label="Benchmark ({})".format(np.sum(adopted_benchmark)),
    #alpha=0.8,
    marker="o",
    #edgecolors="k",
    vmin=vmin,
    vmax=vmax,)

cb = fig.colorbar(sc, ax=ax_cmd)
cb.set_label("[Fe/H]")

ax_cmd.set_xlabel("BP-RP")
ax_cmd.set_ylabel(r"$M_{K_S}$")
ax_cmd.set_ylim(9.75,4)
ax_cmd.set_xlim(1.05,4.65)
plt.tight_layout()

leg = ax_cmd.legend()
plt.savefig("plots/sochias_cmd_0.pdf")
plt.savefig("plots/sochias_cmd_0.png", dpi=300)

# Confirmed
confirmed_host = obs_join["disposition"] == "confirmed"

_ = ax_cmd.scatter(
    x=obs_join["BP-RP_dr3"].values[confirmed_host],
    y=obs_join["K_mag_abs"].values[confirmed_host],
    c=obs_join["Fe_H_value_cannon"].values[confirmed_host],
    label="Confirmed ({})".format(np.sum(confirmed_host)),
    alpha=0.6,
    marker="^",
    edgecolors="k",
    vmin=vmin,
    vmax=vmax,)

leg = ax_cmd.legend()
plt.savefig("plots/sochias_cmd_1.pdf")
plt.savefig("plots/sochias_cmd_1.png", dpi=300)

# Candidate
_ = ax_cmd.scatter(
    x=obs_join["BP-RP_dr3"].values[~confirmed_host],
    y=obs_join["K_mag_abs"].values[~confirmed_host],
    c=obs_join["Fe_H_value_cannon"].values[~confirmed_host],
    label="Candidate ({})".format(np.sum(~confirmed_host)),
    alpha=0.6,
    marker="s",
    edgecolors="k",
    vmin=vmin,
    vmax=vmax,)

leg = ax_cmd.legend()
plt.savefig("plots/sochias_cmd_2.pdf")
plt.savefig("plots/sochias_cmd_2.png", dpi=300)


#------------------------------------------------------------------------------
# Plot #1: [Fe/H]-[Ti/Fe]-R_p
#------------------------------------------------------------------------------
#plt.close("all")
fig, ax_chem = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(8,4))

fig.subplots_adjust(
    left=0.10,
    bottom=0.15,
    right=0.99,
    top=0.975,)

# Determine min/max radii
min_re = np.nanmin(radii)
max_re = np.nanmax(radii)

confirmed_with_radius = np.logical_and(
    is_confirmed, ~np.isnan(planet_info["pl_rade"].values))

# Panel 1
sc = ax_chem[0].scatter(
    planet_info["Fe_H_value_cannon"].values[confirmed_with_radius],
    planet_info["Ti_Fe_value_cannon"].values[confirmed_with_radius],
    c=planet_info["pl_rade"].values[confirmed_with_radius],
    label="Confirmed ({})".format(np.sum(confirmed_with_radius)),
    marker="o",
    vmin=min_re,
    vmax=max_re,)

_ = ax_chem[0].scatter(
    planet_info["Fe_H_value_cannon"].values[~is_confirmed],
    planet_info["Ti_Fe_value_cannon"].values[~is_confirmed],
    c=planet_info["Planet Radius (R_Earth)"].values[~is_confirmed],
    label="Candidate ({})".format(np.sum(~is_confirmed)),
    marker="^",
    vmin=min_re,
    vmax=max_re,)

cb = fig.colorbar(sc, ax=ax_chem)
cb.set_label("$R_P~(R_E)$")

ax_chem[0].set_ylabel("[Ti/Fe]")

leg = ax_chem[0].legend()

# Panel 2
for pl_class in ["rocky", "sub-Neptune", "giant"]:
    mm = planet_info["class"].values == pl_class
    ax_chem[1].hist(
        x=planet_info["Fe_H_value_cannon"].values[mm],
        alpha=0.6,
        label="{} ({})".format(pl_class, np.sum(mm)),)

leg = ax_chem[1].legend(ncol=3,)
ax_chem[1].set_xlabel("[Fe/H]")
ax_chem[1].set_ylabel("# Planet")
ax_chem[1].set_ylim(0,25)

plt.savefig("plots/sochias_Ti_Fe_Fe_H_radius.pdf")
plt.savefig("plots/sochias_Ti_Fe_Fe_H_radius.png", dpi=300)

#------------------------------------------------------------------------------
# Plot #1: Sample Histograms
#------------------------------------------------------------------------------
fig, ax_hists = plt.subplots(ncols=3, nrows=1,sharey=False, figsize=(8,2))

fig.subplots_adjust(
    left=0.07,
    bottom=0.25,
    right=0.99,
    top=0.975,
    hspace=0.000,
    wspace=0.25)

ax_hists[0].hist(obs_join["mass_m19"].values, bins=10, color="C0", alpha=0.8)
ax_hists[0].set_xlabel(r"${\rm M}_\star~({\rm M}_\odot)$")

ax_hists[1].hist(
    obs_join["Fe_H_value_cannon"].values, bins=10, color="C1", alpha=0.8)
ax_hists[1].set_xlabel("[Fe/H]")

ax_hists[2].hist(
    planet_info["radius_adopt"].values, bins=25, color="C2", alpha=0.8)
ax_hists[2].set_xlabel("$R_P~(R_E)$")

ss_planets = [("J", 11.209, 10), ("S",9.14, 10), ("N",3.88, 10)]
ap = dict(facecolor="r", shrink=0.05, headwidth=0.2, width=0.1)

for ssp in ss_planets:
    ax_hists[2].annotate(
        text="$R_{}$".format(ssp[0]),
        xy=(ssp[1],0),
        xytext=(ssp[1], ssp[2]),
        arrowprops=ap,
        horizontalalignment="center")

ax_hists[0].set_ylabel("# Host")
ax_hists[1].set_ylabel("# Host")
ax_hists[2].set_ylabel("# Planet")

# Ticks
ax_hists[0].xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
ax_hists[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))

ax_hists[1].xaxis.set_major_locator(plticker.MultipleLocator(base=0.3))
ax_hists[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.15))

ax_hists[2].xaxis.set_major_locator(plticker.MultipleLocator(base=5))
ax_hists[2].xaxis.set_minor_locator(plticker.MultipleLocator(base=1))

#plt.tight_layout()

plt.savefig("plots/sochias_hist_1.pdf")
plt.savefig("plots/sochias_hist_1.png", dpi=300)

# -------
fig, ax_hist_pl = plt.subplots(ncols=1, nrows=1, figsize=(3,3))

rr = planet_info["radius_adopt"].values
is_small = rr < 4

ax_hist_pl.hist(rr[is_small], bins=10, color="C2", alpha=0.8)
ax_hist_pl.set_xlabel("$R_P~(R_E)$")

ss_planets = [("J", 11.209, 10), ("S",9.14, 10), ("N",3.88, 10)]
ap = dict(facecolor="r", shrink=0.05, headwidth=0.2, width=0.1)

for ssp in ss_planets:
    ax_hist_pl.annotate(
        text="$R_{}$".format(ssp[0]),
        xy=(ssp[1],0),
        xytext=(ssp[1], ssp[2]),
        arrowprops=ap,
        horizontalalignment="center")


ax_hist_pl.set_ylabel("# Planet")

# Ticks
ax_hist_pl.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
ax_hist_pl.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))

plt.tight_layout()


#------------------------------------------------------------------------------
# Plot #1: [Fe/H]-[Ti/Fe]-R_p
#------------------------------------------------------------------------------
fig, ax_2 = plt.subplots(ncols=1, nrows=1)

# Determine min/max radii
fehs = planet_info["Fe_H_value_cannon"].values
bins = np.linspace(np.min(fehs), np.max(fehs), 6)

classes = ["rocky", "sub-Neptune", "giant"]

for hist_i, p_class in enumerate(classes):
    class_mask = planet_class == p_class

    ax_2.hist(
        x=planet_info["Fe_H_value_cannon"].values[class_mask],
        bins=bins,
        label=p_class,
        alpha=0.5,)


ax_2.set_xlabel("[Fe/H]")
ax_2.set_ylabel("N Planet")

leg = ax_2.legend()
plt.tight_layout()

#------------------------------------------------------------------------------
# Plot -- close/far
#------------------------------------------------------------------------------
fig, axes_close_far = plt.subplots()
period_threshold = 10
period_mask = periods < period_threshold

Fe_Hs = planet_info["Fe_H_value_cannon"].values

bins = np.linspace(np.nanmin(Fe_Hs), np.nanmax(Fe_Hs), 6)

plt.hist(
    Fe_Hs[period_mask],
    bins=bins,
    alpha=0.6,
    label="Period < {} days".format(period_threshold, np.sum(period_mask)))

plt.hist(
    Fe_Hs[~period_mask],
    bins=bins,
    alpha=0.6,
    label="Period >= {} days".format(period_threshold, np.sum(period_mask)))

leg = axes_close_far.legend()


#------------------------------------------------------------------------------
# Plot #1: [Fe/H]-[Ti/Fe]-R_p
#------------------------------------------------------------------------------
fig, axes_2 = plt.subplots(ncols=1, nrows=3, figsize=(8,4), sharex=True)

# Determine min/max radii
fehs = planet_info["Fe_H_value_cannon"].values
bins = np.linspace(np.min(fehs), np.max(fehs), 6)

classes = ["rocky", "sub-Neptune", "giant"]

for hist_i, p_class in enumerate(classes):
    class_mask = planet_class == p_class

    axes_2[hist_i].hist(
        x=planet_info["Fe_H_value_cannon"].values[class_mask],
        bins=bins,
        label=p_class,
        alpha=0.5,)


    axes_2[hist_i].set_xlabel("[Fe/H]")
    axes_2[hist_i].set_ylabel("N Planet")

    leg = axes_2[hist_i].legend()

plt.tight_layout()

#------------------------------------------------------------------------------
# Plot #3: [Fe/H]-[Ti/Fe]-R_p
#------------------------------------------------------------------------------
# [X/Fe] vs Density for *rocky* planets
all_species = ["Fe_H", "Ti_Fe", "Ca_Fe", "Mg_Fe"]
x_scale_major = [0.2,]

is_rocky = planet_class == "rocky"

density_sigma = np.array([
    -1*planet_info["pl_denserr2"].values, planet_info["pl_denserr1"].values])

density_sig_max = np.nanmax(density_sigma, axis=0)

has_density_uncertainties = ~np.isnan(planet_info["pl_denserr1"].values)

high_precision_density = (density_sig_max / planet_info["pl_dens"].values) < 0.3

passed_cuts = np.all(
    (is_rocky, has_density_uncertainties, high_precision_density), axis=0)

fig, ax_3 = plt.subplots(
    nrows=1, ncols=len(all_species), figsize=(8,2.5), sharey=True,)

fig.subplots_adjust(
    left=0.08,
    bottom=0.2,
    right=1.05,
    #top=0.975,
    #hspace=0.000,
    wspace=0.05,)


for si, species in enumerate(all_species):

    X_Fe_col = "{}_value_cannon".format(species)
    X_Fe_label = "[{}]".format(species.replace("_", "/"))

    sc = ax_3[si].scatter(
        x=planet_info[X_Fe_col].values[passed_cuts],
        y=planet_info["pl_dens"].values[passed_cuts],
        c=planet_info["pl_rade"].values[passed_cuts],
        label=r"Rocky Planets with $\sigma_\rho < 35\,$% ({})".format(
            np.sum(passed_cuts)),
        marker="o",)

    ax_3[si].errorbar(
        x=planet_info[X_Fe_col].values[passed_cuts],
        y=planet_info["pl_dens"].values[passed_cuts],
        yerr=density_sigma[:,passed_cuts],
        fmt=".",
        marker=".",
        ecolor="k",
        zorder=-1,)



    ax_3[si].set_xlabel("{}".format(X_Fe_label))

    if si == 0:
        ax_3[si].set_ylabel("Density (g cm$^{-3}$)")
        ax_3[si].xaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
        ax_3[si].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    elif si == 3:
        ax_3[si].xaxis.set_major_locator(plticker.MultipleLocator(base=0.05))
        ax_3[si].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.025))
    else:
        ax_3[si].xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax_3[si].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.05))

    #leg = ax_3[si].legend()

cb = fig.colorbar(sc, ax=ax_3, pad=0.01,)#, location="bottom")
cb.set_label("$R_P~(R_E)$")

    
plt.suptitle(
    r"Rocky ($R_P \leq 1.8~R_E$) Planets with $\sigma_\rho < 30\,$% ({})".format(
        np.sum(passed_cuts)))
#plt.tight_layout()
plt.savefig("plots/sochias_density_vs_abund.pdf")
plt.savefig("plots/sochias_density_vs_abund.png", dpi=300)

    # Fitting
    #coef = polyfit(
    #    x=planet_info[X_Fe_col].values[passed_cuts],
    #    y=planet_info["pl_dens"].values[passed_cuts],
    #    w=1/np.max(density_sigma[:,passed_cuts], axis=0),
    #    deg=1,)

# Save the polynomial and continuum pixels
#poly = Polynomial(coef)

#xx = np.linspace(
#    np.min(planet_info[X_Fe_col].values[passed_cuts]),
#    np.max(planet_info[X_Fe_col].values[passed_cuts]),
#    20,)

#ax_3.plot(xx, poly(xx), "-", c="r")


