"""Makes a plot comparing the scatter of Cannon models of different orders and
numbers of labels.
"""
import os
import numpy as np
import stannon.stannon as stannon
import matplotlib.pyplot as plt

n_bins = 500

# List of models
linear_models = [
    "stannon_model_basic_KM_3L_5024P_129S_1O_teff_logg_Fe_H.pkl",
    "stannon_model_basic_KM_4L_5024P_129S_1O_teff_logg_Fe_H_Ti_Fe.pkl",
    "stannon_model_basic_KM_5L_5024P_129S_1O_teff_logg_Fe_H_Ti_Fe_Mg_Fe.pkl",
    "stannon_model_basic_KM_6L_5024P_129S_1O_teff_logg_Fe_H_Ti_Fe_Mg_Fe_Ca_Fe.pkl",]

quadratic_models = [
    "stannon_model_basic_KM_3L_5024P_129S_2O_teff_logg_Fe_H.pkl",
    "stannon_model_basic_KM_4L_5024P_129S_2O_teff_logg_Fe_H_Ti_Fe.pkl",
    "stannon_model_basic_KM_5L_5024P_129S_2O_teff_logg_Fe_H_Ti_Fe_Mg_Fe.pkl",
    "stannon_model_basic_KM_6L_5024P_129S_2O_teff_logg_Fe_H_Ti_Fe_Mg_Fe_Ca_Fe.pkl",]

plt.close("all")
fig, (axis_1, axis_2) = plt.subplots(nrows=2, sharex=True, sharey=True,)

fig.subplots_adjust(
    left=0.1,
    bottom=0.1,
    right=0.97,
    top=0.95,
    hspace=0.15,)

bins = None

for model_fn in linear_models:
    cannon_model_path = os.path.join("spectra", model_fn)
    sm = stannon.load_model(cannon_model_path)

    med_s2 = np.nanmedian(sm.s2)

    sm_labels = [r"$T_{\rm eff}$", r"$\log g$", r"[Fe/H]"] + \
        [r"[{}/Fe]".format(lbl.split("_")[0]) for lbl in sm.label_names[3:]]
    sm_labels_str = r"({})".format(", ".join(sm_labels))

    plot_label = r"{}L, {}C, med($s^2$) = {:0.2E}, {}".format(
        sm.L, sm.N_COEFF, med_s2, sm_labels_str)
    (_, bins, _) = axis_1.hist(
        sm.s2,
        bins=n_bins if bins is None else bins,
        alpha=0.5,
        label=plot_label)

axis_1.legend(fontsize="small")

bins = None

for model_fn in quadratic_models:
    cannon_model_path = os.path.join("spectra", model_fn)
    sm = stannon.load_model(cannon_model_path)

    med_s2 = np.nanmedian(sm.s2)

    sm_labels = [r"$T_{\rm eff}$", r"$\log g$", r"[Fe/H]"] + \
        [r"[{}/Fe]".format(lbl.split("_")[0]) for lbl in sm.label_names[3:]]
    sm_labels_str = r"({})".format(", ".join(sm_labels))

    plot_label = r"{}L, {}C, med($s^2$) = {:0.2E}, {}".format(
        sm.L, sm.N_COEFF, med_s2, sm_labels_str)
    
    (_, bins, _) = axis_2.hist(
        sm.s2,
        bins=n_bins if bins is None else bins,
        alpha=0.5,
        label=plot_label)

axis_2.legend(fontsize="small")

axis_1.set_title("1O")
axis_2.set_title("2O")

axis_1.set_yscale("log")

axis_1.set_ylabel(r"$N_{\rm px}$")
axis_2.set_ylabel(r"$N_{\rm px}$")
axis_2.set_xlabel(r"$s^2$")
axis_2.set_xlim(-0.00001,0.002)

plt.savefig("paper/cannon_scatter_comparison.pdf")
plt.savefig("paper/cannon_scatter_comparison.png", dpi=400)