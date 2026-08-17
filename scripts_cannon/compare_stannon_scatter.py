"""Makes a plot comparing the scatter of Cannon models of different orders and
numbers of labels.
"""
import os
import numpy as np
import stannon.stannon as stannon
import matplotlib.pyplot as plt

n_bins = 1500

# List of models
linear_models = [
    "stannon_model_basic_M_3L_5024P_141S_1O_teff_logg_Fe_H.pkl",
    "stannon_model_basic_M_4L_5024P_141S_1O_teff_logg_Fe_H_Ti_Fe.pkl",
    "stannon_model_basic_M_5L_5024P_141S_1O_teff_logg_Fe_H_Ti_Fe_Mg_Fe.pkl",
    "stannon_model_basic_M_6L_5024P_141S_1O_teff_logg_Fe_H_Ti_Fe_Mg_Fe_Ca_Fe.pkl",]

quadratic_models = [
    "stannon_model_basic_M_3L_5024P_141S_2O_teff_logg_Fe_H.pkl",
    "stannon_model_basic_M_4L_5024P_141S_2O_teff_logg_Fe_H_Ti_Fe.pkl",
    "stannon_model_basic_M_5L_5024P_141S_2O_teff_logg_Fe_H_Ti_Fe_Mg_Fe.pkl",
    "stannon_model_basic_M_6L_5024P_141S_2O_teff_logg_Fe_H_Ti_Fe_Mg_Fe_Ca_Fe.pkl",]

plt.close("all")
fig, (axis_1, axis_2) = plt.subplots(nrows=2, sharex=True, sharey=True,)

bins = None

for model_fn in linear_models:
    cannon_model_path = os.path.join("spectra", model_fn)
    sm = stannon.load_model(cannon_model_path)

    med_s2 = np.nanmedian(sm.s2)
    label = "{}O, {}L, s2 = {:0.2E}".format(sm.O, sm.L, med_s2)
    (_, bins, _) = axis_1.hist(
        sm.s2,
        bins=n_bins if bins is None else bins,
        alpha=0.5,
        label=label)

axis_1.legend()

bins = None

for model_fn in quadratic_models:
    cannon_model_path = os.path.join("spectra", model_fn)
    sm = stannon.load_model(cannon_model_path)

    med_s2 = np.nanmedian(sm.s2)
    label = "{}O, {}L, s2 = {:0.2E}".format(sm.O, sm.L, med_s2)
    (_, bins, _) = axis_2.hist(
        sm.s2,
        bins=n_bins if bins is None else bins,
        alpha=0.5,
        label=label)

axis_2.legend()

axis_1.set_ylabel(r"$N_{\rm px}$")
axis_2.set_ylabel(r"$N_{\rm px}$")
axis_2.set_xlabel("s2")
axis_2.set_xlim(-0.00001,0.0015)
plt.tight_layout()

plt.savefig("paper/cannon_scatter_comparison.pdf")