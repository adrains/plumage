from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import plumage.synthetic as synth
import plumage.utils as utils
import plumage.plotting as pplt
import matplotlib.pyplot as plt

sample = "tess"
save_folder = "fits/tess"

# Import data
if sample == "standard":
    observations = utils.load_observations_fits("standard")
    spectra_b = utils.load_spectra_fits("b", "std")
    spectra_r = utils.load_spectra_fits("r", "std")

elif sample == "tess":
    observations = utils.load_observations_fits("tess")
    spectra_b = utils.load_spectra_fits("b", "tess")
    spectra_r = utils.load_spectra_fits("r", "tess")

fit_results = []
best_fit_spec = []

for ob_i in range(0, len(observations)):
    print("-"*40, "\n{}\n".format(ob_i), "-"*40)
    plt.close("all")
    params = (
        observations.iloc[ob_i]["teff_fit"],
        observations.iloc[ob_i]["logg_fit"],
        observations.iloc[ob_i]["feh_fit"],
        )

    # Do the fit
    opt_res, spec_dict = synth.do_synthetic_fit(
        spectra_r[ob_i, 0], # Red wl
        spectra_r[ob_i, 1], # Red spec
        spectra_r[ob_i, 2], # Red uncertainties
        params, 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"],
        band="red")

    fit_results.append(opt_res)
    best_fit_spec.append(best_fit_spec)

    # Plotting
    date_id = "{}_{}".format(observations.iloc[ob_i]["date"].split("T")[0],
                               observations.iloc[ob_i]["uid"])
    plot_path = os.path.join(save_folder, date_id + ".pdf")

    pplt.plot_synthetic_fit(
        spec_dict["wave"][spec_dict["wl_mask"]], 
        spec_dict["spec_sci"][spec_dict["wl_mask"]], 
        spec_dict["e_spec_sci"][spec_dict["wl_mask"]], 
        spec_dict["spec_synth"][spec_dict["wl_mask"]], 
        opt_res["x"],
        date_id,
        plot_path,
        )