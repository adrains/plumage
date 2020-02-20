from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import plumage.synthetic as synth
import plumage.utils as utils
import matplotlib.pyplot as plt

sample = "standard"
save_folder = "fits/standards"

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

for ob_i in range(0, len(observations)):
    print("-"*40, "\n{}\n".format(ob_i), "-"*40)
    plt.close("all")
    params = (
        observations.iloc[ob_i]["teff_fit"],
        observations.iloc[ob_i]["logg_fit"],
        observations.iloc[ob_i]["feh_fit"],
        )

    # Do the fit
    xx = synth.do_synthetic_fit(
        spectra_r[ob_i, 0], # Red wl
        spectra_r[ob_i, 1], # Red spec
        spectra_r[ob_i, 2], # Red uncertainties
        params, 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"],
        band="red")

    fit_results.append(xx)

    # Save the plot
    date_id = "{}_{}".format(observations.iloc[ob_i]["date"].split("T")[0],
                               observations.iloc[ob_i]["uid"])
    plt.suptitle(date_id)
    plot_path = os.path.join(save_folder, date_id + ".pdf")
    plt.savefig(plot_path)