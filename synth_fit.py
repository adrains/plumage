from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import plumage.synthetic as synth
import plumage.utils as utils
import matplotlib.pyplot as plt

# Import data
observations = utils.load_observations_fits("standard")
spectra_b = utils.load_spectra_fits("b", "std")
spectra_r = utils.load_spectra_fits("r", "std")

for ob_i in range(len(observations)):
    print("-"*40, "\n{}\n".format(ob_i), "-"*40)
    plt.close("all")
    params = (
        observations.iloc[ob_i]["teff_fit"],
        observations.iloc[ob_i]["logg_fit"],
        observations.iloc[ob_i]["feh_fit"],
        )

    # Do the fit
    xx = synth.do_synthetic_fit(
        spectra_r[ob_i, 0], 
        spectra_r[ob_i, 1], 
        spectra_r[ob_i, 2], 
        params, 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"],
        band="red")

    # Save the plot
    date_id = "{}_{}".format(observations.iloc[ob_i]["date"].split("T")[0],
                               observations.iloc[ob_i]["uid"])
    plt.title(date_id)
    plot_path = os.path.join("fits", date_id + ".pdf")
    plt.savefig(plot_path)