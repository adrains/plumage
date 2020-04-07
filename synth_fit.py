from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import plumage.synthetic as synth
import plumage.utils as utils
import plumage.plotting as pplt
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Unique label of the fits file of spectra
label = "tess"

# Where to load from and save to
spec_path = "spectra"
save_folder = "fits/tess"

# Load science spectra
spectra_b, spectra_r, observations = utils.load_fits(label, path=spec_path)
bad_px_masks = utils.load_fits_image_hdu("bad_px", label)

do_plotting_after_each_fit = False

# -----------------------------------------------------------------------------
# Do fits
# -----------------------------------------------------------------------------
params_fit = []
fit_results = []
synth_fits_r = []
spec_dicts = []
rchi2 = []

for ob_i in range(0, len(observations)):
    print("-"*40, "\n{}\n".format(ob_i), "-"*40)
    plt.close("all")
    params_init = (
        observations.iloc[ob_i]["teff_fit"],
        observations.iloc[ob_i]["logg_fit"],
        observations.iloc[ob_i]["feh_fit"],
        )

    # Do the fit
    opt_res, spec_dict = synth.do_synthetic_fit(
        spectra_r[ob_i, 0], # Red wl
        spectra_r[ob_i, 1], # Red spec
        spectra_r[ob_i, 2], # Red uncertainties
        params_init, 
        observations.iloc[ob_i]["rv"], 
        observations.iloc[ob_i]["bcor"],
        ~bad_px_masks[ob_i],
        band="red")

    # Save results
    params_fit.append(opt_res["x"])
    fit_results.append(opt_res)
    synth_fits_r.append(spec_dict["spec_synth"])
    spec_dicts.append(spec_dict)
    rchi2.append(np.sum(opt_res["fun"]**2) - (len(opt_res["fun"]-len(params_init))))

    # Plotting
    if do_plotting_after_each_fit:
        date_id = "{}_{}".format(observations.iloc[ob_i]["date"].split("T")[0],
                                observations.iloc[ob_i]["uid"])
        plot_path = os.path.join(save_folder, date_id + ".pdf")

        pplt.plot_synthetic_fit(
            spec_dict["wave"], 
            spec_dict["spec_sci"], 
            spec_dict["e_spec_sci"], 
            spec_dict["spec_synth"], 
            opt_res["x"],
            date_id,
            plot_path,
            masked_regions=bad_px_masks[ob_i]
            )

# Update observations
params_fit = np.stack(params_fit)

observations["teff_synth"] = params_fit[:,0]
observations["logg_synth"] = params_fit[:,1]
observations["feh_synth"] = params_fit[:,2]
observations["rchi2_synth"] = np.array(rchi2)

utils.update_fits_obs_table(observations, label, path=spec_path)

# Save best fit synthetic spectra
synth_fits_r = np.array(synth_fits_r)
utils.save_fits_image_hdu(synth_fits_r, "synth", label, path=spec_path, arm="r")

# Now do final plotting
"""
pplt.plot_all_synthetic_fits(
    spectra_r, 
    synth_fits_r, 
    observations
    bad_px_masks,
    plot_path,
    ref_table=tess_info,
    )
"""