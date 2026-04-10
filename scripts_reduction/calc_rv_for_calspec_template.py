"""Simple script to fit RVs of CALSPEC templates.
"""
import os
import pandas as pd
import plumage.spectra_mike as sm
import plumage.plotting_mike as pm
import matplotlib.pyplot as plt
import stannon.utils as su

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

# Import flux standard dataframe.
flux_std_info = pd.read_csv(
    filepath_or_buffer=ms.flux_cal_std_info_fn,
    dtype={"source_id":str},
    delimiter="\t",)
flux_std_info.set_index("source_id", inplace=True)

# Properly handle missing templates by repacing np.nan with None.
marcs_templates = flux_std_info["marcs_template"].values
missing_template = [type(ff) != str for ff in marcs_templates]
marcs_templates[missing_template] = None
flux_std_info["marcs_template"] = marcs_templates

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Settings
resolving_power = 1000
rv_min = -750
rv_max = 750
delta_rv = 5

save_path = os.path.join(ms.reduction_folder, "spphot_rv_diagnostics")

# Output arrays
used_ids = []
fit_all_rvs = []

fig, axes = plt.subplots(nrows=len(flux_std_info), sharex=True)

# Loop over all stars, do RV cross-correlation, and make diagnostic plot
for si, (source_id, star_data) in enumerate(flux_std_info.iterrows()):
    if star_data["marcs_template"] is None:
        print('{}:0,'.format(source_id))
        continue

    rv_fit_dict = sm.fit_rv_for_flux_template(
        calspec_temp_fn=star_data["calspec_fn"],
        calspec_synth_fn=star_data["marcs_template"],
        resolving_power=resolving_power,
        rv_min=rv_min,
        rv_max=rv_max,
        delta_rv=delta_rv,)
    
    used_ids.append(source_id)
    fit_all_rvs.append(rv_fit_dict)

    axes[si].plot(rv_fit_dict["rv_steps"], rv_fit_dict["cross_corrs"])

    pm.plot_all_cc_rv_diagnostics(
        all_rv_fit_dicts=fit_all_rvs,
        obj_names=used_ids,
        figsize=(16,4),
        fig_save_path=save_path,
        run_in_wavelength_scale_debug_mode=False,)
    
    print('{}:{:0.0f},'.format(source_id, rv_fit_dict["rv"]))