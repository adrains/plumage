"""Simple script to fit RVs of CALSPEC templates.
"""
import plumage.spectra_mike as sm
import plumage.plotting_mike as pm
import matplotlib.pyplot as plt

stellar_templates = {
    "5709390701922940416":"templates/template_HD74000_250624_norm.fits",  # 6211, 4.155, -1.98
    "3510294882898890880":"templates/template_HD111980_250624_norm.fits", # 5578, 3.89, -1.08
    "22745910577134848":"templates/template_HD74000_250624_norm.fits",    # HACK B star, no MARCS
    "4201781696994073472":None,                                           # WD, no MARCS
    "5164707970261890560":"templates/template_epsEri_250709_norm.fits",   # 5052, 4.63, -0.08 (GBS)
    "4376174445988280576":"templates/template_bd02d3375_250709_norm.fits",# 5950,3.97, -2.27 (SIMBAD)
    "5957698605530828032":"templates/template_HD160617_250709_norm.fits", # 5931, 3.74, -1.79 (SIMBAD)
    "6342346358822630400":"templates/template_HD185975_250709_norm.fits",# 5570, 4.12, -0.23 (Gaia DR3)
    "6477295296414847232":"templates/template_HD200654_250709_norm.fits", # 5219, 2.7, -2.88 (SIMBAD)
    "1779546757669063552":"templates/template_HD209458_250709_norm.fits", # 6100, 4.5, 0.03 (SIMBAD)
}

calspec_templates = {
    "5709390701922940416":"data/flux_standards/hd074000_stis_007.txt",
    "3510294882898890880":"data/flux_standards/hd111980_stis_007.txt",
    "22745910577134848":"data/flux_standards/ksi2ceti_stis_008.txt",
    "4201781696994073472":"data/flux_standards/gj7541a_stis_004.txt",
    "5164707970261890560":"data/flux_standards/hd022049.dat",
    "4376174445988280576":"data/flux_standards/bd02d3375_stis_008.txt",
    "5957698605530828032":"data/flux_standards/hd160617_stis_006.txt",
    "6342346358822630400":"data/flux_standards/hd185975_stis_008.txt",
    "6477295296414847232":"data/flux_standards/hd200654_stis_008.txt",
    "1779546757669063552":"data/flux_standards/hd209458_stisnic_008.txt",}

# Settings
resolving_power = 1000
rv_min = -750
rv_max = 750
delta_rv = 1

# Output arrays
used_ids = []
fit_all_rvs = []

fig, axes = plt.subplots(nrows=len(stellar_templates), sharex=True)

# Loop over all stars, do RV cross-correlation, and make diagnostic plot
for star_i, source_id in enumerate(stellar_templates.keys()):
    if stellar_templates[source_id] is None:
        print('{}:0,'.format(source_id))
        continue

    rv_fit_dict = sm.fit_rv_for_flux_template(
        calspec_temp_fn=calspec_templates[source_id],
        calspec_synth_fn=stellar_templates[source_id],
        resolving_power=resolving_power,
        rv_min=rv_min,
        rv_max=rv_max,
        delta_rv=delta_rv,)
    
    used_ids.append(source_id)
    fit_all_rvs.append(rv_fit_dict)

    axes[star_i].plot(rv_fit_dict["rv_steps"], rv_fit_dict["cross_corrs"])

    pm.plot_all_cc_rv_diagnostics(
        all_rv_fit_dicts=fit_all_rvs,
        obj_names=used_ids,
        figsize=(16,4),
        fig_save_path="plots/rv_diagnostics",)
    
    print('{}:{:0.0f},'.format(source_id, rv_fit_dict["rv"]))