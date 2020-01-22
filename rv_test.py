"""Let's use Gaia DR2 3359074685047632640 (HD 260655) and 
Gaia DR2 2452378776434276992 (Tau Cet) as references - the former has 
+/-11 km/s uncertainties, and the latter +/- 0.11 km/s

Tau Cet: 5,350 K, 4.4, -0.5
HD 260655: 3,800 K, 4.9, -0.34
"""

"""Script to process science spectra

Link if the IERS is iffy: https://github.com/astropy/astropy/issues/8981
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plumage.synthetic as synth
import plumage.spectra as spec
import plumage.plotting as pplt
import plumage.utils as utils
from astropy.table import Table
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Setup + Import
# -----------------------------------------------------------------------------
n_spec = 516                            # If loading, which pickle of N spectra

cat_type="csv"                          # Crossmatch catalogue type
cat_file="data/all_2m3_star_ids.csv"    # Crossmatch catalogue 

# Load in science spectra
print("Importing science spectra...")
observations, spectra_b, spectra_r = spec.load_pkl_spectra(n_spec) 

# Only grab the stars we're interested in
mask = np.logical_or(observations["id"]=="2452378776434276992", 
                     observations["id"]=="3359074685047632640")

observations = observations[mask]
spectra_b = spectra_b[mask]
spectra_r = spectra_r[mask]

# -----------------------------------------------------------------------------
# Normalise science and template spectra
# -----------------------------------------------------------------------------
print("Normalise red science spectra...")
spectra_r_norm = spec.normalise_spectra(spectra_r, True)

# Load in template spectra
print("Load in synthetic templates...")
ref_params, ref_spec = synth.load_synthetic_templates(setting="R7000")

# Normalise template spectra
print("Normalise synthetic templates...")
ref_spec_norm = spec.normalise_spectra(ref_spec)  

# -----------------------------------------------------------------------------
# Compute barycentric correction
# -----------------------------------------------------------------------------
print("Compute barycentric corrections...")

bcors = spec.compute_barycentric_correction(observations["ra"], 
                                            observations["dec"], 
                                            observations["mjd"], "SSO",
                                            disable_auto_max_age=False)
observations["bcor"] = bcors

tau_cet_i = 0
hd260655_i = 1

# -----------------------------------------------------------------------------
# Calculate RVs and RV correct
# -----------------------------------------------------------------------------
print("Compute RVs...")
rvs, e_rvs, rchi2, all_nres, params, grid_rchi2 = spec.do_all_template_matches(
    spectra_r_norm, 
    observations, 
    ref_params, 
    ref_spec_norm,)# print_diagnostics=True)

# Tau Cet
rv_tc, e_rv_tc, rchi2_tc, infodict_tc = spec.calc_rv_shift(
    ref_spec_norm[1,0], 
    ref_spec_norm[1,1], 
    spectra_r_norm[tau_cet_i,0], 
    spectra_r_norm[tau_cet_i,1], 
    spectra_r_norm[tau_cet_i,2], 
    bcors[tau_cet_i])

# HD260655
rv_hd, e_rv_hd, rchi2_hd, infodict_hd = spec.calc_rv_shift(
    ref_spec_norm[0,0], 
    ref_spec_norm[0,1], 
    spectra_r_norm[hd260655_i,0], 
    spectra_r_norm[hd260655_i,1], 
    spectra_r_norm[hd260655_i,2], 
    bcors[hd260655_i])

observations["rv"] = [rv_tc, rv_hd]
observations["e_rv"] = [e_rv_tc, e_rv_hd]

# Red arm
wl_min_r = 5400
wl_max_r = 7000
n_px_r = 3637
wl_per_pixel_r = (wl_max_r - wl_min_r) / n_px_r 
wl_new_r = np.arange(wl_min_r, wl_max_r, wl_per_pixel_r) 

# RV correct the spectra
spec_rvcor_r = spec.correct_all_rvs(spectra_r_norm, observations, wl_new_r)

# Import in files
tc_fits = fits.open("spectra/20190722/20190722_2452378776434276992_r.fits")
hd_fits = fits.open("spectra/20191014/20191014_3359074685047632640_r.fits")

tc_data = tc_fits[1].data
hd_data = hd_fits[1].data

# Diagnostic plotting
plt.close("all")

alpha = 0.8
lw = 0.5

# Tau Cet
fig_tc, axes_tc = plt.subplots(1, 1)
divider_tc = make_axes_locatable(axes_tc)
res_ax_tc = divider_tc.append_axes("bottom", size="35%", pad=0.1)
axes_tc.figure.add_axes(res_ax_tc, sharex=axes_tc)

axes_tc.errorbar(spec_rvcor_r[tau_cet_i,0], spec_rvcor_r[tau_cet_i,1], 
             spec_rvcor_r[tau_cet_i,2], alpha=alpha, label="Tau Cet (data)",
             linewidth=lw)

axes_tc.plot(ref_spec_norm[1,0], ref_spec_norm[1,1], "--", alpha=alpha,
         label="Tau Cet (synth)",linewidth=lw)

axes_tc.plot(tc_data["wave"], 
             tc_data["spectrum"]/np.nanmedian(tc_data["spectrum"]), 
             alpha=alpha,label="Tau Cet (Unfluxed)",linewidth=lw) 

axes_tc.text(np.nanmean(spec_rvcor_r[tau_cet_i,0]), 0.5, 
                        r"%0.2f$\pm$%0.2f$\,$km$\,$s$^{-1}$" % (rv_tc,e_rv_tc),
                        horizontalalignment="center")

#res_ax_tc.plot(spec_rvcor_r[tau_cet_i,0][10:-10], nres_tc, ".",
#               markersize=1)

res_ax_tc.set_ylim([-25,25])

axes_tc.legend(loc="best")
axes_tc.set_ylabel("Flux (Normalised)")
res_ax_tc.set_xlabel("Wavelength (A)")
res_ax_tc.set_ylabel("Residuals (Normalised)")

# HD260655
fig_hd, axes_hd = plt.subplots(1, 1)
divider_hd = make_axes_locatable(axes_hd)
res_ax_hd = divider_hd.append_axes("bottom", size="35%", pad=0.1)
axes_hd.figure.add_axes(res_ax_hd, sharex=axes_hd)

axes_hd.errorbar(spec_rvcor_r[hd260655_i,0], spec_rvcor_r[hd260655_i,1], 
             spec_rvcor_r[hd260655_i,2], alpha=alpha, label="HD260655 (data)",
             linewidth=lw)

axes_hd.plot(ref_spec_norm[0,0], ref_spec_norm[0,1], "--", alpha=alpha,
         label="HD260655 (synth)",linewidth=lw)

axes_hd.plot(hd_data["wave"], 
             hd_data["spectrum"]/np.nanmedian(hd_data["spectrum"]), 
             alpha=alpha,label="HD260655 (Unfluxed)",linewidth=lw) 

axes_hd.text(np.nanmean(spec_rvcor_r[hd260655_i,0]), 0.5, 
                        r"%0.2f$\pm$%0.2f$\,$km$\,$s$^{-1}$" % (rv_hd,e_rv_hd),
                        horizontalalignment="center")

#res_ax_hd.plot(spec_rvcor_r[hd260655_i,0][10:-10], nres_tc, ".",
#               markersize=1)

res_ax_hd.set_ylim([-25,25])

axes_hd.legend(loc="best")
axes_hd.set_ylabel("Flux (Normalised)")
res_ax_hd.set_xlabel("Wavelength (A)")
res_ax_hd.set_ylabel("Residuals (Normalised)")

plt.show()
