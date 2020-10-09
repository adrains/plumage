"""
"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import plumage.synthetic as synth
import matplotlib.pyplot as plt

root = "/home/bessell/"

band_settings_ngsl = {
    "inst_res_pow":1000,
    "wl_min":3060,
    "wl_max":10100,
    "n_px":1920,
    "wl_per_px":2.95,
    "wl_broadening":2.95,
    "grid":"full",
}

star_dict = {
    "Gl 109":("SJM/renorm_h_stis_ngsl_gl109_v2.fits", 3405, 4.85, -0.1, "NGSL"),    # RA12
    "Gl 412 A":("SJM/renorm_h_stis_ngsl_bd442051_v2.fits", 3684, 4.7, -0.4, "NGSL"),    #RA12
    "GJ 825":("SJM/renorm_h_stis_ngsl_gj825_v2.fits", 3776, 4.55, -0.62, "NGSL"),   # G14, misc
    "HD 33793":("SJM/renorm_h_stis_ngsl_hd033793_v2.fits", 3722, 4.71, -0.84, "NGSL"),   # misc
    "HD 217357":("SJM/renorm_h_stis_ngsl_hd217357_v2.fits", 4225, 4.6, -0.11, "NGSL"), # G14, misc logg
    "HD 201092":("SJM/renorm_h_stis_ngsl_hr8086_v2.fits", 4045, 4.53, -0.38, "NGSL"), # G14
    "HD 201091":("SJM/renorm_h_stis_ngsl_hd201091_v2.fits", 4481, 4.67, -0.14, "NGSL"), #L14

    "GJ 205":("Mdwarfs_mar2012/m15v_gj205allsc.fits", 3801, 4.71, 0.49, "SPEX"), #M15
}

idl = synth.idl_init()

sid = "Gl 109"   # good
sid = "Gl 412 A"   # good
sid = "GJ 825"
sid = "HD 33793"
sid = "HD 217357"
sid = "HD 201092"   # good
sid = "HD 201091"
sid = "GJ 205"

#------------------------------------------------------------------------------    
# NGSL
#------------------------------------------------------------------------------
if star_dict[sid][4] == "NGSL":
    wl_mins = [3060, 5649,]
    wl_maxes = [5649, 10100]
    delta_wl = [2.744, 4.878]
    npix = [944, 932]

    wl_synth = []
    spec_synth = []

    for wlmin, wlmax, dw, npx in zip(wl_mins, wl_maxes, delta_wl, npix):
        ww, ss = synth.get_idl_spectrum(
            idl, 
            star_dict[sid][1], 
            star_dict[sid][2], 
            star_dict[sid][3], 
            wlmin, 
            wlmax, 
            ipres=band_settings_ngsl["inst_res_pow"],
            grid=band_settings_ngsl["grid"],
            resolution=None,
            norm="abs",
            do_resample=True, 
            wl_per_px=dw,
            rv_bcor=0,
            )
        wl_synth.append(ww)
        spec_synth.append(ss)

    wl_synth = np.concatenate(wl_synth)
    spec_synth = np.concatenate(spec_synth)

    path = os.path.join(root, star_dict[sid][0])

    ff = fits.open(path)

    wl = ff[1].data["WAVELENGTH"]
    flux = ff[1].data["FLUX"]

    med_mask_obs = np.logical_and(wl > 6000, wl < 10000)
    med_mask_synth = np.logical_and(wl_synth > 6000, wl_synth < 10000)

    xlim = (3000, 10100)

#------------------------------------------------------------------------------    
# Spex
#------------------------------------------------------------------------------
elif star_dict[sid][4] == "SPEX":
    path = os.path.join(root, star_dict[sid][0])
    ff = fits.open(path)

    flux = ff[0].data
    wl = np.linspace(3400, 54200, len(flux))
    
    wl_px = (54200-3400)/len(flux)

    wl_synth, spec_synth = synth.get_idl_spectrum(
            idl, 
            star_dict[sid][1], 
            star_dict[sid][2], 
            star_dict[sid][3], 
            3400, 
            54200, 
            ipres=2000,
            grid=band_settings_ngsl["grid"],
            resolution=None,
            norm="abs",
            do_resample=True, 
            wl_per_px=wl_px,
            rv_bcor=0,
            )

    wl_synth *= 1.013

    xlim = (3400, 24000)

    med_mask_obs = np.logical_and(wl > 6000, wl < 10000)
    med_mask_synth = np.logical_and(wl_synth > 6000, wl_synth < 10000)

plt.close("all")
plt.plot(wl, flux/np.median(flux[med_mask_obs]), linewidth=0.3, label=sid)
plt.plot(wl_synth, spec_synth/np.median(spec_synth[med_mask_synth]), linewidth=0.3, label="synth")
plt.legend(loc="best")

filters = ["v", "g", "r", "i", "z", "BP", "RP", "J", "H", "K"]

for filt_i, filt in enumerate(filters):
    wl_f, fp = synth.load_filter_profile(filt, 3000, 10100, do_zero_pad=True)
    plt.plot(wl_f, fp, "--", linewidth=1.0)

plt.xlabel("Wavelength (A)")
title = "{} (Teff~{:0.0f} K, logg~{:0.2f}, [Fe/H]~{:0.2f})".format(
    sid, star_dict[sid][1], star_dict[sid][2], star_dict[sid][3])
plt.title(title)
plt.xlim(xlim)
plt.gcf().set_size_inches(16, 8)
plt.savefig("plots/bessell_{}_{}.pdf".format(star_dict[sid][1], sid.replace(" ","")))

