"""Script to investigate the influence of changing abundances on synthetic
spectra of cool stars.
"""
import numpy as np
import matplotlib.pyplot as plt
import plumage.synthetic as synth
import matplotlib.ticker as plticker
import plumage.plotting as pplt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# These are the elements Thomas's special grid supports, passed in as the 
# 'alpha' abundance.
#elements = ["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "Fe"]
#element_is = [6, 7, 8, 11, 12, 13, 14, 20, 22, 26]

do_dominant_absorbers = True

# First set of elements are the dominant ones
if do_dominant_absorbers:
    elements = ["C", "O", "Ti",]
    element_is = [6, 8, 22,]
    do_sign_change = True
    save_label = "dominant_absorbers"

# And everything else
else:
    elements = ["N", "Na", "Mg", "Al", "Si", "Ca", "Fe"]
    element_is = [7, 11, 12, 13, 14, 20, 26]
    do_sign_change = False
    save_label = "minor_absorbers"

# Settings
wl_min = 3500
wl_max = 7000

# These are the values of [X/H] that we want to plot
xihs = [-0.1, 0, 0.1] 

# Teff/gravity pairs
params = (
    (3000,5.0),
    (3500,4.75),
    (4000,4.65),)

# Initialise IDL
idl = synth.idl_init(drive="priv")

# Now loop for every Teff-logg pair
for (teff,logg) in params:
    print("Running on Teff = {}, logg = {}".format(teff, logg))
    # First we want to obtain our reference spectra for just changing the bulk 
    # metallicity by +/-0.1 dex
    ref_spectra = []

    for feh in xihs:
        wave, spec = synth.get_idl_spectrum(
            idl,
            teff=teff,
            logg=logg,
            feh=feh,
            abund_alpha=None,
            abund=None,
            wl_min=wl_min,
            wl_max=wl_max,
            ipres=7000,
            do_resample=True,
            wave_pad=100,
            grid="full",
            wl_per_px=0.44,
            norm="norm",)

        ref_spectra.append(spec)

    # Setup plot
    plt.close("all")
    fig, ax = plt.subplots()

    # Plot residuals
    divider = make_axes_locatable(ax)
    resid_ax = divider.append_axes("bottom", size="100%", pad=0)
    ax.figure.add_axes(resid_ax, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    resid_ax.hlines(
        0,
        wl_min,
        wl_max,
        linestyles="dashed",
        linewidth=0.1,
        color="black")

    # Plot the base spectrum
    ax.plot(
        wave, ref_spectra[1], linewidth=0.2, label="[Fe/H] = 0", color="black")

    # And plot the percentage change in flux as:
    # F(solar-0.1) - F(solar+0.1) / F(solar+0.1)
    frac_change = (ref_spectra[0] - ref_spectra[2]) / ref_spectra[2]
    resid_ax.plot(wave, frac_change, linewidth=0.2, label="[M/H]",)

    # Now we want to get spectra for each of the elements
    abund_spectra_all = {}

    for (element, element_i) in zip(elements, element_is):
        print("\t...{}".format(element))
        element_spectra = []

        for xih in xihs:
            _, spec = synth.get_idl_spectrum(
                idl,
                teff=teff,
                logg=logg,
                feh=0.0,
                abund_alpha=element_i,
                abund=xih,
                wl_min=wl_min,
                wl_max=wl_max,
                ipres=7000,
                do_resample=True,
                wave_pad=100,
                grid="m_dwarf",
                wl_per_px=0.44,
                norm="norm",)

            element_spectra.append(spec)
        
        abund_spectra_all[element_i] = element_spectra

        # Plot fractional change in flux
        frac_change = \
            (element_spectra[0] - element_spectra[2]) / element_spectra[2]

        # Plot, but check whether this is a positive or negative trend
        if do_sign_change and frac_change[3650] < 0:
            resid_ax.plot(wave, -1*frac_change, linewidth=0.2, 
                label=r"$-1\times $ {}".format(element),)

        else:
            resid_ax.plot(wave, frac_change, linewidth=0.2, label=element,)
    
    # Finish setting up plot
    ax.set_ylabel("Flux (Cont. Norm.)")
    resid_ax.set_xlabel("Wavelength (A)")
    resid_ax.set_ylabel(
        r"$\frac{{\rm F(subsolar)}-{\rm F(supersolar)}}{{\rm F(supersolar)}}$")

    resid_ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=100))
    resid_ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
    resid_ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    resid_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.2))

    # Put a legend below current axis
    leg = resid_ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
            fancybox=True, ncol=len(elements)+1,)

    # Update width of legend objects
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.5)

    ax.set_title(
        r"$T_{{\rm eff}}={:0.0f}\,$K, $\log g={:0.2f}$, [Fe/H]$ = 0.0$".format(
        teff, logg))
    ax.set_xlim([wl_min,wl_max])
    resid_ax.set_xlim([wl_min,wl_max])
    plt.gcf().set_size_inches(16, 4)
    plt.tight_layout()
    plt.savefig("paper/synth_spec_flux_vs_abund_teff_{}.pdf".format(teff))
    plt.savefig(
        "paper/synth_spec_flux_vs_abund_teff_{}.png".format(teff), dpi=500)

pplt.merge_spectra_pdfs(
    "paper/synth_spec_flux_vs_abund_teff*pdf",
    "paper/synth_spec_flux_vs_abund_{}.pdf".format(save_label))