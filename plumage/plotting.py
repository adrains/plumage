"""
"""
from __future__ import print_function, division
import numpy as np
import glob
import matplotlib.pylab as plt

def plot_nightly_spectra(night_a="20190827", compare_spectra=False, 
                         night_b="20190827", plot_step_id="10", 
                         snr_step_id="08", 
                         base_path="/priv/mulga2/arains/ys/wifes/reduced"):
    """Plots red and blue band spectra for each night, stacked and offset.
    """
    # Import 
    path_plt = "%s/%s/ascii/*_%s_*"  % (base_path, night_a, plot_step_id)
    path_snr = "%s/%s/ascii/*_%s_*"  % (base_path, night_a, snr_step_id)
    files_plt = glob.glob(path_plt)
    files_snr = glob.glob(path_snr)

    files_plt.sort()
    files_snr.sort()

    assert len(files_plt) == len(files_snr)
    
    # Abort if still no files
    if len(files_plt) == 0:
        print("No fits files, aborting.")
        return
    
    plt.close("all")
    fig, axes = plt.subplots(1, 2, sharey=True)
    bands = ["b", "r"]

    snr_sort_i = None

    # Plot bands on separate axes
    for band, axis in zip(bands, axes):
        fplt_band = np.array([fp for fp in files_plt 
                              if "_%s.dat" % band in fp])
        fsnr_band = np.array([fs for fs in files_snr 
                              if "_%s.dat" % band in fs])

        # Sort by SNR
        snrs = []
        for fsnr in fsnr_band:
            sp_snr = np.loadtxt(fsnr)
            snrs.append(np.median(sp_snr[:,1])/np.sqrt(np.median(sp_snr[:,1])))

        snrs = np.array(snrs)

        # Sort only once (with order from a single band)
        if snr_sort_i is None:
            snr_sort_i = np.argsort(snrs)
            print("Sorting")

        fplt_band = fplt_band[snr_sort_i]
        fsnr_band = fsnr_band[snr_sort_i]
        snrs = snrs[snr_sort_i]
    
        # Now plot
        for sp_i, (fplt, fsnr, snr) in enumerate(zip(fplt_band, fsnr_band, 
                                                     snrs)):
            sp_plt = np.loadtxt(fplt)
            sp_snr = np.loadtxt(fsnr)

            axis.plot(sp_plt[:,0], sp_plt[:,1]/np.median(sp_plt[:,1])+2*sp_i, 
                    linewidth=0.1)

            # Plot label
            label = "%s [%i]" % (fplt.split("/")[-1][9:-4], snr)
            print(sp_i, label)
            axis.text(sp_plt[:,0].mean(), 2*sp_i, label, fontsize="x-small", 
                      ha="center")
    
    axis.set_ylim([-1,sp_i*2+4])
    fig.suptitle(night_a)
    fig.tight_layout()
    fig.text(0.5, 0.04, "Wavelength (A)", ha='center')
    fig.text(0.04, 0.5, "Flux (scaled)", va='center', rotation='vertical')
    plt.gcf().set_size_inches(9, 16)
    plt.savefig("/home/arains/code/plumage/plots/spectra_%s.pdf" % night_a)

def merge_spectra_pdfs():
    """Merge diagnostic pdfs together for easy checking.
    
    Code from:
    https://stackoverflow.com/questions/3444645/merge-pdf-files
    """
    from PyPDF2 import PdfFileMerger

    pdfs = glob.glob("/home/arains/code/plumage/plots/spectra_br_2019*")
    pdfs.sort()
    
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    fn = "/home/arains/code/plumage/plots/spectra_summary.pdf"

    with open(fn, 'wb') as fout:
        merger.write(fout)