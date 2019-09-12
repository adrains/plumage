"""
"""
from __future__ import print_function, division
import numpy as np
import glob
import matplotlib.pylab as plt

def plot_nightly_spectra(night_a="20190827", compare_spectra=False, 
                         night_b="20190827"):
    """
    """
    # Two ways files are stored - red and blue separately, or together. Try 
    # both
    path_a = "/priv/mulga1/marusa/2m3reduced/wifes/%s/ascii/*" % night_a
    files_a = glob.glob(path_a)
    
    
    
    # if this is empty, try another format
    if len(files_a) == 0:
        path_a_base = "/priv/mulga1/marusa/2m3reduced/wifes/%s" % night_a
    
        files_a_r = glob.glob("%s/reduced_r*/ascii/*" % path_a_base)
        files_a_b = glob.glob("%s/reduced_b*/ascii/*" % path_a_base)
    
        files_a = files_a_r + files_a_b
    
    # Abort if still no files
    if len(files_a) == 0:
        print("No fits files, aborting.")
        return
    
    files_a.sort()
    
    if compare_spectra:
        path_b = "/priv/mulga2/arains/ys/wifes/reduced/%s/ascii/*" % night_b
        files_b = glob.glob(path_b)
        files_b.sort()
    else:
        files_b = np.zeros_like(files_a)
    
    
    
    plt.close("all")
    plt.figure()
    
    for sp_i, (fa, fb) in enumerate(zip(files_a, files_b)):
        sp_a = np.loadtxt(fa)
        plt.plot(sp_a[:,0], sp_a[:,1]/np.median(sp_a[:,1])+2*sp_i, linewidth=0.1)
        
        snr = np.median(sp_a[:,1]) / np.sqrt(np.median(sp_a[:,1]))
        
        if compare_spectra:
            sp_b = np.loadtxt(fb)
            plt.plot(sp_b[:,0], sp_b[:,1]/np.median(sp_b[:,1])+2*sp_i, "--", 
                     linewidth=0.1)
        
        # Plot label
        label = "%s [%i]" % (fa.split("/")[-1][9:-4], snr)
        print(sp_i, label)
        plt.text(sp_a[:,0].mean(), 2*sp_i, label, fontsize="x-small", ha="center")
    
    plt.ylim([-1,sp_i*2+2])
    plt.title(night_a)
    plt.tight_layout()
    plt.ylabel("Flux (scaled)")
    plt.xlabel("Wavelength (A)")
    plt.gcf().set_size_inches(9, 16)
    plt.savefig("/home/arains/code/plumage/plots/spectra_%s.pdf" % night_a)

def merge_spectra_pdfs():
    """Merge diagnostic pdfs together for easy checking.
    
    Code from:
    https://stackoverflow.com/questions/3444645/merge-pdf-files
    """
    from PyPDF2 import PdfFileMerger

    pdfs = glob.glob("/home/arains/code/plumage/plots/spectra_2019*")
    pdfs.sort()
    
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    with open("/home/arains/code/plumage/plots/spectra_summary.pdf", 'wb') as fout:
        merger.write(fout)