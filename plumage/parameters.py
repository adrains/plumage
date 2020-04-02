"""
"""
import numpy as np
import pandas as pd
import plumage.spectra as spec
from numpy.polynomial.polynomial import polyval as polyval

def compare_to_standard_spectrum(spec_sci, spec_std):
    """
    """
    # Get a wavelength mask
    wl_mask = spec.make_wavelength_mask(spec_sci[0,:], True,
                                        mask_sky_emission=True)

    # Get the chi^2 of the fit
    chi2 = np.nansum(((spec_sci[1,:][wl_mask] - spec_std[1,:][wl_mask]) 
                        / spec_std[2,:][wl_mask])**2)

    return chi2


def compare_sci_to_all_standards(spec_sci, spec_stds, std_params):
    """
    """
    chi2_all = np.ones(len(spec_stds))

    for std_i, spec_std in enumerate(spec_stds):
        chi2 = compare_to_standard_spectrum(spec_sci, spec_std)
        chi2_all[std_i] = chi2

    sorted_std_idx = np.argsort(chi2_all)

    print(chi2_all[sorted_std_idx][:3])
    print(std_params.iloc[sorted_std_idx][:3])

    #wl_mask = spec.make_wavelength_mask(spec_sci[0,:], True,
    #                                    mask_sky_emission=True)

    #plt.plot(spec_sci[0,:][wl_mask], spec_sci[1,:][wl_mask], label="Science")
    #plt.plot(spec_stds[0,:][wl_mask], spec_sci[1,:][wl_mask], label="Science")
    

def compute_mann_2019_masses():
    """
    """
    coeff = np.array([

    ])

def compute_mann_2015_teff(
    colour,
    j_h,
    feh=None,
    relation="BP - RP, J - H",
    teff_file="data/mann_2015_teff.txt", 
    r_file="data/mann_2015_teff.txt",
    sigma_spec=60,
    ):
    """
    """
    supported_relations = ("BP - RP, J - H")
    if relation not in supported_relations:
        raise ValueError("Unsupported relation. Must be one of %s"
                         % supported_relations)

    m15_teff = pd.read_csv(teff_file, delimiter="\t", comment="#", index_col="X")

    x_coeff = m15_teff.loc[relation][["a", "b", "c", "d", "e"]]

    jh_coeff = m15_teff.loc[relation][["f", "g"]]

    teffs = (polyval(colour, x_coeff) + polyval(j_h, jh_coeff)) * 3500
    e_teffs = np.ones_like(teffs) * np.sqrt(m15_teff.loc[relation]["sigma"]**2+sigma_spec**2)

    return teffs, e_teffs