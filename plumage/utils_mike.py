"""Utilities functions to work with MIKE spectra.
"""
import os
from astropy.io import fits
from astropy.table import Table

def save_fits_from_dict(obs_dict, label, path="spectra"):
    """Save blue and red spectra, plus observational log table as a fits file 
    with the format:
        HDU 0: 1D blue arm echelle orders, [n_order]
        HDU 1: 3D blue arm wavelength scale, [n_star, n_order, n_px].
        HDU 2: 3D blue arm spectra [n_star, n_order, n_px].
        HDU 3: 3D blue arm uncertainties, [n_star, n_order, n_px].
        HDU 4: 1D red arm echelle orders, [n_order]
        HDU 5: 3D red arm wavelength scale, [n_star, n_order, n_px].
        HDU 6: 3D red arm spectra [n_star, n_order, n_px].
        HDU 7: 3D red arm uncertainties, [n_star, n_order, n_px].
        HDU 8: table of observational information
    
    File will be saved as <path>/mike_spectra_<label>.fits

    Parameters
    ----------
    obs_dict: dict
        Dictionary containing the compiled MIKE data, with the following keys:
            'obs_info' - pandas DataFrame with info from fits headers
            'orders_b' - blue echelle orders, shape [n_order]
            'wave_b'   - blue wavelength scales, shape [n_star, n_order, n_px]
            'spec_b'   - blue spectra, shape [n_star, n_order, n_px]
            'sigma_b'  - blue uncertainties, shape [n_star, n_order, n_px]
            'orders_r' - red echelle orders, shape [n_order]
            'wave_r'   - red wavelength scales, shape [n_star, n_order, n_px]
            'spec_r'   - red spectra, shape [n_star, n_order, n_px]
            'sigma_r'  - red uncertainties, shape [n_star, n_order, n_px]

    label: string
        Unique label for the resulting fits file, which will be saved as
        mike_spectra_<label>.fits.
    
    path: string
        Path to save the fits file to.
    """
    # Intialise HDU List
    hdu = fits.HDUList()

    # -------------------------------------------------------------------------
    # Blue Arm
    # -------------------------------------------------------------------------
    # HDU 1: Blue orders
    orders_b_img =  fits.PrimaryHDU(obs_dict["orders_b"])
    orders_b_img.header["EXTNAME"] = (
        "ORDERS_B",
        "Blue arm echelle order numbers of shape [n_order].",)
    hdu.append(orders_b_img)

    # HDU 2: Blue wavelength scale
    wave_b_img =  fits.PrimaryHDU(obs_dict["wave_b"])
    wave_b_img.header["EXTNAME"] = (
        "WAVE_3D_B",
        "Blue wavelength scale of shape [n_star, n_order, n_px].",)
    hdu.append(wave_b_img)

    # HDU 3: Blue band flux
    spec_b_img =  fits.PrimaryHDU(obs_dict["spec_b"])
    spec_b_img.header["EXTNAME"] = (
        "SPEC_3D_B",
        "Blue arm spectra scale of shape [n_star, n_order, n_px].",)
    hdu.append(spec_b_img)

    # HDU 4 Blue band flux uncertainty
    sigma_b_img =  fits.PrimaryHDU(obs_dict["sigma_b"])
    sigma_b_img.header["EXTNAME"] = (
        "SIGMA_3D_B",
        "Blue arm uncertainties of shape [n_star, n_order, n_px].",)
    hdu.append(sigma_b_img)

    # -------------------------------------------------------------------------
    # Red Arm
    # -------------------------------------------------------------------------
    # HDU 5: Blue orders
    orders_r_img =  fits.PrimaryHDU(obs_dict["orders_r"])
    orders_r_img.header["EXTNAME"] = (
        "ORDERS_B",
        "Red arm echelle order numbers of shape [n_order].",)
    hdu.append(orders_r_img)

    # HDU 6: Blue wavelength scale
    wave_r_img =  fits.PrimaryHDU(obs_dict["wave_r"])
    wave_r_img.header["EXTNAME"] = (
        "WAVE_3D_R",
        "Red wavelength scale of shape [n_star, n_order, n_px].",)
    hdu.append(wave_r_img)

    # HDU 7: Blue band flux
    spec_r_img =  fits.PrimaryHDU(obs_dict["spec_r"])
    spec_r_img.header["EXTNAME"] = (
        "SPEC_3D_R",
        "Red arm spectra scale of shape [n_star, n_order, n_px].",)
    hdu.append(spec_r_img)

    # HDU 8: Blue band flux uncertainty
    sigma_r_img =  fits.PrimaryHDU(obs_dict["sigma_r"])
    sigma_r_img.header["EXTNAME"] = (
        "SIGMA_3D_R",
        "Red arm uncertainties of shape [n_star, n_order, n_px].",)
    hdu.append(sigma_r_img)

    # -------------------------------------------------------------------------
    # Table
    # -------------------------------------------------------------------------
    # HDU 9: table of observational information
    obs_tab = fits.BinTableHDU(Table.from_pandas(
        obs_dict["obs_info"].reset_index()))
    obs_tab.header["EXTNAME"] = (
        "OBS_TAB",
        "Observation info table, of length [n_star].")
    hdu.append(obs_tab)
    
    # Done, save
    save_path = os.path.join(path, "mike_spectra_{}.fits".format(label))
    hdu.writeto(save_path, overwrite=True)