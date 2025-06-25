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
        "ORDERS_R",
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


def load_3D_spec_from_fits(label, arm="r", path="spectra"):
    """TODO 
    
    File will be loaded from {path}/spectra_{label}.fits

    Parameters
    ----------
    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    path: string
        Path to save the fits file to. Defaults to spectra/

    Returns 
    -------
    spectra_b: float array
        3D numpy array containing blue arm spectra of form 
        [N_ob, wl/spec/sigma, value].

    spectra_r: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, value].
    
    observations: pandas dataframe
        Dataframe containing information about each observation.
    """
    # Load in the fits file
    fits_path = os.path.join(path,  "mike_spectra_{}.fits".format(label))

    wave_hdu = "WAVE_3D_{}".format(arm.upper())
    spec_hdu = "SPEC_3D_{}".format(arm.upper())
    sigma_hdu = "SIGMA_3D_{}".format(arm.upper())
    order_hdu = "ORDERS_{}".format(arm.upper())
    
    with fits.open(fits_path) as fits_file:
        wave = fits_file[wave_hdu].data
        spec = fits_file[spec_hdu].data
        sigma = fits_file[sigma_hdu].data
        orders = fits_file[order_hdu].data

    return wave, spec, sigma, orders


def load_fits_table(
    extension,
    label,
    path="spectra",
    ext_label="",
    set_index_col=False,):
    """Loads in the data from specified fits table HDU.

    Parameters
    ----------
    extension: string
        Which fits table extension to save. Currently either 'OBS_TAB' or 
        'TRANSIT_FITS'

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    path: string
        Path to save the fits file to.

    ext_label: str, default: ''
        Sublabel of the extension. At the moment only applicable to the 
        'CANNON_MODEL' extension to note the specifics of the model.

    Returns
    -------
    obs_pd: pandas dataframe
        Dataframe containing information about each observation.
    """
    # List of valid extensions
    valid_ext = ["OBS_TAB", "CANNON_INFO", "CANNON_MODEL"]

    # Needed to reapply the DataFrame index, which astropy does not respect
    ext_index = {
        "OBS_TAB":"source_id",
        "CANNON_INFO":"source_id_dr3",
        "CANNON_MODEL":"source_id_dr3",}

    # Input checking
    if extension not in valid_ext:
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext))

    # Write out the full extention name
    if ext_label != "":
        ext_full = "{}_{}".format(extension, ext_label)
    else:
        ext_full = extension

    # Load in the fits file
    fits_path = os.path.join(path, "mike_spectra_{}.fits".format(label))

    with fits.open(fits_path, mode="readonly") as fits_file:
        if ext_full in fits_file:
            obs_tab = Table(fits_file[ext_full].data)
            obs_df = obs_tab.to_pandas()
        else:
            raise Exception("No table of that extension or wrong fits format")

    if set_index_col:
        obs_df.set_index(ext_index[extension], inplace=True)
    else:
        obs_df.set_index("index", inplace=True)

    return obs_df


def convert_air_to_vacuum_wl(wavelengths_air,):
    """Converts provided air wavelengths to vacuum wavelengths using the 
    formalism described here: 
        https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Parameters
    ----------
    wavelengths_air: float array
        Array of air wavelength values to convert.

    Returns
    -------
    wavelengths_vac: float array
        Corresponding array of vacuum wavelengths
    """
    # Calculate the refractive index for every wavelength
    ss = (10**4 / wavelengths_air)**2
    n_ref = (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - ss)
         + 0.0001599740894897 / (38.92568793293 - ss))

    # Calculate vacuum wavelengths
    wavelengths_vac = wavelengths_air * n_ref

    return wavelengths_vac


def convert_vacuum_to_air_wl(wavelengths_vac,):
    """Converts provided vacuum wavelengths to air wavelengths using the 
    formalism described here: 
        https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Parameters
    ----------
    wavelengths_vac: float array
        Array of vacuum wavelength values to convert.

    Returns
    -------
    wavelengths_air: float array
        Corresponding array of air wavelengths
    """
    # Calculate the refractive index for every wavelength
    ss = (10**4 / wavelengths_vac)**2
    n_ref = (1 + 0.0000834254 + 0.02406147 / (130 - ss)
         + 0.00015998 / (38.9 - ss))

    # Calculate vacuum wavelengths
    wavelengths_air = wavelengths_vac / n_ref

    return wavelengths_air