"""Utilities functions to work with MIKE spectra.
"""
import os
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table

def save_fits_from_dict(obs_dict, label, path="spectra"):
    """Save blue and red spectra, plus observational log table as a fits file 
    with the format:
        HDU 0: 1D blue arm echelle orders, [n_order]
        HDU 1: 1D blue arm order spectral dispersion, [n_order]
        HDU 2: 3D blue arm wavelength scale, [n_star, n_order, n_px].
        HDU 3: 3D blue arm spectra [n_star, n_order, n_px].
        HDU 4: 3D blue arm uncertainties, [n_star, n_order, n_px].
        HDU 5: 1D red arm echelle orders, [n_order]
        HDU 6: 1D red arm order spectral dispersion, [n_order]
        HDU 7: 3D red arm wavelength scale, [n_star, n_order, n_px].
        HDU 8: 3D red arm spectra [n_star, n_order, n_px].
        HDU 9: 3D red arm uncertainties, [n_star, n_order, n_px].
        HDU 10: table of observational information
    
    File will be saved as <path>/mike_spectra_<label>.fits

    Parameters
    ----------
    obs_dict: dict
        Dictionary containing the compiled MIKE data, with the following keys:
            'obs_info' - pandas DataFrame with info from fits headers
            'orders_b' - blue echelle orders, shape [n_order]
            'disp_b'   - blue order dispersion, shape [n_order]
            'wave_b'   - blue wavelength scales, shape [n_star, n_order, n_px]
            'spec_b'   - blue spectra, shape [n_star, n_order, n_px]
            'sigma_b'  - blue uncertainties, shape [n_star, n_order, n_px]
            'orders_r' - red echelle orders, shape [n_order]
            'disp_r'   - red order dispersion, shape [n_order]
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

    # HDU 2: Blue orders
    disp_b_img =  fits.PrimaryHDU(obs_dict["disp_b"])
    disp_b_img.header["EXTNAME"] = (
        "DISP_B",
        "Blue order dispersion of shape [n_order].",)
    hdu.append(disp_b_img)

    # HDU 3: Blue wavelength scale
    wave_b_img =  fits.PrimaryHDU(obs_dict["wave_b"])
    wave_b_img.header["EXTNAME"] = (
        "WAVE_3D_B",
        "Blue wavelength scale of shape [n_star, n_order, n_px].",)
    hdu.append(wave_b_img)

    # HDU 4: Blue band flux
    spec_b_img =  fits.PrimaryHDU(obs_dict["spec_b"])
    spec_b_img.header["EXTNAME"] = (
        "SPEC_3D_B",
        "Blue arm spectra scale of shape [n_star, n_order, n_px].",)
    hdu.append(spec_b_img)

    # HDU 5 Blue band flux uncertainty
    sigma_b_img =  fits.PrimaryHDU(obs_dict["sigma_b"])
    sigma_b_img.header["EXTNAME"] = (
        "SIGMA_3D_B",
        "Blue arm uncertainties of shape [n_star, n_order, n_px].",)
    hdu.append(sigma_b_img)

    # -------------------------------------------------------------------------
    # Red Arm
    # -------------------------------------------------------------------------
    # HDU 6: Blue orders
    orders_r_img =  fits.PrimaryHDU(obs_dict["orders_r"])
    orders_r_img.header["EXTNAME"] = (
        "ORDERS_R",
        "Red arm echelle order numbers of shape [n_order].",)
    hdu.append(orders_r_img)

    # HDU 7: red dispersion
    disp_r_img =  fits.PrimaryHDU(obs_dict["disp_r"])
    disp_r_img.header["EXTNAME"] = (
        "DISP_R",
        "Red order dispersion of shape [n_order].",)
    hdu.append(disp_r_img)

    # HDU 8: Blue wavelength scale
    wave_r_img =  fits.PrimaryHDU(obs_dict["wave_r"])
    wave_r_img.header["EXTNAME"] = (
        "WAVE_3D_R",
        "Red wavelength scale of shape [n_star, n_order, n_px].",)
    hdu.append(wave_r_img)

    # HDU 9: Blue band flux
    spec_r_img =  fits.PrimaryHDU(obs_dict["spec_r"])
    spec_r_img.header["EXTNAME"] = (
        "SPEC_3D_R",
        "Red arm spectra scale of shape [n_star, n_order, n_px].",)
    hdu.append(spec_r_img)

    # HDU 10: Blue band flux uncertainty
    sigma_r_img =  fits.PrimaryHDU(obs_dict["sigma_r"])
    sigma_r_img.header["EXTNAME"] = (
        "SIGMA_3D_R",
        "Red arm uncertainties of shape [n_star, n_order, n_px].",)
    hdu.append(sigma_r_img)

    # -------------------------------------------------------------------------
    # Table
    # -------------------------------------------------------------------------
    # HDU 11: table of observational information
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
    """Reads in the wavelength scale, fluxes, uncertainties, order numbering, 
    and order dispersions associated with a given MIKE arm.
    
    File will be loaded from {path}/mike_spectra_{label}.fits

    Parameters
    ----------
    label: string
        Unique label for the fits file.
    
    arm: str, default: 'r'
        Spectrograph arm to import data for, either 'b' or 'r'.
        
    path: string, default: 'spectra'
        Path to load the fits file from.

    Returns 
    -------
    wave, spec, sigma: 3D float array
        3D float arrays of shape [n_obs, n_order, n_px] corresponding to the
        wavelength scale, spectra, or sigmas of the given spectral arm.

    orders: 1D float array
        Order numbering corresponding to this arm, of shape [n_order].
    
    disp: 2D float array
        Order dispersion for this arm, of shape [n_obs, n_order].
    """
    # Load in the fits file
    fits_path = os.path.join(path,  "mike_spectra_{}.fits".format(label))

    wave_hdu = "WAVE_3D_{}".format(arm.upper())
    spec_hdu = "SPEC_3D_{}".format(arm.upper())
    sigma_hdu = "SIGMA_3D_{}".format(arm.upper())
    order_hdu = "ORDERS_{}".format(arm.upper())
    disp_hdu = "DISP_{}".format(arm.upper())
    
    with fits.open(fits_path) as fits_file:
        wave = fits_file[wave_hdu].data
        spec = fits_file[spec_hdu].data
        sigma = fits_file[sigma_hdu].data
        orders = fits_file[order_hdu].data
        disp = fits_file[disp_hdu].data

    return wave, spec, sigma, orders, disp


def load_fits_table(
    extension,
    label,
    path="spectra",
    ext_label="",
    set_index_col=False,):
    """Loads in the data from specified fits table HDU.

    File will be loaded from {path}/mike_spectra_{label}.fits

    Parameters
    ----------
    extension: string
        Which fits table extension to load. Currently either 'OBS_TAB',
        'CANNON_INFO', or 'CANNON_MODEL'.

    label: string
        Unique label for the fits file.
    
    path: string
        Path to save the fits file to.

    ext_label: str, default: ''
        Sublabel of the extension. At the moment only applicable to the 
        'CANNON_MODEL' extension to note the specifics of the model.

    set_index_col: boolean, default: False
        Whether to reset the index column for the dataframe.
        
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


def save_flux_calibration_poly_coeff(
    poly_order,
    poly_coeff,
    wave_mins,
    wave_maxes,
    orders,
    arm,
    save_path,
    label,):
    """Function to save flux calibration coefficients to disk. The data file
    has 1+ n_coeff columns, where the first column is the integer order number,
    and the remainder of the columns are float polynomial coefficients. The 
    file has a header row of columns names, and has filename format:
    'flux_cal_coeff_<arm>_<label>_<poly_order>.tsv'.

    Parameters
    ----------
    poly_order: int
        Polynomial order to be fit as the transfer function for each order.

    poly_coeff: 2D float array
        Polynomial coefficient array, of shape [n_order, poly_order].

    wave_mins, wave_maxes: 1D float array
        Minimum and maximum wavelength values of each order respectively, of
        shape [n_order]. To avoid poorly conditioned polynomial fits, we
        rescaled the wavelength scale to [-1, 1] using the minimum and maximum
        wavelengths of each order. This is equaivalent to the window and domain
        parameters used in numpy's Polynomial module. 

    orders: 1D int array
        Echelle order numbers, of shape [n_order].

    arm: str
        String identifying the spectrograph arm.

    save_path: str
        File path to save coeffients to.

    label: str
        Label for the file used to identify the flux standard, e.g.
        "<night>_<target_id>".
    """
    # Construct filename
    fn = "flux_cal_coeff_{}_{}_{}.txt".format(arm, label, poly_order)
    path = os.path.join(save_path, fn)
    
    # Concatenate together the orders, wave min/max and polynomial coefficients
    data = np.hstack(
        (orders[:,None], wave_mins[:,None], wave_maxes[:,None], poly_coeff))

    # Construct header and format array
    coef_cols = ["coeff_{}".format(pc) for pc in range(poly_order)]
    header = ["order", "domain_min", "domain_max"] + coef_cols
    header = "\t".join(header)
 
    fmt = ["%0.0f", "%0.8f", "%0.8f"] + ["%.18e" for pc in range(poly_order)]

    # Save
    np.savetxt(path, data, fmt=fmt, delimiter="\t", header=header,)


def load_flux_calibration_poly_coeff(
    poly_order,
    arm,
    save_path,
    label,):
    """Function to load flux calibration coefficients from disk. The data file
    has 1+ n_coeff columns, where the first column is the integer order number,
    and the remainder of the columns are float polynomial coefficients. The 
    file has a header row of columns names, and has filename format:
    'flux_cal_coeff_<arm>_<label>_<poly_order>.tsv'.

    Parameters
    ----------
    poly_order: int
        Polynomial order to be fit as the transfer function for each order.

    arm: str
        String identifying the spectrograph arm.

    save_path: str
        File path to save coeffients to.

    label: str
        Label for the file used to identify the flux standard, e.g.
        "<night>_<target_id>".

    Returns
    -------
    orders: 1D int array
        Echelle order numbers, of shape [n_order].
    
    wave_mins, wave_maxes: 1D float array
        Minimum and maximum wavelength values of each order respectively, of
        shape [n_order]. To avoid poorly conditioned polynomial fits, we
        rescaled the wavelength scale to [-1, 1] using the minimum and maximum
        wavelengths of each order. This is equaivalent to the window and domain
        parameters used in numpy's Polynomial module. 

    poly_coeff: 2D float array
        Polynomial coefficient array, of shape [n_order, poly_order].
    """
    # Construct filename
    fn = "flux_cal_coeff_{}_{}_{}.txt".format(arm, label, poly_order)
    path = os.path.join(save_path, fn)
    
    # Load
    data = np.loadtxt(path, delimiter="\t", comments="#",)

    # Unpack
    orders = data[:,0].astype(int)
    wave_mins = data[:,1]
    wave_maxes = data[:,2]
    poly_coeff = data[:,3:]

    # Return
    return orders, wave_mins, wave_maxes, poly_coeff


def output_snr_for_spreadsheet(source_ids, obs_dict,):
    """Extracts exposure times and SNR values for the blue and red arms, and
    formats such that this can be directly copied and pasted into the MIKE
    target spreadsheet (assuming, of course, that source_ids is in order).

    Prints one row per star, formatted and tab separated as:
        <source_id> <exps_b> <snrs_b> <exps_r> <snrs_r>

    Where exps_b, snrs_b, exps_r, and snrs_r are either integers or lists of
    integers in the case the target has been observed multiple times.

    Parameters
    ----------
    source_ids: str list
        List of source_ids.

    obs_dict: dict
        Dictionary containing the compiled MIKE data as output by 
        plumage.spectra_mike.collate_mike_obs, with the following keys:
            'obs_info' - pandas DataFrame with info from fits headers
            'orders_b' - blue echelle orders, shape [n_order]
            'wave_b'   - blue wavelength scales, shape [n_star, n_order, n_px]
            'spec_b'   - blue spectra, shape [n_star, n_order, n_px]
            'sigma_b'  - blue uncertainties, shape [n_star, n_order, n_px]
            'orders_r' - red echelle orders, shape [n_order]
            'wave_r'   - red wavelength scales, shape [n_star, n_order, n_px]
            'spec_r'   - red spectra, shape [n_star, n_order, n_px]
            'sigma_r'  - red uncertainties, shape [n_star, n_order, n_px]
    """
    # Initialise lists to store compiled values
    exps_b = []
    exps_r = []
    snrs_b = []
    snrs_r = []

    # Loop over all expected source_ids and check against observations
    for sid in source_ids:
        # Find the index/indices corresponding to this star
        ii = np.argwhere(obs_dict["obs_info"]["source_id"].values == sid)

        # If we didn't find the star, use default values and continue
        if len(ii) == 0:
            exps_b.append(0)
            exps_r.append(0)
            snrs_r.append(0)
            snrs_b.append(0)
            continue

        ii = ii.flatten()
        
        # Initialise lists for just this star
        exp_bs = []
        exp_rs = []
        snr_bs = []
        snr_rs = []
        
        # Loop over all indices and extract exposure times and SNRs
        for iii in ii:
            exp_b = obs_dict["obs_info"]["exp_time_b"][iii]
            exp_r = obs_dict["obs_info"]["exp_time_r"][iii]

            if np.isnan(exp_b):
                exp_b = 0
            if np.isnan(exp_r):
                exp_r = 0
            
            exp_bs.append(int(exp_b))
            exp_rs.append(int(exp_r))
            
            with warnings.catch_warnings():
                msg1 = "divide by zero encountered in true_divide"
                msg2 = "invalid value encountered in true_divide"
                msg3 = "Mean of empty slice"
                warnings.filterwarnings(action='ignore', message=msg1)
                warnings.filterwarnings(action='ignore', message=msg2)
                warnings.filterwarnings(action='ignore', message=msg3)

                spec_b = obs_dict["spec_b"][iii]
                sigma_b = obs_dict["sigma_b"][iii]
                snr_b = spec_b/sigma_b
                mm_b = np.isfinite(snr_b)
                med_b = np.nanmedian(snr_b[mm_b])
                
                # Account for NaNs
                if np.isnan(med_b):
                    med_b = 0
                
                snr_bs.append(int(med_b))
                
                spec_r = obs_dict["spec_r"][iii]
                sigma_r = obs_dict["sigma_r"][iii]
                snr_r = spec_r / sigma_r
                mm_r = np.isfinite(snr_r)
                snr_rs.append(int(np.nanmedian(snr_r[mm_r])))
        
        exps_b.append(exp_bs)
        exps_r.append(exp_rs)    
        snrs_b.append(snr_bs)
        snrs_r.append(snr_rs)

    # Print tab-separated row per star, and format to remove []
    print("source_id\texps_b\tsnrs_b\texps_r\tsnrs_r")

    for sid_i, sid in enumerate(source_ids):
        print_str = "\t".join(
            (sid, str(exps_b[sid_i]), str(snrs_b[sid_i]), str(exps_r[sid_i]), 
             str(snrs_r[sid_i]),))
        print_str = print_str.replace("[", "").replace("]", "")

        print(print_str)