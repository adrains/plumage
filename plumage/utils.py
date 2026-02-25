"""Utilities functions to assist.
"""
import os
import numpy as np
import pandas as pd
import mk_mass 
from astropy.io import fits
from astropy.table import Table
import plumage.parameters as params

# -----------------------------------------------------------------------------
# Crossmatching
# ----------------------------------------------------------------------------- 
def do_id_crossmatch(observations, catalogue):
    """Do an ID crossmatch and add the Gaia DR2 ID to observations

    Note that this is a bit messy at present, consider placeholder.

    Parameters
    ----------
    observations: pandas dataframe
        Pandas dataframe logging details about each observation to match to.

    catalogue: pandas dataframe
        The imported catalogue of all potential observed stars and their 
        science programs in the form of a pandas dataframe
    """
    # Get the IDs
    ob_ids = observations["id"].values
    
    # Initialise array of unique IDs
    source_ids = []
    program = []
    subset = []

    id_cols = ["source_id", "2MASS_Source_ID", "HD", "TOI", "bayer", "other"]

    for ob_id_i, ob_id in enumerate(ob_ids):
        id_found = False

        for id_col in id_cols:
            if id_col == "HD":
                trunc_id = ob_id.replace(" ","")
                idx = np.argwhere(catalogue[id_col].values == trunc_id)
            elif id_col == "TOI":
                trunc_id = ob_id.replace("TOI", "").strip()
                idx = np.argwhere(catalogue[id_col].values == trunc_id)
            else:
                idx = np.argwhere(catalogue[id_col].values==ob_id)

            if len(idx) == 1:
                source_ids.append(catalogue.iloc[int(idx)]["source_id"])
                program.append(catalogue.iloc[int(idx)]["program"])
                subset.append(catalogue.iloc[int(idx)]["subset"])
                id_found = True
                break
        
        
        # If get to this point and no ID/program, put placeholder and print
        if not id_found:
            print("No ID match for #%i: %s" % (ob_id_i, ob_id))
            source_ids.append("")
            program.append("")
            subset.append("")

    observations["source_id"] = source_ids
    observations["program"] = program
    observations["subset"] = subset

    observations.set_index("source_id", inplace=True) 


def do_id_crossmatch_modern(observations, star_info_fn,):
    """Modern ID crossmatch to add the Gaia DR3 ID to observations. To do this
    we import the same TSV file of Gaia and 2MASS info we do later for
    parameter determination, and use it to search for IDs.

    Parameters
    ----------
    observations: pandas DataFrame
        Pandas dataframe logging details about each observation to match to.

    star_info_fn: str
        Filepath to TSV of star literature info and IDs to crossmatch with.

    Returns
    -------
    successful_crossmatch: 1D bool array
        Mask of shape [n_stars] indicating whether the star crossmatched
        successfully.

    star_info: pandas DataFrame
        DataFrame of literature info now with a bool column 'has_observation'
        indicating whether this particular star of interest has a matched
        observation.
    """
    # Import the dataframe
    star_info = pd.read_csv(
        star_info_fn,
        delimiter="\t",
        comment="#",
        dtype={"source_id_dr3":str, "source_id_dr2":str, "TOI":str},)

    # Get the IDs that each target was *observed* with
    ob_ids = observations["id"].values
    
    # Initialise array of unique IDs
    source_ids_dr3 = []
    successful_crossmatch = np.full_like(ob_ids, True).astype(bool)
    
    # Add column to star_info to keep track of which expected stars were
    # crossmatched successfully.
    star_info["has_observation"] = False

    # Potential IDs that the target could have been observed under
    id_cols = ["source_id_dr2", "source_id_dr3", "TOI",]

    for ob_id_i, ob_id in enumerate(ob_ids):
        id_found = False

        for id_col in id_cols:
            # Check TOI ID (requires string modification)
            if id_col == "TOI" and id_col in star_info.columns:
                trunc_id = ob_id.replace("TOI", "").strip()
                idx = np.argwhere(star_info[id_col].values == trunc_id)

            # Otherwise ID 'as is' was used as the object ID (e.g. Gaia ID)
            else:
                idx = np.argwhere(star_info[id_col].values==ob_id)

            # If we've found an ID, update
            if len(idx) == 1:
                idx = int(idx)
                source_ids_dr3.append(star_info.iloc[idx]["source_id_dr3"])
                id_found = True
                star_info.loc[idx, "has_observation"] = True
                break
        
        # Mark those stars which weren't crossmatched
        if not id_found:
            source_ids_dr3.append("")
            successful_crossmatch[ob_id_i] = False

    print("Successful crossmatches: {}/{}".format(
        np.sum(successful_crossmatch), len(successful_crossmatch)))

    # Add the DR3 source_id to our dataframe as the index
    observations["source_id_dr3"] = source_ids_dr3
    observations.set_index("source_id_dr3", inplace=True)

    return successful_crossmatch, star_info


def do_activity_crossmatch(observations, activity):
    """

    Parameters
    ----------
    observations: pandas dataframe
        Pandas dataframe logging details about each observation to match to.

    activity: 
        
    """
    # Observation IDs
    ob_ids = observations["uid"].values

    # Activity IDs
    activity_ids = activity["Gaia_ID"].astype(str)
    
    # Initialise arrays
    ew_li = []
    ew_ha = []
    ew_ca_hk = []
    ew_ca_h = []
    ew_ca_k = []

    for ob_id_i, ob_id in enumerate(ob_ids):
        # Gaia DR2
        idx = np.argwhere(activity_ids==ob_id)

        if len(idx) == 1:
            ew_li.append(activity['EW(Li)'][int(idx)])
            ew_ha.append(activity['EW(Ha)'][int(idx)])
            ew_ca_hk.append(activity['EW(HK)'][int(idx)])
            ew_ca_h.append(activity['EW(H)'][int(idx)])
            ew_ca_k.append(activity['EW(K)'][int(idx)])
            continue
        else:
            ew_li.append(np.nan)
            ew_ha.append(np.nan)
            ew_ca_hk.append(np.nan)
            ew_ca_h.append(np.nan)
            ew_ca_k.append(np.nan)

        

    observations["ew_li"] = ew_li
    observations["ew_ha"] = ew_ha
    observations["ew_ca_hk"] = ew_ca_hk
    observations["ew_ca_h"] = ew_ca_h
    observations["ew_ca_k"] = ew_ca_k


def load_crossmatch_catalogue(
    cat_type="csv", 
    cat_file="data/all_2m3_star_ids.csv"):
    """Load in the catalogue of all stars observed. Currently the csv 
    catalogue is complete, whereas the fits catalogue is meant to be a 
    FunnelWeb input catalogue crossmatch which is broken/not complete.

    Parameters
    ----------
    cat_type: string
        Kind of catalogue to load in. Accepts either "csv" or "fits"
    cat_file: string
        Location of the catalogue to import

    Returns
    -------
    catalogue: pandas dataframe
        The imported catalogue of all potential observed stars and their 
        science programs in the form of a pandas dataframe
    """
    # Import catalogue
    if cat_type == "csv":
        catalogue_file = cat_file
        catalogue = pd.read_csv(catalogue_file, sep=",", header=0, 
                                dtype={"source_id":str, "TOI":str, 
                                "2MASS_Source_ID":str, "subset":str},
                                na_values=[], keep_default_na=False)
        catalogue.rename(columns={"source_id":"source_id"}, inplace=True)

    elif cat_type == "fits":
        catalogue_file = cat_file
        catalogue = Table.read(catalogue_file).to_pandas() 
        catalogue.rename(columns={"Gaia ID":"source_id"}, inplace=True)  
        catalogue["source_id"] = catalogue["source_id"].astype(str)
        catalogue["TOI"] = catalogue["TOI"].astype(str)
        catalogue["2MASS_Source_ID_1"] = [
            id.decode().replace(" ", "") 
            for id in catalogue["2MASS_Source_ID_1]"]]
        catalogue["program"] = [prog.decode().replace(" ", "") 
                                for prog in catalogue["program"]]
        catalogue["subset"] = [ss.decode().replace(" ", "") 
                                for ss in catalogue["subset"]]
    
    return catalogue

# -----------------------------------------------------------------------------
# Spectra and table of observations
# ----------------------------------------------------------------------------- 
def load_fits(label, path="spectra"):
    """Load blue and red spectra, plus observational log table from fits file 
    with the format:
        HDU 0: 1D blue wavelength scale
        HDU 1: 2D blue band flux [star, wl]
        HDU 2: 2D blue band flux uncertainties [star, wl]
        HDU 3: 1D red wavelength scale
        HDU 4: 2D red band flux [star, wl]
        HDU 5: 2D red band flux uncertainties [star, wl]
        HDU 6: table of observational information
    
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
    fits_path = os.path.join(path,  "spectra_{}.fits".format(label))

    with fits.open(fits_path) as fits_file: 
        # TODO - do this with fits headers
        n_px_b = len(fits_file[0].data)
        n_px_r = len(fits_file[3].data)
        n_stars = len(fits_file[1].data)

        # Blue
        wl_b = np.tile(fits_file[0].data, n_stars).reshape((n_stars, n_px_b))
        flux_b = fits_file[1].data
        e_flux_b = fits_file[2].data
        spec_b = np.stack((wl_b, flux_b, e_flux_b))
        spec_b = np.swapaxes(spec_b, 0, 1)

        # Red
        wl_r = np.tile(fits_file[3].data, n_stars).reshape((n_stars, n_px_r))
        flux_r = fits_file[4].data
        e_flux_r = fits_file[5].data
        spec_r = np.stack((wl_r, flux_r, e_flux_r))
        spec_r = np.swapaxes(spec_r, 0, 1)

        # Extract the table of observations
        obs_tab = Table(fits_file[6].data)
        obs_pd = obs_tab.to_pandas().set_index("source_id")

        return spec_b, spec_r, obs_pd


def save_fits(spectra_b, spectra_r, observations, label, path="spectra"):
    """Save blue and red spectra, plus observational log table as a fits file 
    with the format:
        HDU 0: 1D blue wavelength scale
        HDU 1: 2D blue band flux [star, wl]
        HDU 2: 2D blue band flux uncertainties [star, wl]
        HDU 3: 1D red wavelength scale
        HDU 4: 2D red band flux [star, wl]
        HDU 5: 2D red band flux uncertainties [star, wl]
        HDU 6: table of observational information
    
    File will be saved as {path}/spectra_{label}.fits

    Parameters
    ----------
    spectra_b: float array
        3D numpy array containing blue arm spectra of form 
        [N_ob, wl/spec/sigma, flux].

    spectra_r: float array
        3D numpy array containing red arm spectra of form 
        [N_ob, wl/spec/sigma, flux].
    
    observations: pandas dataframe
        Dataframe containing information about each observation.

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    path: string
        Path to save the fits file to
    """
    # Intialise HDU List
    hdu = fits.HDUList()

    # Assert that all wavelength scales are the same
    for wl_i in range(len(spectra_b[0,0])):
        assert len(set(spectra_b[:, 0, wl_i])) == 1
    
    for wl_i in range(len(spectra_r[0,0])):
        assert len(set(spectra_r[:, 0, wl_i])) == 1

    # HDU 1: Blue wavelength scale
    wave_img =  fits.PrimaryHDU(spectra_b[0,0])
    wave_img.header["EXTNAME"] = ("WAVE_B", "Blue band wavelength scale")
    hdu.append(wave_img)

    # HDU 2: Blue band flux
    spec_img =  fits.PrimaryHDU(spectra_b[:,1])
    spec_img.header["EXTNAME"] = ("SPEC_B", "Blue band fluxes for all stars")
    hdu.append(spec_img)

    # HDU 3: Blue band flux uncertainty
    e_spec_img =  fits.PrimaryHDU(spectra_b[:,2])
    e_spec_img.header["EXTNAME"] = ("SIGMA_B", 
                                  "Blue band flux uncertainties for all stars")
    hdu.append(e_spec_img)

    # HDU 4: Red wavelength scale
    wave_img =  fits.PrimaryHDU(spectra_r[0,0])
    wave_img.header["EXTNAME"] = ("WAVE_R", "Red band wavelength scale")
    hdu.append(wave_img)

    # HDU 5: Red band flux
    spec_img =  fits.PrimaryHDU(spectra_r[:,1])
    spec_img.header["EXTNAME"] = ("SPEC_R", "Red band fluxes for all stars")
    hdu.append(spec_img)

    # HDU 6: Red band flux uncertainty
    e_spec_img =  fits.PrimaryHDU(spectra_r[:,2])
    e_spec_img.header["EXTNAME"] = ("SIGMA_R", 
                                   "Red band flux uncertainties for all stars")
    hdu.append(e_spec_img)

    # HDU 7: table of observational information
    obs_tab = fits.BinTableHDU(Table.from_pandas(observations.reset_index()))
    obs_tab.header["EXTNAME"] = ("OBS_TAB", "Observation info table")
    hdu.append(obs_tab)
    
    # Done, save
    save_path = os.path.join(path,  "spectra_{}.fits".format(label))
    hdu.writeto(save_path, overwrite=True)


# -----------------------------------------------------------------------------
# Loading and saving/updating fits table
# -----------------------------------------------------------------------------
def load_fits_table(
    extension,
    label,
    fn_base="spectra",
    path="spectra",
    ext_label="",
    do_use_dr3_id=False,):
    """Loads in the data from specified fits table HDU.

    The file is loaded from <path>/<fn_base>_<label>.fits.

    Parameters
    ----------
    extension: string
        Which fits table extension to save. Currently either 'OBS_TAB' or 
        'TRANSIT_FITS'

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    fn_base: string, default: "spectra"
        Base string of filename.

    path: string
        Path to save the fits file to.

    ext_label: str, default: ''
        Sublabel of the extension. At the moment only applicable to the 
        'CANNON_MODEL' extension to note the specifics of the model.

    do_use_dr3_id: bool, default: False
        Set to true if we're using 'source_id_dr3' as the ID column

    Returns
    -------
    obs_pd: pandas dataframe
        Dataframe containing information about each observation.
    """
    # List of valid extensions
    valid_ext = ["OBS_TAB", "TRANSIT_FITS", "CANNON_INFO", "CANNON_MODEL"]

    # Needed to reapply the DataFrame index, which astropy does not respect
    ext_index = {
        "OBS_TAB":"source_id",
        "TRANSIT_FITS":"TOI",
        "CANNON_INFO":"source_id_dr3",
        "CANNON_MODEL":"source_id_dr3",
    }

    if do_use_dr3_id:
        ext_index["OBS_TAB"] = "source_id_dr3"

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
    fits_path = os.path.join(path, "{}_{}.fits".format(fn_base, label))

    with fits.open(fits_path, mode="readonly") as fits_file:
        if ext_full in fits_file:
            obs_tab = Table(fits_file[ext_full].data)
            obs_pd = obs_tab.to_pandas()
        else:
            raise Exception("No table of that extension or wrong fits format")

    return obs_pd.set_index(ext_index[extension])


def convert_1e20_to_nans(fits_table):
    """Converts all instances of the astropy default filler values 1e20 to nans

    Parameters
    ----------
    fits_table: astropy.table.table.Table
        The fits table to convert default 1e20 values to nan in.
    """
    for col in fits_table.columns:
        if fits_table[col].dtype == np.dtype('<f8'):
            fits_table[col][fits_table[col] == 1e20] = np.nan


def save_fits_table(
    extension,
    dataframe,
    label,
    fn_base="spectra",
    path="spectra",
    ext_label="",):
    """Update table of observations stored in given fits file.

    The file is saved as <path>/<fn_base>_<label>.fits.

    Parameters
    ----------
    extension: string
        Which fits table extension to save. Currently either 'OBS_TAB' or 
        'TRANSIT_FITS'

    dataframe: pandas.DataFrame
        Dataframe table to be saved

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    fn_base: string, default: "spectra"
        Base string of filename.

    path: string, default: 'spectra'
        Path to save the fits file to.

    ext_label: str, default: ''
        Sublabel of the extension. At the moment only applicable to the 
        'CANNON_MODEL' extension to note the specifics of the model.
    """
    # Dict mapping extensions to their descriptions
    valid_ext = {
        "OBS_TAB":"Observation info table", 
        "TRANSIT_FITS":"Table of transit light curve fit results",
        "CANNON_INFO":"Table of obs/lit info for Cannon benchmarks",
        "CANNON_MODEL":"Table of adopted benchmarks and model results."}

    if extension not in valid_ext.keys():
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext.keys()))

    # Load in the fits file
    fits_path = os.path.join(path, "{}_{}.fits".format(fn_base, label))

    with fits.open(fits_path, mode="update") as fits_file:
        # Write out the full extention name
        if ext_label != "":
            ext_full = "{}_{}".format(extension, ext_label)
        else:
            ext_full = extension

        if ext_full in fits_file:
            # Update table
            astropy_tab = Table.from_pandas(dataframe.reset_index())
            convert_1e20_to_nans(astropy_tab)

            fits_tab = fits.BinTableHDU(astropy_tab)
            fits_file[ext_full].data = fits_tab.data
            fits_file.flush()
        else:
            # Save table for first time
            astropy_tab = Table.from_pandas(dataframe.reset_index())
            convert_1e20_to_nans(astropy_tab)

            fits_tab = fits.BinTableHDU(astropy_tab)
            fits_tab.header["EXTNAME"] = (ext_full, valid_ext[extension])
            fits_file.append(fits_tab)
            fits_file.flush()


def merge_activity_table_with_obs(
    observations,
    label,
    path="data/tess_wifes_youth_indicators.fits",
    fix_missing_source_id=False,):
    """Function to merge activity table with observations list.
    """
    # Import
    youth_indicators = Table(fits.open(path)[1].data).to_pandas()

    # Fix incorrect source_id. TODO: fix this properly
    if fix_missing_source_id:
        youth_indicators.at[0, "source_id"] = 4785886941312921344

    # Reset the index
    youth_indicators["source_id"] = youth_indicators["source_id"].astype(str)
    youth_indicators.set_index("source_id", inplace=True)

    # Grab only the part we care about
    cols = ["Sraw", "SMW_WiFeS", "logR'HK", "EW(Ha)", "EW(Li)"]
    youth_indicators = youth_indicators[cols]

    observations = observations.join(
        youth_indicators,
        how="inner",)

    return observations


# -----------------------------------------------------------------------------
# Loading and saving/updating fits image HDUs
# ----------------------------------------------------------------------------- 
def load_fits_image_hdu(extension, label, path="spectra", arm="r"):
    """Loads in the data from specified fits image HDU.

    Parameters
    ----------
    extension: string
        Which fits image HDU to load. Currently either wave, spec, sigma, 
        bad_px, or synth.

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    path: string
        Path to save the fits file to

    arm: string
        Arm of the spectrograph, either "b" or "r".

    Returns
    -------
    data: numpy array
        Data to load from the fits image HDU. Currently supported:
         1) wave, the wavelength scale, [n_stars, n_pixels]
         2) spec, science spectra fluxes, [n_stars, n_pixels]
         3) sigma, science spectra uncertainties, [n_stars, n_pixels]
         4) bad_px, boolean array indicating pixels flagged as bad (i.e. 
         telluric contamination, sigma cut on residuals from science compared 
         to synthetic spectrum), [n_stars, n_pixels] 
         5) synth, best fit synthetic spectra, [n_stars, n_pixels] 
        Which can either correspond to the blue or red arms of the spectrograph
    """
    # Ensure extension is valid - maps the user specification to the start of
    # EXTNAME (non-arm component) and the data type to load as
    valid_ext = {
        "wave":("WAVE_", float),
        "spec":("SPEC_", float),
        "sigma":("SIGMA_", float),
        "bad_px":("BAD_PX_MASK_", bool),
        "synth":("SYNTH_FIT_", float),
        "synth_lit":("SYNTH_LIT_", float),
        "rest_frame_synth_lit":("REST_FRAME_SYNTH_LIT_", float),
        "rest_frame_wave":("REST_FRAME_WAVE_", float),
        "rest_frame_spec":("REST_FRAME_SPEC_", float),
        "rest_frame_sigma":("REST_FRAME_SIGMA_", float),
        "rest_frame_spec_norm":("REST_FRAME_SPEC_NORM_", float),
        "rest_frame_sigma_norm":("REST_FRAME_SIGMA_NORM_", float),
    }

    if extension not in valid_ext.keys():
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext.keys()))

    # Ensure correct value of arm is passed
    arm = arm.upper()
    valid_arms = ("B", "R", "BR")
    
    if arm not in valid_arms:
        raise ValueError("Arm must be in {}".format())

    # All good, so construct the extension name
    extname = valid_ext[extension][0] + arm

    # Load in the fits file
    fits_path = os.path.join(path,  "spectra_{}.fits".format(label))

    with fits.open(fits_path, mode="readonly") as fits_file:
        if extname in fits_file:
            data = fits_file[extname].data.astype(valid_ext[extension][1])
        else:
            raise Exception("No {} extension for {} arm".format(extension, arm))

    return data


def save_fits_image_hdu(
    data,
    extension,
    label,
    fn_base="spectra",
    path="spectra",
    arm="r"):
    """Saves/updates the data from specified fits image HDU.

    The existing file is <path>/<fn_base>_<label>.fits.

    Parameters
    ----------
    data: numpy array
        Data to save/update to fits image HDU. Currently supported:
         1) wave, the wavelength scale, [n_stars, n_pixels]
         2) spec, science spectra fluxes, [n_stars, n_pixels]
         3) sigma, science spectra uncertainties, [n_stars, n_pixels]
         4) bad_px, boolean array indicating pixels flagged as bad (i.e. 
         telluric contamination, sigma cut on residuals from science compared 
         to synthetic spectrum), [n_stars, n_pixels] 
         5) synth, best fit synthetic spectra, [n_stars, n_pixels] 

    extension: string
        Which fits image HDU to load. Currently either wave, spec, sigma, 
        bad_px, or synth.

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    fn_base: string, default: "spectra"
        Base string of filename.

    path: string
        Path to save the fits file to

    arm: string
        Arm of the spectrograph, either "b" or "r".
    """
    # Ensure extension is valid - maps the user specification to the start of
    # EXTNAME (non-arm component) and the data type to save as
    valid_ext = {
        "wave":("WAVE_", float),
        "spec":("SPEC_", float),
        "sigma":("SIGMA_", float),
        "bad_px":("BAD_PX_MASK_", int),
        "synth":("SYNTH_FIT_", float),
        "synth_lit":("SYNTH_LIT_", float),
        "rest_frame_synth_lit":("REST_FRAME_SYNTH_LIT_", float),
        "rest_frame_wave":("REST_FRAME_WAVE_", float),
        "rest_frame_spec":("REST_FRAME_SPEC_", float),
        "rest_frame_sigma":("REST_FRAME_SIGMA_", float),
        "rest_frame_spec_norm":("REST_FRAME_SPEC_NORM_", float),
        "rest_frame_sigma_norm":("REST_FRAME_SIGMA_NORM_", float),
        "stellar_frame_telluric_trans":("STELLAR_FRAME_T_TRANS_", float),
        "wave_3D":("WAVE_3D_", float),      # MIKE
        "spec_3D":("SPEC_3D_", float),      # MIKE
        "sigma_3D":("SIGMA_3D_", float),    # MIKE
    }

    if extension not in valid_ext.keys():
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext.keys()))

    # Ensure correct value of arm is passed
    arm = arm.upper()
    valid_arms = ("B", "R", "BR")
    
    if arm not in valid_arms:
        raise ValueError("Arm must be in {}".format())

    # All good, so construct the extension name
    extname = valid_ext[extension][0] + arm

    # Load in the fits file
    fits_path = os.path.join(path, "{}_{}.fits".format(fn_base, label))

    with fits.open(fits_path, mode="update") as fits_file: 
        # First check if the HDU already exists
        if extname in fits_file:
            fits_file[extname].data = data.astype(valid_ext[extension][1])
        
        # Not there, make and append
        else:
            hdu = fits.PrimaryHDU(data.astype(valid_ext[extension][1]))
            hdu.header["EXTNAME"] = (extname,
                "{} extension for {} arm".format(extension, arm)
                )
            fits_file.append(hdu)

        fits_file.flush()


# -----------------------------------------------------------------------------
# Loading in literature info (e.g. photometry)
# ----------------------------------------------------------------------------- 
def load_info_cat(
    path, 
    make_observed_col_bool_on_yes=True,
    only_import_observed=False,
    only_import_in_paper=True,
    use_mann_code_for_masses=True,
    gdr="",
    has_2mass=True,
    do_use_mann_15_JHK=False,
    m15_fn="data/mann15_all_dr3.tsv",
    do_calculate_gaia_mag_uncertainties=False,):
    """Function to import a data catalogue of containing Gaia and 2MASS data
    for our science targets, and prepare a pandas DataFrame.

    Note: this function is no longer compatible with the code from Rains+2021,
    and legacy parameters have been removed.

    Parameters
    ----------
    path: str
        Filepath to the catalogue, either a TSV file or a pre-imported fits.

    make_observed_col_bool_on_yes: bool, default: True
        If True, we replace the contents of the 'observed' column with True
        where the existing values are 'yes' and False otherwise.

    only_import_observed: bool, default: False
        If True, removes all stars where 'observed' == False.

    only_import_in_paper: bool, default: True
        If True, removes stars where 'in_paper' == False.

    use_mann_code_for_masses: bool, default: True
        If True, uses https://github.com/awmann/M_-M_K- for computing M_star to
        properly take into account posteriors. If False, simply uses the
        empirical relations from Mann+19.

     gdr: str, default: ""
        String expected to be at the end of all Gaia columns to represent the
        Gaia data release, e.g. 'dr3' means that we expect 'BP_mag_dr3' as a 
        column name.

    has_2mass: bool, default: True
        If False, we don't continue to do anything in this function that
        requires a 2MASS magnitude and return early.

    do_use_mann_15_JHK: bool, default: False
        If True, we swap out saturated 2MASS photometry for those targets with
        spectrophotometrically integrated 2MASS magnitudes from Mann+2015.

    m15_fn: str, default: "data/mann15_all_dr3.tsv"
        Filepath to the catalogue from Mann+2015 to crossmatch with if we want
        to adopt their JHKs band photometry but haven't already done the
        crossmatch ourselves.

    do_calculate_gaia_mag_uncertainties: bool, default: False
        Whether to calculate the Gaia G, BP, and RP magnitude uncertainties
        from the Gaia fluxes.

    Returns
    -------
    info_cat: pandas DataFrame
        DataFrame containing our stellar catalogue.
    """
    # We'll maintain compatability between DR2 and DR3 data, but it requires a
    # bit of extra work
    if gdr == "":
        sid_drx = "source_id"
    else:
        gdr = "_{}".format(gdr)
        sid_drx = "source_id{}".format(gdr)

    # If loading a fits file, can't start with pandas
    if ".fits" in path:
        with fits.open(path, mode="readonly") as fits_file:
            fits_tab = Table(fits_file[1].data)
            info_cat = fits_tab.to_pandas()

        info_cat[sid_drx] = info_cat[sid_drx].astype(str)

    # Otherwise import using pandas
    else:
        info_cat = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            dtype={"source_id":str, "source_id_dr2":str, "source_id_dr3":str})

    # Convert 'observed' column properly to boolean
    if make_observed_col_bool_on_yes:
        info_cat["observed"] = info_cat["observed"] == "yes"
    
    # Set the index to be source_id
    info_cat.set_index(sid_drx, inplace=True)

    # Don't import stars that haven't been observed
    if only_import_observed:
        info_cat = info_cat[info_cat["observed"].values]
    
    # Don't import stars that aren't marked as 'in_paper' True
    if only_import_in_paper and "in_paper" in info_cat.columns:
        info_cat = info_cat[info_cat["in_paper"].values]

    # Make boolean for blended 2MASS photometry
    if "blended_2mass" in info_cat:
        info_cat["blended_2mass"] = [True if xx == "yes" else False 
                                    for xx in info_cat["blended_2mass"].values]
    else:
        info_cat["blended_2mass"] = np.nan

    if "wife_obs" not in info_cat:
        info_cat["wife_obs"] = 1

    # Set Gaia dup column to be boolean
    info_cat["dup{}".format(gdr)] = info_cat["dup{}".format(gdr)].astype(bool)

    # -------------------------------------------------------------------------
    # Compute Gaia magnitude uncertainties
    # -------------------------------------------------------------------------
    if do_calculate_gaia_mag_uncertainties:
        g_flux = info_cat["phot_g_mean_flux{}".format(gdr)]
        g_error = info_cat["phot_g_mean_flux_error{}".format(gdr)]
        e_G_mag = ((-2.5/np.log(10)*g_error/g_flux)**2+0.00279017**2)**0.5
        info_cat["e_G_mag{}".format(gdr)] = e_G_mag

        bp_flux = info_cat["phot_bp_mean_flux{}".format(gdr)]
        bp_error = info_cat["phot_bp_mean_flux_error{}".format(gdr)]
        e_BP_mag = ((-2.5/np.log(10)*bp_error/bp_flux)**2+0.00279017**2)**0.5
        info_cat["e_BP_mag{}".format(gdr)] = e_BP_mag

        rp_flux = info_cat["phot_rp_mean_flux{}".format(gdr)]
        rp_error = info_cat["phot_rp_mean_flux_error{}".format(gdr)]
        e_RP_mag = ((-2.5/np.log(10)*rp_error/rp_flux)**2+0.00279017**2)**0.5
        info_cat["e_RP_mag{}".format(gdr)] = e_RP_mag

    # -------------------------------------------------------------------------
    # Distance, absolute magnitudes, and colours
    # -------------------------------------------------------------------------
    # Grab parallax information for convenience 
    plx = info_cat["plx{}".format(gdr)]
    e_plx = info_cat["e_plx{}".format(gdr)]

    # Compute distance
    info_cat["dist"] = 1000 / plx
    info_cat["e_dist"] = np.abs(info_cat["dist"] * e_plx / plx)
    
    # If no 2MASS, return early
    if not has_2mass:
        return info_cat

    # -------------------------------------------------------------------------
    # Swapping in Mann+15 photometry
    # -------------------------------------------------------------------------
    # For stars in Mann+15 with non-AAA 2MASS photometry, we'll adopt the
    # Mann+2015 values as these won't be saturated for use with our photometric
    # relations.
    if do_use_mann_15_JHK:
        # Magnitudes and uncertainties we're interested in
        mags = ["J_mag", "H_mag", "K_mag"]
        e_mags = ["e_J_mag", "e_H_mag", "e_K_mag"]
        
        # If there's not already M15 information in the file, do a crossmatch
        if "K_mag_m15" not in info_cat.columns.values:
            # Import M15 catalogue
            m15_df = pd.read_csv(
                m15_fn,
                delimiter="\t",
                dtype={"source_id":str, "source_id_dr3":str},)
            m15_df.rename(columns={"source_id":"source_id_dr3"}, inplace=True)
            m15_df.set_index("source_id_dr3", inplace=True)

            icj = info_cat.join(m15_df, on="source_id_dr3", rsuffix="_m15")

            # Add crossmatched Mann+15 magnitudes to our original DataFrame
            for mag, e_mag in zip(mags, e_mags):
                info_cat["{}_m15".format(mag)] = icj["{}_m15".format(mag)]
                info_cat["{}_m15".format(e_mag)] = icj["{}_m15".format(e_mag)]

        # Determine which stars need to have their photometry swapped.
        swap_mask = np.logical_and(
            info_cat["Qflg"] != "AAA",               # Stars w/ saturated 2MASS
            ~np.isnan(info_cat["K_mag_m15"].values)) # Stars w/ M+15 photometry

        for mag, e_mag in zip(mags, e_mags):
            # Grab 2MASS columns
            tm_mag = info_cat[mag].values
            e_tm_mag = info_cat[e_mag].values

            # Rename to preserve data
            info_cat.rename(columns={
                mag:"{}_2M".format(mag), e_mag:"{}_2M".format(e_mag),},)
            
            # Replace saturated 2MASS with Mann+2015 data
            tm_mag[swap_mask] = \
                info_cat["{}_m15".format(mag)].values[swap_mask]
            e_tm_mag[swap_mask] = \
                info_cat["{}_m15".format(e_mag)].values[swap_mask]
            
            info_cat[mag] = tm_mag
            info_cat[e_mag] = e_tm_mag
        
        # Also adjust the Qflag
        tm_qflg = info_cat["Qflg"].values
        info_cat.rename(columns={"Qflg":"Qflg_2M"},)
        tm_qflg[swap_mask] = "AAA"
        info_cat["Qflg"] = tm_qflg

        # And add in a boolean for good measure
        info_cat["using_m15_jhk_photometry"] = swap_mask

    # -------------------------------------------------------------------------
    # Absolute magnitudes, and colours
    # -------------------------------------------------------------------------
    # Absolute mags for Gaia and 2MASS photometry
    info_cat["G_mag_abs"] = \
        info_cat["G_mag{}".format(gdr)] - 5*np.log10(info_cat["dist"]/10)
    info_cat["e_G_mag_abs"] = np.sqrt(
        info_cat["e_G_mag{}".format(gdr)]**2
        + (5/(info_cat["dist"]*np.log(10)))**2 * info_cat["e_dist"]**2)

    info_cat["BP_mag_abs"] = \
        info_cat["BP_mag{}".format(gdr)] - 5*np.log10(info_cat["dist"]/10)
    info_cat["e_BP_mag_abs"] = np.sqrt(
        info_cat["e_BP_mag{}".format(gdr)]**2
        + (5/(info_cat["dist"]*np.log(10)))**2 * info_cat["e_dist"]**2)

    info_cat["RP_mag_abs"] = \
        info_cat["RP_mag{}".format(gdr)] - 5*np.log10(info_cat["dist"]/10)
    info_cat["e_RP_mag_abs"] = np.sqrt(
        info_cat["e_RP_mag{}".format(gdr)]**2
        + (5/(info_cat["dist"]*np.log(10)))**2 * info_cat["e_dist"]**2)
    
    info_cat["J_mag_abs"] = info_cat["J_mag"] - 5*np.log10(info_cat["dist"]/10)
    info_cat["e_J_mag_abs"] = np.sqrt(
        info_cat["e_J_mag"]**2
        + (5/(info_cat["dist"]*np.log(10)))**2 * info_cat["e_dist"]**2)

    info_cat["H_mag_abs"] = info_cat["H_mag"] - 5*np.log10(info_cat["dist"]/10)
    info_cat["e_H_mag_abs"] = np.sqrt(
        info_cat["e_H_mag"]**2
        + (5/(info_cat["dist"]*np.log(10)))**2 * info_cat["e_dist"]**2)
    
    info_cat["K_mag_abs"] = info_cat["K_mag"] - 5*np.log10(info_cat["dist"]/10)
    info_cat["e_K_mag_abs"] = np.sqrt(
        info_cat["e_K_mag"]**2
        + (5/(info_cat["dist"]*np.log(10)))**2 * info_cat["e_dist"]**2)

    # Compute additional colours
    info_cat["RP-J"] = info_cat["RP_mag{}".format(gdr)] - info_cat["J_mag"]
    info_cat["J-H"] = info_cat["J_mag"] - info_cat["H_mag"]
    info_cat["H-K"] = info_cat["H_mag"] - info_cat["K_mag"]

    info_cat["G-K"] = info_cat["G_mag{}".format(gdr)] - info_cat["K_mag"]
    info_cat["J-K"] = info_cat["J_mag"] - info_cat["K_mag"]

    # Widest colour lever
    info_cat["BP-K"] = info_cat["BP_mag{}".format(gdr)] - info_cat["K_mag"]
    info_cat["RP-K"] = info_cat["RP_mag{}".format(gdr)] - info_cat["K_mag"]

    # Compute colour uncertainties (assuming no cross-correlation)
    info_cat["e_BP-RP"] = np.sqrt(info_cat["e_BP_mag{}".format(gdr)]**2
                                 + info_cat["e_RP_mag{}".format(gdr)]**2)
    info_cat["e_RP-J"] = np.sqrt(info_cat["e_RP_mag{}".format(gdr)]**2 
                                 + info_cat["e_J_mag"]**2)
    info_cat["e_J-H"] = np.sqrt(info_cat["e_J_mag"]**2 
                                 + info_cat["e_H_mag"]**2)
    info_cat["e_H-K"] = np.sqrt(info_cat["e_H_mag"]**2
                                 + info_cat["e_K_mag"]**2)

    info_cat["e_BP-K"] = np.sqrt(info_cat["e_BP_mag{}".format(gdr)]**2
                                 + info_cat["e_K_mag"]**2)

    info_cat["e_RP-K"] = np.sqrt(info_cat["e_RP_mag{}".format(gdr)]**2
                                 + info_cat["e_K_mag"]**2)
    
    info_cat["e_G-K"] = np.sqrt(info_cat["e_G_mag{}".format(gdr)]**2
                                 + info_cat["e_K_mag"]**2)

    # -------------------------------------------------------------------------
    # Empirical relations
    # -------------------------------------------------------------------------
    # Mann+15 teff (Bp-Rp)
    teffs, e_teffs = params.compute_mann_2015_teff(
        info_cat["BP-RP{}".format(gdr)],
        relation="BP - RP")

    info_cat["teff_m15_bprp"] = teffs
    info_cat["e_teff_m15_bprp"] = e_teffs

    # Mann+15 teff (Bp-Rp, J-H)
    teffs, e_teffs = params.compute_mann_2015_teff(
        info_cat["BP-RP{}".format(gdr)], 
        j_h=info_cat["J-H"],
        relation="BP - RP, J - H")

    info_cat["teff_m15_bprp_jh"] = teffs
    info_cat["e_teff_m15_bprp_jh"] = e_teffs

    # Mann+19 radii
    radii, e_radii = params.compute_mann_2015_radii(info_cat["K_mag_abs"])
    info_cat["radii_m19"] = radii
    info_cat["e_radii_m19"] = e_radii

    # Compute Mann+19 masses from provided code that samples posteriors
    if use_mann_code_for_masses:
        masses = np.full(len(info_cat), np.nan)
        e_masses = np.full(len(info_cat), np.nan)

        for star_i, (source_id, star_info) in enumerate(info_cat.iterrows()):
            # Assign defaults if outside the absolute K bounds of the relation
            if star_info["K_mag_abs"] < 4 or star_info["K_mag_abs"] > 11:
                continue

            # Calculate masses and uncertainties from code provided at:
            # https://github.com/awmann/M_-M_K-
            mass, e_mass = mk_mass.posterior(
                star_info["K_mag"], 
                star_info["dist"],
                star_info["e_K_mag"],
                star_info["e_dist"],
                oned=True,
                silent=True)

            masses[star_i] = mass
            e_masses[star_i] = e_mass

    # Otherwise just use relations from the paper
    else:
        # Compute masses from relation
        masses, e_masses = params.compute_mann_2019_masses(
            info_cat["K_mag_abs"])

        # Exclude those beyond the bounds of the relation
        outside_bounds = np.logical_or(
            info_cat["K_mag_abs"].values < 4,
            info_cat["K_mag_abs"].values > 11)

        masses[outside_bounds] = np.nan
        e_masses[outside_bounds] = np.nan
    
    # Whatever the method, add to dataframe
    info_cat["mass_m19"] = masses
    info_cat["e_mass_m19"] = e_masses

    # Compute logg and e_log from Mann params
    logg, e_logg = params.compute_logg(masses, e_masses, radii, e_radii,)
    info_cat["logg_m19"] = logg
    info_cat["e_logg_m19"] = e_logg

    # Return copy so it isn't fragmented by whatever we've done here.
    info_cat = info_cat.copy()

    return info_cat


def load_exofop_toi_cat(
    toi_cat_path="data/exofop_tess_tois.csv",
    do_ctoi_merge=False,
    ctoi_cat_path="data/exofop_tess_ctois.csv",
    import_additional_tois=True,
    additional_tois_path="data/additional_tois.tsv"):
    """Imports the catalogue of TOIs from NASA ExoFOP, pre-selected on the 
    website to only have TOIs for the TIC IDs we are interested in.

    Note that the files have been modified by commenting out the initial lines,
    and by changing 'TIC ID' to 'TIC'.

    Returns
    -------
    efi: pandas.core.frame.DataFrame
        ExoFOP info dataframe
    
    ...
    """
    # Load in tess info cat and use to clean efi
    tic_info = load_info_cat(
        remove_fp=True,
        only_observed=True,
        do_extinction_correction=False,
        use_mann_code_for_masses=False,)
    toi_info = pd.read_csv(
        toi_cat_path, 
        quoting=1, 
        comment="#",
        index_col="TOI",)

    # Merge with community TOIs
    if do_ctoi_merge:
        ctoi_info = pd.read_csv(
            ctoi_cat_path, 
            quoting=1, 
            comment="#",
            index_col="CTOI",)
        
        # CTOI catalogue uses "Midpoint (BJD)" rather than "Epoch (BJD)"
        #ctoi_info.rename(columns={"Midpoint (BJD)":"Epoch (BJD)"}, inplace=True)

        toi_info = pd.concat(
            [toi_info,ctoi_info],
            axis=0,
            ignore_index=False, 
            sort=False)
    
    # Import additional targets
    if import_additional_tois:
        additional_targets = pd.read_csv(
            "data/additional_tois.tsv", delimiter="\t")
        additional_targets.set_index("TOI", inplace=True)

        toi_info = pd.concat([toi_info,additional_targets], sort=False)

    return toi_info[np.isin(toi_info["TIC"], tic_info["TIC"])]

def load_linelist(
    filename,
    wl_lower=3500,
    wl_upper=7000,
    ew_min_ma=200,):
    """Load in a Thomas Nordlander sourced line list of EWs computed for 
    M-dwarfs into a pandas dataframe format. 

    Parameters
    ----------
    filename: string
        Filepath to the line list file.

    wl_lower: float, default: 3500
        Lower wavelength bound in Angstroms.

    wl_upper: float, default: 7000
        Upper wavelength bound in Angstroms.

    ew_min_ma: float, default: 200
        Minimum line EW in milli-Angstroms.

    Returns
    -------
    line_list: pd.DataFrame
        DataFrame of line list data.

    """
    cols = [
        "atom",
        "state",
        "wl",
        "elow",
        "log_gf",
        "ew_ma",
        "7", "8", "9", "10", "11", "12","13"]  # Don't care about rest

    # Import
    line_list = pd.read_csv(filename, delim_whitespace=True, names=cols)

    # Combine first two columns
    line_list["ion"] = (line_list["atom"].astype(str) + str(" ")
                        + line_list["state"].astype(str))

    # Reorder and get rid of irrelevant columns
    new_cols = ["ion", "wl", "elow", "log_gf", "ew_ma",]
    
    line_list = line_list[new_cols]

    # Apply limits
    mask = np.logical_and(
        np.logical_and(line_list["wl"] > wl_lower, line_list["wl"] < wl_upper),
        line_list["ew_ma"] > ew_min_ma,
    )

    return line_list[mask]