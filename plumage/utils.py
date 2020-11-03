"""Utilities functions to assist.
"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import plumage.parameters as params

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


def load_crossmatch_catalogue(cat_type="csv", 
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

        #catalogue["source_id"] = catalogue["source_id"].astype(str)
        #catalogue["subset"] = catalogue["subset"].astype(str)

        #catalogue["2MASS_Source_ID"] = [str(id).replace(" ", "") 
        #                                for id in catalogue["2MASS_Source_ID"]]
        #catalogue["program"] = [prog.replace(" ", "") 
        #                        for prog in catalogue["program"]]
        #catalogue["subset"] = [str(ss).replace(" ", "") 
        #                        for ss in catalogue["subset"]]

    elif cat_type == "fits":
        catalogue_file = cat_file
        catalogue = Table.read(catalogue_file).to_pandas() 
        catalogue.rename(columns={"Gaia ID":"source_id"}, inplace=True)  
        catalogue["source_id"] = catalogue["source_id"].astype(str)
        catalogue["TOI"] = catalogue["TOI"].astype(str)
        catalogue["2MASS_Source_ID_1"] = [id.decode().replace(" ", "") 
                                        for id in catalogue["2MASS_Source_ID_1]"]]
        catalogue["program"] = [prog.decode().replace(" ", "") 
                                for prog in catalogue["program"]]
        catalogue["subset"] = [ss.decode().replace(" ", "") 
                                for ss in catalogue["subset"]]
    
    return catalogue

def do_standard_crossmatch(catalogue):
    """Crossmatch standards with the catalogue of all potential science
    targets. Very TBD and TODO.
    """
    # Load in standards
    standards = load_standards()

    # Initialise 
    catalogue["teff_lit"] = np.nan
    catalogue["e_teff_lit"] = np.nan
    catalogue["logg_lit"] = np.nan
    catalogue["e_logg_lit"] = np.nan
    catalogue["feh_lit"] = np.nan
    catalogue["e_feh_lit"] = np.nan

    # For each standard catalogue in standards, find matching IDs and add
    # lit values
    for cat_i, std_cat in enumerate(standards):
        print("Running on standard catalogue %i" % cat_i)

        for std_i in range(len(std_cat)):
            gaia_id = std_cat.iloc[std_i].name

            idx = np.argwhere(catalogue["source_id"].values==gaia_id)

            if len(idx) == 1:
                print("Adding %s" % gaia_id)
                try:
                    catalogue.at[int(idx), "teff_lit"] = std_cat.loc[gaia_id]["teff"]
                    catalogue.at[int(idx), "e_teff_lit"] = std_cat.loc[gaia_id]["e_teff"]
                    catalogue.at[int(idx), "logg_lit"] = std_cat.loc[gaia_id]["logg"]
                    catalogue.at[int(idx), "e_logg_lit"] = std_cat.loc[gaia_id]["e_logg"]
                    catalogue.at[int(idx), "feh_lit"] = std_cat.loc[gaia_id]["feh"]
                    catalogue.at[int(idx), "e_feh_lit"] = std_cat.loc[gaia_id]["e_feh"]
                except:
                    print("Gaia DR2 %s has duplicate values" % gaia_id)
                    continue

def load_standards():
    """Load in various standard catalogues. This should be cleaned.
    """
    curr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 
                                             "standards"))

    # Import Mann et al. 2015 Teff standards catalogue                    
    mann = pd.read_csv(os.path.join(curr_path, "mann_constrain_2015.tsv"), 
                       sep="\t", header=1, skiprows=0, 
                       dtype={"source_id":str, "useful":bool})
    mann.set_index("source_id", inplace=True)
    mann = mann[[type(ii) == str for ii in mann.index.values]] 
    mann = mann[mann["useful"]]

    # Import Royas-Ayala et al. 2012 [Fe/H] standards catalogue
    royas = pd.read_csv(os.path.join(curr_path, "rojas-ayala_2012.tsv"), 
                        sep="\t", header=1, skiprows=0, 
                        dtype={"source_id":str, "useful":bool})
    royas.set_index("source_id", inplace=True)
    royas = royas[[type(ii) == str for ii in royas.index.values]]
    royas = royas[royas["useful"]]
    
    # Import Newton et al. 2014 [Fe/H] standard catalogue
    newton = pd.read_csv(os.path.join(curr_path, "newton_cpm_2014.tsv"), 
                         sep="\t", header=1, skiprows=0, 
                         dtype={"source_id":str, "logg":np.float64, 
                                "useful":bool})
    newton.set_index("source_id", inplace=True)
    newton = newton[[type(ii) == str for ii in newton.index.values]]
    newton = newton[newton["useful"]]

    # Import interferometry standards (various sources)
    interferometry = pd.read_csv(os.path.join(curr_path, "interferometry.tsv"), 
                                 sep="\t", header=1, skiprows=0, 
                                 dtype={"source_id":str, "useful":bool})
    interferometry.set_index("source_id", inplace=True)
    interferometry = interferometry[[type(ii) == str 
                                     for ii in interferometry.index.values]] 
    interferometry = interferometry[interferometry["useful"]]

    # Import Herczeg & Hillenbrand 2014 young star standard catalogue
    herczeg = pd.read_csv(os.path.join(curr_path, 
                                       "herczeg_2014_standards_gaia.tsv"), 
                          sep="\t", header=1, skiprows=0, comment="#",
                          dtype={"source_id":str, "useful":bool})
    herczeg.set_index("source_id", inplace=True)
    herczeg = herczeg[[type(ii) == str for ii in herczeg.index.values]]
    herczeg = herczeg[herczeg["useful"]]

    # Miscellaneous SpT standards
    spt = pd.read_csv(os.path.join(curr_path, "spt_standards.tsv"), 
                          sep="\t", header=0, comment="#",
                          dtype={"source_id":str})#, "useful":bool})
    spt.set_index("source_id", inplace=True)
    spt = spt[[type(ii) == str for ii in spt.index.values]]
    #herczeg = herczeg[herczeg["useful"]]

    standards = {"royas":royas, "newton":newton, "mann":mann, 
                 "interferometry":interferometry, "herczeg":herczeg, "spt":spt}

    return standards


def consolidate_standards(
    standards, 
    force_unique=False, 
    remove_standards_with_nan_params=False,
    teff_lims=None,
    logg_lims=None, 
    feh_lims=None,
    assign_default_uncertainties=False,
    force_solar_missing_feh=False,
    ):
    """WARNING: force_unique is a temporary HACK

    TODO: this can be simplified using a dictionary and looping over the keys
    """
    ids = []
    teffs = []
    e_teffs = []
    loggs = []
    e_loggs = []
    fehs = []
    e_fehs = []
    rvs = []
    e_rvs = []
    sources = []

    # There may be some overlap in standards from different catalogues
    for key in standards.keys():
        ids.extend(standards[key].index.values)

        # teff
        teffs.extend(standards[key]["teff"].values)

        if "e_teff" in standards[key]:
            e_teffs.extend(standards[key]["e_teff"].values)
        else:
            e_teffs.extend(np.nan*np.ones_like(standards[key]["teff"]))

        # logg
        loggs.extend(standards[key]["logg"].values)

        if "e_logg" in standards[key]:
            e_loggs.extend(standards[key]["e_logg"])
        else:
            e_loggs.extend(np.nan*np.ones_like(standards[key]["logg"]))
        
        # [Fe/H]
        fehs.extend(standards[key]["feh"].values)

        if "e_feh" in standards[key]:
            e_fehs.extend(standards[key]["e_feh"].values)
        else:
            e_fehs.extend(np.nan*np.ones_like(standards[key]["feh"]))

        # Radial Velocity
        try:
            rvs.extend(standards[key]["rv"].values)
            e_rvs.extend(standards[key]["e_rv"].values)
        except:
            import pdb
            pdb.set_trace()
        
        sources.extend([key]*len(standards[key]))

    data = (ids, teffs, e_teffs, loggs, e_loggs, fehs, e_fehs, rvs, e_rvs, 
            sources)
    columns = ["ids", "teffs", "e_teffs", "loggs", "e_loggs", "fehs", "e_fehs", 
               "rv", "e_rv", "sources"]
    std_params_all = pd.DataFrame(
        {"source_id": ids,
         "teff": teffs,
         "e_teff": e_teffs,
         "logg": loggs,
         "e_logg": e_loggs,
         "feh": fehs,
         "e_feh": e_fehs,
         "rv": rvs,
         "e_rv": e_rvs,
         "source": sources,
         })

    if force_unique:
        std_params_all.drop_duplicates(subset="source_id", inplace=True)

    if force_solar_missing_feh:
        std_params_all.loc[~np.isfinite(std_params_all["feh"]), "feh"] = 0.0

    if remove_standards_with_nan_params:
        std_params_all = std_params_all[np.isfinite(std_params_all["teff"])]
        std_params_all = std_params_all[np.isfinite(std_params_all["logg"])]
        std_params_all = std_params_all[np.isfinite(std_params_all["feh"])]

    if teff_lims is not None:
        mask = np.logical_and(std_params_all["teff"] > teff_lims[0],
                              std_params_all["teff"] < teff_lims[1])
        std_params_all = std_params_all[mask]

    if logg_lims is not None:
        mask = np.logical_and(std_params_all["logg"] > logg_lims[0],
                              std_params_all["logg"] < logg_lims[1])
        std_params_all = std_params_all[mask]

    if feh_lims is not None:
        mask = np.logical_and(std_params_all["feh"] > feh_lims[0],
                              std_params_all["feh"] < feh_lims[1])
        std_params_all = std_params_all[mask]

    if assign_default_uncertainties:
        std_params_all.loc[~np.isfinite(std_params_all["e_teff"]), "e_teff"] = 100
        std_params_all.loc[~np.isfinite(std_params_all["e_logg"]), "e_logg"] = 0.1
        std_params_all.loc[~np.isfinite(std_params_all["e_feh"]), "e_feh"] = 0.1

    return std_params_all


def mask_spectral_wavelengths(spectra_b, spectra_r, ob_mask=None):
    """TODO: This shouldn't be here
    """
    import plumage.spectra as spec
    if ob_mask is None:
        ob_mask = np.ones(len(spectra_b)).astype(bool)
    
    spec_b_subset = spectra_b[ob_mask]
    spec_r_subset = spectra_r[ob_mask]

    # Mask blue
    wl_mask = spec.make_wavelength_mask(spec_b_subset[0,0], 
                                        mask_blue_edges=True)
    dims = spec_b_subset.shape
    wl_mask = np.tile(wl_mask, dims[0]*dims[1]).reshape(dims)
    
    spec_b_subset = spec_b_subset[wl_mask]
    spec_b_subset = spec_b_subset.reshape(
        [dims[0], dims[1], int(len(spec_b_subset)/np.prod(dims[:2]))])

    # Mask red
    wl_mask = spec.make_wavelength_mask(spec_r_subset[0,0])
    dims = spec_r_subset.shape
    wl_mask = np.tile(wl_mask, dims[0]*dims[1]).reshape(dims)
    
    spec_r_subset = spec_r_subset[wl_mask]
    spec_r_subset = spec_r_subset.reshape(
        [dims[0], dims[1], int(len(spec_r_subset)/np.prod(dims[:2]))])

    return spec_b_subset, spec_r_subset


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
def load_fits_table(extension, label, path="spectra"):
    """Loads in the data from specified fits table HDU.

    Parameters
    ----------
    extension: string
        Which fits table extension to save. Currently either 'OBS_TAB' or 
        'TRANSIT_FITS'

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    path: string
        Path to save the fits file to

    Returns
    -------
    obs_pd: pandas dataframe
        Dataframe containing information about each observation.
    """
    # List of valid extensions
    valid_ext = ["OBS_TAB", "TRANSIT_FITS"]

    # Needed to reapply the DataFrame index, which astropy does not respect
    ext_index = {
        "OBS_TAB":"source_id",
        "TRANSIT_FITS":"TOI"
    }

    if extension not in valid_ext:
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext))

    # Load in the fits file
    fits_path = os.path.join(path, "spectra_{}.fits".format(label))

    with fits.open(fits_path, mode="readonly") as fits_file:
        if extension in fits_file:
            obs_tab = Table(fits_file[extension].data)
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


def save_fits_table(extension, dataframe, label, path="spectra"):
    """Update table of observations stored in given fits file.

    Parameters
    ----------
    extension: string
        Which fits table extension to save. Currently either 'OBS_TAB' or 
        'TRANSIT_FITS'

    dataframe: pandas.DataFrame
        Dataframe table to be saved

    label: string
        Unique label (e.g. std, TESS) for the resulting fits file.
    
    path: string
        Path to save the fits file to
    """
    # Dict mapping extensions to their descriptions
    valid_ext = {
        "OBS_TAB":"Observation info table", 
        "TRANSIT_FITS":"Table of transit light curve fit results"}

    if extension not in valid_ext.keys():
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext.keys()))

    # Load in the fits file
    fits_path = os.path.join(path, "spectra_{}.fits".format(label))

    with fits.open(fits_path, mode="update") as fits_file:
        if extension in fits_file:
            # Update table
            astropy_tab = Table.from_pandas(dataframe.reset_index())
            convert_1e20_to_nans(astropy_tab)

            fits_tab = fits.BinTableHDU(astropy_tab)
            fits_file[extension].data = fits_tab.data
            fits_file.flush()
        else:
            # Save table for first time
            astropy_tab = Table.from_pandas(dataframe.reset_index())
            convert_1e20_to_nans(astropy_tab)

            fits_tab = fits.BinTableHDU(astropy_tab)
            fits_tab.header["EXTNAME"] = (extension, valid_ext[extension])
            fits_file.append(fits_tab)
            fits_file.flush()

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
    }

    if extension not in valid_ext.keys():
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext.keys()))

    # Ensure correct value of arm is passed
    arm = arm.capitalize()
    valid_arms = ("B", "R")
    
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


def save_fits_image_hdu(data, extension, label, path="spectra", arm="r"):
    """Saves/updates the data from specified fits image HDU.

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
    }

    if extension not in valid_ext.keys():
        raise ValueError("Invalid extension type. Must be in {}".format(
            valid_ext.keys()))

    # Ensure correct value of arm is passed
    arm = arm.capitalize()
    valid_arms = ("B", "R")
    
    if arm not in valid_arms:
        raise ValueError("Arm must be in {}".format())

    # All good, so construct the extension name
    extname = valid_ext[extension][0] + arm

    # Load in the fits file
    fits_path = os.path.join(path,  "spectra_{}.fits".format(label))

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
def load_info_cat(path="data/tess_info.tsv", clean=True, remove_fp=False, 
                  only_observed=False, use_plx_systematic=True, in_paper=True):
    """

    Incorporates the systematic offset in Gaia DR2 by subtracting the offset
    from the parallax, then adding its uncertainty in quadrature. This makes 
    the parallax *bigger*.

    https://ui.adsabs.harvard.edu/abs/2018ApJ...862...61S/abstract
    """
    # If loading a fits file, can't start with pandas
    if ".fits" in path:
        with fits.open(path, mode="readonly") as fits_file:
            fits_tab = Table(fits_file[1].data)
            info_cat = fits_tab.to_pandas()

        info_cat["source_id"] = info_cat["source_id"].astype(str)

    # Otherwise import using pandas
    else:
        info_cat = pd.read_csv(path, sep="\t", dtype={"source_id":str})

    # Clean
    if clean:
        info_cat["observed"] = info_cat["observed"] == "yes"

    # Set the index to be source_id
    info_cat.set_index("source_id", inplace=True)

    # Make new boolean column for planet candidates or known planets
    if "TOI" in info_cat:
        pc_mask = np.logical_and(
            info_cat["TFOPWG Disposition"] != "FP",
            np.logical_or(
                info_cat["TESS Disposition"] == "PC",
                info_cat["TESS Disposition"] == "KP")
        ).values
        info_cat["pc"] = pc_mask

        if remove_fp:
            info_cat = info_cat[pc_mask]

    if only_observed:
        info_cat = info_cat[info_cat["observed"]]
    
    if in_paper and "in_paper" in info_cat.columns:
        info_cat = info_cat[info_cat["in_paper"]]

    # Make boolean for blended 2MASS photometry
    if "blended_2mass" in info_cat:
        info_cat["blended_2mass"] = [True if xx == "yes" else False 
                                    for xx in info_cat["blended_2mass"].values]
    else:
        info_cat["blended_2mass"] = np.nan

    if "wife_obs" not in info_cat:
        info_cat["wife_obs"] = 1

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Distance, absolute magnitudes, and colours
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Stassun & Torres systematic offsets
    if use_plx_systematic:
        plx_off = -0.082    # mas
        e_plx_off = 0.033   # mas

        # Incorporate the systematic
        plx = info_cat["plx"] - plx_off
        e_plx = np.sqrt(info_cat["e_plx"]**2 + e_plx_off**2)
    
    # Not using offsets
    else:
        plx = info_cat["plx"]
        e_plx = info_cat["e_plx"]
    
    # Compute distance and absolute magnitudes
    info_cat["dist"] = 1000 / plx
    info_cat["e_dist"] = np.abs(info_cat["dist"] * e_plx / plx)

    info_cat["G_mag_abs"] = info_cat["G_mag"] - 5*np.log10(info_cat["dist"]/10)
    info_cat["K_mag_abs"] = info_cat["K_mag"] - 5*np.log10(info_cat["dist"]/10)

    # Compute additional colours
    info_cat["Rp-J"] = info_cat["Rp_mag"] - info_cat["J_mag"]
    info_cat["J-H"] = info_cat["J_mag"] - info_cat["H_mag"]
    info_cat["H-K"] = info_cat["H_mag"] - info_cat["K_mag"]

    info_cat["G-K"] = info_cat["G_mag"] - info_cat["K_mag"]
    info_cat["J-K"] = info_cat["J_mag"] - info_cat["K_mag"]

    # Compute colour uncertainties (assuming no cross-correlation)
    info_cat["e_Bp-Rp"] = np.sqrt(info_cat["e_Bp_mag"]**2
                                 + info_cat["e_Rp_mag"]**2)
    info_cat["e_Rp-J"] = np.sqrt(info_cat["e_Rp_mag"]**2 
                                 + info_cat["e_J_mag"]**2)
    info_cat["e_J-H"] = np.sqrt(info_cat["e_J_mag"]**2 
                                 + info_cat["e_H_mag"]**2)
    info_cat["e_H-K"] = np.sqrt(info_cat["e_H_mag"]**2
                                 + info_cat["e_K_mag"]**2)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Empirical relations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Mann+15 teff (Bp-Rp)
    teffs, e_teffs = params.compute_mann_2015_teff(
        info_cat["Bp-Rp"],
        relation="BP - RP")

    info_cat["teff_m15_bprp"] = teffs
    info_cat["e_teff_m15_bprp"] = e_teffs

    # Mann+15 teff (Bp-Rp, J-H)
    teffs, e_teffs = params.compute_mann_2015_teff(
        info_cat["Bp-Rp"], 
        j_h=info_cat["J-H"],
        relation="BP - RP, J - H")

    info_cat["teff_m15_bprp_jh"] = teffs
    info_cat["e_teff_m15_bprp_jh"] = e_teffs

    # Mann+19 radii
    radii, e_radii = params.compute_mann_2015_radii(info_cat["K_mag_abs"])
    info_cat["radii_m19"] = radii
    info_cat["e_radii_m19"] = e_radii

    # Mann+19 masses
    masses, e_masses = params.compute_mann_2019_masses(info_cat["K_mag_abs"]) 
    info_cat["mass_m19"] = masses
    info_cat["e_mass_m19"] = e_masses

    # Compute logg and e_log from Mann params
    logg, e_logg = params.compute_logg(masses, e_masses, radii, e_radii,)
    info_cat["logg_m19"] = logg
    info_cat["e_logg_m19"] = e_logg

    return info_cat


def load_exofop_toi_cat(
    toi_cat_path="data/exofop_tess_tois.csv",
    do_ctoi_merge=False,
    ctoi_cat_path="data/exofop_tess_ctois.csv",):
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
    tic_info = load_info_cat(remove_fp=True, only_observed=True)
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

    return toi_info[np.isin(toi_info["TIC"], tic_info["TIC"])]

