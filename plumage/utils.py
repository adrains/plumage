"""Utilities functions to assist.
"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

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
    u_ids = []
    program = []

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
                u_ids.append(catalogue.iloc[int(idx)]["source_id"])
                program.append(catalogue.iloc[int(idx)]["program"])
                id_found = True
                break
        
        
        # If get to this point and no ID/program, put placeholder and print
        if not id_found:
            print("No ID match for #%i: %s" % (ob_id_i, ob_id))
            u_ids.append("")
            program.append("")

    observations["uid"] = u_ids
    observations["program"] = program


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
                                dtype={"Gaia ID":str, "TOI":str, 
                                "2MASS_Source_ID":str, "subset":str},
                                na_values=[], keep_default_na=False)
        catalogue.rename(columns={"Gaia ID":"source_id"}, inplace=True)

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
        
        sources.extend([key]*len(standards[key]))

    data = (ids, teffs, e_teffs, loggs, e_loggs, fehs, e_fehs, sources)
    columns = ["ids", "teffs", "e_teffs", "loggs", "e_loggs", "fehs", "e_fehs", 
               "sources"]
    std_params_all = pd.DataFrame(
        {"source_id": ids,
         "teff": teffs,
         "e_teff": e_teffs,
         "logg": loggs,
         "e_logg": e_loggs,
         "feh": fehs,
         "e_feh": e_fehs,
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


def save_observations_fits(observations, label):
    """Save observations table
    """
    save_path = os.path.join("spectra", "observations_{}.fits".format(label))
    obs_table = Table.from_pandas(observations)
    obs_table.write(save_path, format="fits", overwrite=True)


def load_observations_fits(label):
    """Load in the saved observation log from a fits file.
    """
    load_path = os.path.join("spectra", "observations_{}.fits".format(label))
    obs_tab = Table(fits.open(load_path)[1].data)
    obs_pd = obs_tab.to_pandas()

    return obs_pd


def save_spectra_fits(spectra, band, label):
    """Save spectra as a fits file, with a different HDU table per star
    """
    save_path = os.path.join("spectra", 
                             "spectra_{}_{}.fits".format(label, band))
    
    hdu = fits.HDUList()

    for spectrum in spectra:
        spec_tab =  Table(spectrum.T, names=["wave", "spec", "e_spec"])
        hdu.append(fits.BinTableHDU(spec_tab))

    hdu.writeto(save_path, overwrite=True)


def load_spectra_fits(band, label):
    """Load in the spectra from a fits file.
    """
    load_path = os.path.join("spectra", 
                             "spectra_{}_{}.fits".format(label, band))

    fits_file = fits.open(load_path)
    spectra = []

    for hdu_i in range(1, len(fits_file)):
        # Convert to numpy array
        spec_rec = fits_file[hdu_i].data
        spec_array = np.asarray(spec_rec).view((spec_rec.dtype[1], 3))
        spectra.append(spec_array.T)

    return np.stack(spectra)