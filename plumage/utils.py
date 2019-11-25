"""Utilities functions to assist.
"""
import os
import numpy as np
import pandas as pd

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

    for ob_id_i, ob_id in enumerate(ob_ids):
        # Gaia DR2
        idx = np.argwhere(catalogue["source_id"].values==ob_id)

        if len(idx) == 1:
            u_ids.append(catalogue.iloc[int(idx)]["source_id"])
            continue

        # 2MASS
        idx = np.argwhere(catalogue["2MASS_Source_ID"].values==ob_id)

        if len(idx) == 1:
            u_ids.append(catalogue.iloc[int(idx)]["source_id"])
            continue

        # HD
        idx = np.argwhere(catalogue["HD"].values==ob_id.replace(" ",""))

        if len(idx) == 1:
            u_ids.append(catalogue.iloc[int(idx)]["source_id"])
            continue

        # TOI
        idx = np.argwhere(catalogue["TOI"].values==ob_id.replace("TOI ", ""))

        if len(idx) == 1:
            u_ids.append(catalogue.iloc[int(idx)]["source_id"])
            continue

        # Bayer
        idx = np.argwhere(catalogue["bayer"].values==ob_id)

        if len(idx) == 1:
            u_ids.append(catalogue.iloc[int(idx)]["source_id"])
            continue

        # other
        idx = np.argwhere(catalogue["other"].values==ob_id)

        if len(idx) == 1:
            u_ids.append(catalogue.iloc[int(idx)]["source_id"])
            continue
        
        # If get to this point and no ID, put placeholder and print
        print("No ID match for #%i: %s" % (ob_id_i, ob_id))
        u_ids.append("")

    observations["uid"] = u_ids


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

    # Import Mann et al. 2015 Teff standards catalogue                    
    mann = pd.read_csv(os.path.join(curr_path, "mann_constrain_2015.tsv"), 
                       sep="\t", header=1, skiprows=0, 
                       dtype={"source_id":str, "useful":bool})
    mann.set_index("source_id", inplace=True)
    mann = mann[[type(ii) == str for ii in mann.index.values]] 
    mann = mann[mann["useful"]]

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

    standards = {"royas":royas, "newton":newton, "mann":mann, 
                 "interferometry":interferometry, "herczeg":herczeg}

    return standards


def consolidate_standards(standards, force_unique=False, 
    assign_default_uncertainties=False):
    """WARNING: force_unique is a temporary HACK
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

    if assign_default_uncertainties:
        std_params_all.loc[~np.isfinite(std_params_all["e_teff"]), "e_teff"] = 100
        std_params_all.loc[~np.isfinite(std_params_all["e_logg"]), "e_logg"] = 0.1
        std_params_all.loc[~np.isfinite(std_params_all["e_feh"]), "e_feh"] = 0.1

    return std_params_all

def prepare_training_set(observations, spectra_r, std_params_all, 
    do_wavelength_masking=True):
    """Need to prepare a list of labels corresponding to our science 
    observations. Easiest thing to do now is to just construct a new label
    dataframe with the same order as the observations. Then we won'd have any
    issues doing crossmatches, and we can worry about duplicates later.
    """
    import plumage.spectra as spec

    # First thing to do is to select all the observations that are standards
    is_std_mask = np.isin(observations["uid"], std_params_all["source_id"])
    std_observations = observations.copy().iloc[is_std_mask]
    std_spectra_r = spectra_r[is_std_mask]

    # Mask wavelengths
    if do_wavelength_masking:
        wl_mask = spec.make_wavelength_mask(std_spectra_r[0,0])
        dims = std_spectra_r.shape
        wl_mask = np.tile(wl_mask, dims[0]*dims[1]).reshape(dims)
        
        std_spectra_r = std_spectra_r[wl_mask]
        std_spectra_r = std_spectra_r.reshape(
            [dims[0], dims[1], int(len(std_spectra_r)/np.prod(dims[:2]))])

    # Do something with the duplicates
    pass

    # Initialise new label dataframe
    std_params = std_params_all.copy()
    std_params.drop(std_params_all.index, inplace=True)

    # Now need to go through this one row at a time and construct the 
    # corresponding label vector
    for ob_i, source_id in enumerate(std_observations["uid"]):
        # Determine labels
        idx = std_params_all[std_params_all["source_id"]==source_id].index[0]
        std_params = std_params.append(std_params_all.loc[idx])

    return std_observations, std_spectra_r, std_params