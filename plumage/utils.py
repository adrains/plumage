"""
"""
import numpy as np
import pandas as pd

def do_id_crossmatch(observations, catalogue):
    """Do an ID crossmatch and add the Gaia DR2 ID to observations.
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
        idx = np.argwhere(catalogue["2MASS_Source_ID_1"].values==ob_id)

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


def do_standard_crossmatch(catalogue):
    """
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
    """
    """
    royas = pd.read_csv("rojas-ayala_2012.tsv", sep="\t", header=1, 
                     skiprows=0, dtype={"source_id":str, "useful":bool})
    royas.set_index("source_id", inplace=True)
    royas = royas[[type(ii) == str for ii in royas.index.values]]
                        
    newton = pd.read_csv("newton_cpm_2014.tsv", sep="\t", header=1,  
                        skiprows=0, dtype={"source_id":str, "logg":np.float64, "useful":bool})
    newton.set_index("source_id", inplace=True)
    newton = newton[[type(ii) == str for ii in newton.index.values]]
                        
    mann = pd.read_csv("mann_constrain_2015.tsv", sep="\t", header=1, 
                        skiprows=0, dtype={"source_id":str, "useful":bool})
    mann.set_index("source_id", inplace=True)
    mann = mann[[type(ii) == str for ii in mann.index.values]] 

    interferometry = pd.read_csv("interferometry.tsv", sep="\t", header=1, 
                        skiprows=0, dtype={"source_id":str, "useful":bool})
    interferometry.set_index("source_id", inplace=True)
    interferometry = interferometry[[type(ii) == str for ii in interferometry.index.values]] 

    herczeg = pd.read_csv("data/herczeg_2014_standards_gaia.tsv", sep="\t", header=1, 
                        skiprows=0, dtype={"source_id":str, "useful":bool})
    herczeg.set_index("source_id", inplace=True)
    herczeg = herczeg[[type(ii) == str for ii in herczeg.index.values]]

    standards = [royas, newton, mann, interferometry, herczeg]

    return standards

