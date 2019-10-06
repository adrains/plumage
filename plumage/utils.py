"""
"""
import numpy as np

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

