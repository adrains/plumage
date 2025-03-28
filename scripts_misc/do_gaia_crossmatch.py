"""Basic script to take a list of IDs and take the Gaia DR3 ID listed on SIMBAD
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from astroquery.simbad import Simbad

# File to import as DataFrame, and ID column to search
fn = "data/t15_2mass.csv"
index = "ID"
df = pd.read_csv(fn, sep="\t", index_col=index,)

star_ids = df.index
gaia_ids = np.full_like(star_ids, "")

desc = "Finding Gaia IDs"

for star_i, star_id in enumerate(tqdm(star_ids, desc=desc, leave=False)):
    # HACK to avoid a failure if the connection drops.
    try:
        simbad_ids = Simbad.query_objectids(star_id)["ID"].tolist()
    except:
        simbad_ids = Simbad.query_objectids(star_id)["ID"].tolist()
    is_gaia_dr3 = ["Gaia DR3" in sid for sid in simbad_ids]
    has_gaia_dr3 = np.any(is_gaia_dr3)

    if has_gaia_dr3:
        assert np.sum(is_gaia_dr3) == 1
        gaia_dr3_id_i = int(np.argwhere(is_gaia_dr3))
        gaia_dr3_id = simbad_ids[gaia_dr3_id_i].replace("Gaia DR3 ", "")
    else:
        gaia_dr3_id = ""

    gaia_ids[star_i] = gaia_dr3_id

df["source_id_dr3"] = gaia_ids
df.to_csv(fn, sep="\t")