"""Script to download and save complete lightcurves for all TOIs
"""
import numpy as np
import plumage.utils as utils
import plumage.transits as transit

# Import catalogue of TESS info
tess_info = utils.load_info_cat(remove_fp=True, only_observed=True)

# Load ExoFOP info
efi = utils.load_exofop_toi_cat()

all_lightcurves = []
unmatched_tics = []

for star_i, star in tess_info.iterrows():
    print("\nDownloading lightcurve {}/{} for TIC {}".format(
        star_i+1, len(tess_info, star["TIC"])))
    
    # Download light curve for current star
    try:
        lc = transit.download_lc_all_sectors(
            star["TIC"], 
            star["source_id"],
            save_fits=True, 
            save_path="lightcurves")

    # If we failed to match either TIC or Gaia ID, log and continue
    except ValueError:
        print("No match for TIC {}".format(star["TIC"]))
        unmatched_tics.append(star["TIC"])
        lc = None
        
    all_lightcurves.append(lc)