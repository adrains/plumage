"""Script to download and save complete lightcurves for all TOIs
"""
import numpy as np
import plumage.utils as utils
import plumage.transits as transit

# Binning factor. Note that we're assuming a minimum cadence of 2 mins, so 
# extended mission data will get binned to at least that level
bin_fac = 2

# Import catalogue of TESS info
tess_info = utils.load_info_cat(remove_fp=True, only_observed=True)

# Load ExoFOP info
efi = utils.load_exofop_toi_cat()

all_lightcurves = []
unmatched_tics = []
sector_list = []

for star_i, (source_id, star) in enumerate(tess_info.iterrows()):
    print("\nDownloading lightcurve {}/{} for TIC {}".format(
        star_i+1, len(tess_info), star["TIC"]))
    
    # Download light curve for current star
    try:
        lc, sectors = transit.download_lc_all_sectors(
            star["TIC"], 
            source_id,
            save_fits=True, 
            save_path="lightcurves",
            bin_fac=bin_fac,)

        # Save sector
        sector_list.append([star["TIC"], transit.format_sectors(sectors)])

    # If we failed to match either TIC or Gaia ID, log and continue
    except ValueError:
        print("No match for TIC {}".format(star["TIC"]))
        unmatched_tics.append(star["TIC"])
        lc = None
        sector_list.append([star["TIC"], ""])

    all_lightcurves.append(lc)

# Save list of sectors to disk
np.savetxt(
    "lightcurves/tess_lc_sectors_binning_x{}.tsv".format(bin_fac),
    np.asarray(sector_list).astype(str),
    fmt="%s", 
    delimiter="\t")