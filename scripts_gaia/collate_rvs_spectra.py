"""Initial working script for APOGEE + Gaia RVS crossmatch.
"""
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.cm as cm
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from collections import Counter

N_RVS_PX = 2401
RVS_WAVE_MIN = 846
RVS_WAVE_MAX = 870
RVS_WAVE_DELTA = 0.01

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def read_single_rvs_csv(csv_fn):
    """Reads a single Gaia RVS CSV and returns the wavelengths, fluxes, sigmas,
    and an associated DataFrame.

    Parameters
    ----------
    csv_fn: str
        Filename of the csv to import. We expect these to be 'csv.gz' files.

    Returns
    -------
    wave: 1D float array
        Wavelength array for the RVS spectra.

    fluxes, e_fluxes: 2D float array
        Flux and uncertainty arrays for the RVS spectra, of shape 
        [N_star, N_RVS_PX].

    rvs_df: pandas DataFrame
        DataFrame containing RVS info with columns ['source_id', 'solution_id',
        'ra', 'dec', 'combined_transits', 'combined_ccds', 'deblended_ccds',
        'rvs_snr'], of length [N_star].
    """
    # Import Gaia RVS CSV, with columns: ['solution_id', 'ra', 'dec', 'flux',
    # 'flux_error', 'combined_transits', 'combined_ccds', 'deblended_ccds']
    rvs_df = pd.read_csv(
        csv_fn,
        compression="gzip",
        comment="#",
        dtype={"source_id":str})
    rvs_df.set_index("source_id", inplace=True)

    # Grab the number of stars
    n_star = len(rvs_df)

    # Initalise numpy flux and uncertainty arrays
    fluxes = np.full((n_star, N_RVS_PX), np.nan)
    e_fluxes = np.full((n_star, N_RVS_PX), np.nan)

    # Extract the fluxes and uncertainties from the DataFrame, star-by-star.
    # The fluxes and uncertainties are stored as a single string (each)
    # representation of a float list (including NaNs).
    for star_i in range(n_star):
        flux_str = rvs_df.iloc[star_i]["flux"][1:-1].split(",")
        fluxes[star_i] = np.array(flux_str).astype(np.float64)

        e_flux_str = rvs_df.iloc[star_i]["flux_error"][1:-1].split(",")
        e_fluxes[star_i] = np.array(e_flux_str).astype(np.float64)

    # Drop the now unnecessary str flux and error columns
    rvs_df.drop(columns=["flux", "flux_error"], inplace=True)

    # Compute median SNR column
    snr = np.nanmedian(fluxes/e_fluxes, axis=1)
    rvs_df["rvs_snr"] = snr

    # Construct wavelength scale
    wave = np.arange(RVS_WAVE_MIN, RVS_WAVE_MAX+RVS_WAVE_DELTA, RVS_WAVE_DELTA)

    return wave, fluxes, e_fluxes, rvs_df


def read_all_rvs_csv(rvs_folder, max_files_to_import=1E10):
    """Imports a folder of many Gaia RVS spectra CSVs files, imports them each
    separately via repeated calls to read_single_rvs_csv, then collates them
    together into a single DataFrame and set of wave/flux/sigma arrays.

    Parameters
    ----------
    rvs_folder: str
        Folder of Gaia RVS spectra in CSV form to import.

    Returns
    -------
    wave_1D: 1D float array
        Wavelength array of shape [n_px].

    fluxes_2D, sigmas_2D: 2D float array
        Gaia RVS flux and uncertainty arrays of shape [n_star, n_px].
    """
    # Locate all RVS files
    path_wildcard = os.path.join(rvs_folder, "*.csv.gz")
    all_csvs = glob(path_wildcard)
    all_csvs.sort()

    # Initialise interim lists to hold arrays and DataFrames from separate CSVs
    fluxes_list = []
    e_fluxes_list = []
    rvs_df_list = []

    desc = "Collating Gaia RVS csv files"

    for csv_i, csv_fn in enumerate(tqdm(all_csvs, leave=False, desc=desc)):
        wave_1D, fluxes, e_fluxes, rvs_df = read_single_rvs_csv(csv_fn)

        fluxes_list.append(fluxes)
        e_fluxes_list.append(e_fluxes)
        rvs_df_list.append(rvs_df)

        # For testing, allow us to break out after only importing N files.
        if csv_i > max_files_to_import:
            break

    # Collate all arrays and dataframes
    fluxes_2D = np.vstack(fluxes_list)
    sigmas_2D = np.vstack(e_fluxes_list)
    rvs_df = pd.concat(rvs_df_list)

    return wave_1D, fluxes_2D, sigmas_2D, rvs_df


def import_apogee_fits(apogee_fits_fn):
    """Imports an APOGEE data release fits file (e.g. DR17) as a pandas
    DataFrame. We drop all 2D columns with the exception of [X/H] and e_[X/H]
    which are added as individual 1D columns, and drop all observations without
    a valid Gaia EDR3 crossmatch.

    Parameters
    ----------
    apogee_fits_fn: str
        Path to APOGEE fits file.

    Returns
    -------
    apogee_df: pandas DataFrame
        DataFrame version of APOGEE Data Release fits file with integer index.
    """
    with fits.open(apogee_fits_fn, mode="readonly") as fits_file:
        # Import the main table as a fits Table
        apogee_tab = Table(fits_file[1].data)

        # Create mask to rule out multidimensional columns
        cols = [name for name in apogee_tab.colnames
                if len(apogee_tab[name].shape) <= 1]

        # Convert to pandas and use str IDs
        dtype_dict = {"GAIAEDR3_SOURCE_ID":str}
        apogee_df = apogee_tab[cols].to_pandas().astype(dtype=dtype_dict,)

        # Grab the element order for the [X/H] value and sigma arrays
        XX = fits_file[3].data["ELEM_SYMBOL"][0].tolist()

        # Create column names for new [X/H] and e_[X/H] columns
        XX_col = ["{}_H".format(xx) for xx in XX]
        e_XX_col = ["e_{}_H".format(xx) for xx in XX]

        # Grab [X/H] and e_[X/H] arrays from the pre-pandas fits Table
        X_H_values = apogee_tab["X_H"].data
        X_H_sigmas = apogee_tab["X_H_ERR"].data

        # Interleave [X/H] and e_[X/H] data + column names
        XX_comb = np.full((X_H_values.shape[0], X_H_values.shape[1]*2), np.nan)
        XX_comb[:, 0::2] = X_H_values
        XX_comb[:, 1::2] = X_H_sigmas

        col_names = [i for sublist in zip(XX_col, e_XX_col) for i in sublist]

        # Add [X/H] columns back in by DataFrame concatenation
        X_H_df = pd.DataFrame(columns=col_names, data=XX_comb)
        apogee_df = pd.concat([apogee_df, X_H_df], axis=1)

        # Drop any targets without a Gaia crossmatch, reset indexing
        has_source_id = apogee_df["GAIAEDR3_SOURCE_ID"] != "0"
        apogee_df = apogee_df[has_source_id].reset_index()

    return apogee_df


def remove_duplicates_from_apogee_dataframe(apogee_df,):
    """Removes duplicate observations from the APOGEE data release fits file
    imported via import_apogee_fits() by selecting retaining only the highest
    SNR exposure per target. The returned dataframe uses Gaia source_id as
    index.

    Parameters
    ----------
    apogee_df: pandas DataFrame
        DataFrame version of APOGEE Data Release fits file with integer index.

    Returns
    -------
    apogee_df_1x: pandas DataFrame
        As apogee_df, but now with only the highest SNR observation for each
        star and using Gaia source_id as the DataFrame index.
    """
    # Select only highest SNR row of repeat observations
    n_obs = apogee_df.shape[0]
    keep_mask = np.full(n_obs, False)
    id_checked = set()

    # Count duplicates in advance
    apogee_ids = apogee_df["APOGEE_ID"].values
    dup_id_counter = Counter(apogee_ids)
    has_dup = np.array([dup_id_counter[aid] > 1 for aid in apogee_ids])

    # Loop over all rows, skipping if we've already checked this ID, and
    # selecting only the highest SNR entry for each duplicate observation.
    desc = "Selecting highest-SNR of duplicate exposures"

    # Pre-select the duplicate ids, snrs, and indices for speed
    dup_apogee_ids = apogee_df["APOGEE_ID"].values[has_dup]
    dup_snrs = apogee_df["SNR"].values[has_dup]
    dup_indices = apogee_df.index.values[has_dup]

    for obs_i in tqdm(range(n_obs), desc=desc,leave=False):
        apogee_id = apogee_df.iloc[obs_i]["APOGEE_ID"]

        # If we've already checked this ID, continue
        if apogee_id in id_checked:
            continue

        # If there's only a single exposure, no masking required
        elif dup_id_counter[apogee_id] == 1:
            # Flag to be kept
            keep_mask[obs_i] = True

            # Add to the set
            id_checked.add(apogee_id)

        # Duplicate ID, masking required
        elif dup_id_counter[apogee_id] > 1:
            # Just operate on the subset that are confirmed duplicates
            has_id = dup_apogee_ids == apogee_id

            # Determine observation with maximum SNR
            dup_i_max_snr = np.argmax(dup_snrs[has_id])
            obs_i_max_snr = dup_indices[has_id][dup_i_max_snr]

            # Set only the maximum to be kept
            keep_mask[obs_i_max_snr] = True

            # Add to the set
            id_checked.add(apogee_id)

        # Shouldn't be able to get here?
        else:
            raise Exception("Something is wrong!")

    # Use the mask to select only one observation of each star
    apogee_df_1x = apogee_df[keep_mask].copy()

    # Update index to Gaia source_id and return
    apogee_df_1x.rename(
        columns={"GAIAEDR3_SOURCE_ID":"source_id"}, inplace=True)
    apogee_df_1x.set_index("source_id", inplace=True)

    return apogee_df_1x


def save_to_new_apogee_fits(
    wave_1D,
    fluxes_2D,
    sigmas_2D,
    dataframe,
    label="APOGEE_DR17",
    fn_base="rvs_spectra",
    path="spectra",):
    """Saves the collated Gaia RVS wavelengths, fluxes, and uncertainties to a
    single fits file, along with the combined crossmatch between Gaia and 
    APOGEE.

    The fits file is saved as <path>/<fn_base>_<label>.fits.

    Parameters
    ----------
    wave_1D: 1D float array
        Wavelength array of shape [n_px].

    fluxes_2D, sigmas_2D: 2D float array
        Gaia RVS flux and uncertainty arrays of shape [n_star, n_px].

    dataframe: pandas DataFrame
        DataFrame of Gaia + APOGEE information, of length [n_star].
    
    label: string, default: 'APOGEE_DR17'
        Unique label for the resulting fits file.
    
    fn_base: string, default: 'rvs_spectra'
        Base string of filename.

    path: strin, default: 'spectra'
        Path to save the fits file to.
    """
    # Intialise HDU List
    hdu = fits.HDUList()

    # Sanity checking
    assert wave_1D.size == sigmas_2D.shape[1]
    assert fluxes_2D.shape == sigmas_2D.shape
    assert fluxes_2D.shape[0] == dataframe.shape[0]

    # HDU 1: wavelength scale
    wave_img =  fits.PrimaryHDU(wave_1D)
    wave_img.header["EXTNAME"] = ("WAVE_1D", "Gaia RVS wavelength scale.")
    hdu.append(wave_img)

    # HDU 2: RVS flux values
    spec_img =  fits.PrimaryHDU(fluxes_2D)
    spec_img.header["EXTNAME"] = ("SPECTRA_RVS", "Gaia RVS fluxes.")
    hdu.append(spec_img)

    # HDU 3: RVS flux uncertainties
    e_spec_img =  fits.PrimaryHDU(sigmas_2D)
    e_spec_img.header["EXTNAME"] = ("SIGMAS_RVS", "Gaia RVS uncertainties.")
    hdu.append(e_spec_img)

    # HDU 4: table of Gaia + APOGEE information
    obs_tab = fits.BinTableHDU(Table.from_pandas(dataframe.reset_index()))
    obs_tab.header["EXTNAME"] = ("INFO_TAB", "Dataframe of Gaia/APOGEE info.")
    hdu.append(obs_tab)
    
    # Done, save
    save_path = os.path.join(path, "{}_{}.fits".format(fn_base, label))
    hdu.writeto(save_path, overwrite=True)

# -----------------------------------------------------------------------------
# Main body
# -----------------------------------------------------------------------------
# Import the spectra
csv_fn = "/Users/adamrains/data/RvsMeanSpectrum_000000-003111.csv.gz"
rvs_folder = "/Users/adamrains/data/rvs/"

print("Ingesting spectra...")
wave, fluxes, e_fluxes, rvs_df = read_all_rvs_csv(rvs_folder)

# -----------------------------------------------------------------------------
# Gaia crossmatch
# -----------------------------------------------------------------------------
# Import Gaia data, convert to pandas DataFrame
gaia_fits_fn = "/Users/adamrains/data/all_rvs-result.fits"

with fits.open(gaia_fits_fn, mode="readonly") as fits_file:
    gaia_tab = Table(fits_file["votable"].data)
    gaia_df = gaia_tab.to_pandas().astype(dtype={"source_id":str})
    gaia_df.set_index("source_id", inplace=True)

print("Crossmatching Gaia...")
# Crossmatch spectra with main Gaia catalogue
gaia_df_cm = rvs_df.join(other=gaia_df, on="source_id", rsuffix="_2")

# -----------------------------------------------------------------------------
# APOGEE import
# -----------------------------------------------------------------------------
# Import Gaia data, convert to pandas DataFrame
apogee_fits_fn = "/Users/adamrains/data/allStarLite-dr17-synspec_rev1.fits"

apogee_df = import_apogee_fits(apogee_fits_fn)

apogee_df_1x = remove_duplicates_from_apogee_dataframe(apogee_df)

# -----------------------------------------------------------------------------
# Crossmatch Gaia fits and APOGEE fits
# -----------------------------------------------------------------------------
print("Crossmatching APOGEE...")
# Crossmatch spectra with APOGEE catalogue
apogee_df_cm = gaia_df_cm.join(other=apogee_df_1x, on="source_id", rsuffix="_2")

# Apply quality cuts
has_good_plx = apogee_df_cm["parallax_over_error"].values > 20
has_apogee = ~np.isnan(apogee_df_cm["FE_H"].values)

passes_quality_cuts = np.logical_and(has_good_plx, has_apogee)

gaia_df_cut = apogee_df_cm[passes_quality_cuts]

# Apply to fluxes
fluxes_2D = fluxes[passes_quality_cuts,:]
sigmas_2D = e_fluxes[passes_quality_cuts,:]

# Save
save_to_new_apogee_fits(wave, fluxes_2D, sigmas_2D, gaia_df_cut)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
bp_rp_all = gaia_df_cut["bp_rp"].values
bp_rp_min = np.nanmin(bp_rp_all)
bp_rp_max = np.nanmax(bp_rp_all)
cmap = cm.get_cmap("magma")

plt.close("all")
fig_spec, axis_spec = plt.subplots(1, figsize=(18,8))
for star_i in range(len(gaia_df_cut)):
    if gaia_df_cut.iloc[star_i]["rvs_snr"] > 30:
        bp_rp = bp_rp_all[star_i]
        colour = cmap((bp_rp-bp_rp_min)/(bp_rp_max-bp_rp))
        axis_spec.plot(wave, fluxes[star_i], linewidth=0.1, c=colour)
        sc = axis_spec.scatter(0,1,c=bp_rp, cmap=cmap.reversed(), vmin=bp_rp_min, vmax=bp_rp_max)
axis_spec.set_xlabel("Wavelength (nm)")
axis_spec.set_xlim(RVS_WAVE_MIN-10*RVS_WAVE_DELTA,RVS_WAVE_MAX+10*RVS_WAVE_DELTA)

cb = fig_spec.colorbar(sc, ax=axis_spec, fraction=0.05)
cb.ax.set_title(r"$BP-RP$")

fig_spec.subplots_adjust(left=0.05, bottom=0.075, right=1.0, top=0.95,)

# ---------
dist = 1000 / gaia_df_cut["parallax"].values
e_dist = gaia_df_cut["parallax_error"].values
M_G = gaia_df_cut["phot_g_mean_mag"].values - 5*np.log10(dist/10)

Fe_Hs = gaia_df_cut["FE_H"].values

fig_cmd, axis_cmd = plt.subplots(1, figsize=(8,6))
sc = axis_cmd.scatter(x=bp_rp_all, y=M_G, c=Fe_Hs, cmap="cividis")

cb = fig_cmd.colorbar(sc, ax=axis_cmd,)
cb.ax.set_title("[Fe/H]")

axis_cmd.set_xlabel(r"$BP-RP$")
axis_cmd.set_ylabel(r"$M_G$")

axis_cmd.set_ylim(13, -5)
fig_cmd.tight_layout()


# ---------
is_ms = np.logical_and(dist < 300, bp_rp_all > 1.3)
fig_cmd, axis_ms = plt.subplots(ncols=2, figsize=(12,6))
axis_ms[0].hist(gaia_df_cut["phot_rp_mean_mag"].values[is_ms], 100)
axis_ms[1].hist(dist[is_ms], 100)


axis_ms[0].set_xlabel(r"$RP$")
axis_ms[1].set_xlabel(r"Distance (pc)")

fig_cmd.tight_layout()
