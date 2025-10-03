"""Script to import extracted blue and red 1D science spectra plus 
corresponding observational information extracted from fits headers (e.g.
exposure times, MJDs, airmass). Spectra are cleaned for any non-physical pixels
(i.e. negative or zero flux), and barycentric velocities are computed.

It is highly recommended that all spectra should be at least fluxed, thus 
ensuring that they all have physical/black body shapes for simplicity later 
when fitting to synthetic templates/spectra for RVs or parameters.

Spectra are saved to a multi-extension fits file within 'fits_save_path' that
has the following extensions:
 HDU 0: 1D blue wavelength scale
 HDU 1: 2D blue band flux [star, wl]
 HDU 2: 2D blue band flux uncertainties [star, wl]
 HDU 3: 1D red wavelength scale
 HDU 4: 2D red band flux [star, wl]
 HDU 5: 2D red band flux uncertainties [star, wl]
 HDU 6: table of observational information

Spectra can be loaded from one of two directory structures:
 1) All 1D fits spectra in a (set of) single folder(s)
 2) All 1D fits spectra sorted into subfolders (e.g. nights, science program)
    within a single base folder.
These are controlled by the variables 'spectra_folder' and 'include_subfolders'
respectively.

Finally, stars will be crossmatched to their science program and a unique ID
(typically Gaia DR2) based on the ID they were observed under (typically 
2MASS for young stars, and TOI for TESS targets). This catalogue is specified
through 'cat_file'
"""
import plumage.spectra as ps
import plumage.utils as pu

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Unique label of fits file of all spec
label =  "planet_mk"

# Base folder of 1D fits spectra--each folder is globbed separately.
spectra_folder = ["spectra/standard", "spectra/tess", "spectra/standard_other"]

# If 1D fits spectra are within additional subfolders (e.g. nights)
include_subfolders = False

# Folder to save the new fits file to
fits_save_path = "spectra"

# Crossmatch method #1: csv crossmatch file, used for Rains+2020 and 2024
cat_type = "csv"
cat_file = "data/all_2m3_star_ids.csv"
do_crossmatch_old = False

# Crossmatch method #2: new format where we use our TSV of stellar Gaia + 
# 2MASS info for the crossmatch, enabling us to remove stars we won't use.
star_info_fn = "data/all_known_and_candidate_hosts.tsv"
do_crossmatch_modern = True

# Whether to import the spectra in units of counts, and then flux ourselves
use_counts_ext_and_flux = True

# Whether to enforce only one of each star by removing the lower SNR ob
do_remove_duplicates = True

# -----------------------------------------------------------------------------
# Import and clean spectra, compute bcor
# -----------------------------------------------------------------------------
# Do initial import
print("Importing Spectra...")
observations, spectra_b, spectra_r = ps.load_all_spectra(
    spectra_folder=spectra_folder,
    include_subfolders=include_subfolders,
    use_counts_ext_and_flux=use_counts_ext_and_flux,
    do_remove_duplicates=do_remove_duplicates,)

# Clean spectra by setting negative or zero flux values + errors to np.nan
ps.clean_spectra(spectra_b, do_set_nan_for_neg_px=True)
ps.clean_spectra(spectra_r, do_set_nan_for_neg_px=True)

# Compute barycentric correction
print("\nComputing barycentric velocities...")
observations["bcor"] = ps.compute_barycentric_correction(
    observations["ra"], 
    observations["dec"], 
    observations["mjd"], 
    observations["exp_time"],
    "Siding Spring Observatory",)

# -----------------------------------------------------------------------------
# Crossmatch
# -----------------------------------------------------------------------------
print("\nCrossmatching...")

# Option #1: Rains+21 and Rains+24 crossmatch
if do_crossmatch_old:
    catalogue = pu.load_crossmatch_catalogue(cat_type, cat_file)
    pu.do_id_crossmatch(observations, catalogue)

# Option #2: modern crossmatch that drops irrelevant stars, no science programs
elif do_crossmatch_modern:
    # Import the dataframe
    successful_crossmatch, star_info = pu.do_id_crossmatch_modern(
        observations, star_info_fn,)

    # Apply mask so that we only carry forward those stars we're interested in.
    spectra_b = spectra_b[successful_crossmatch]
    spectra_r = spectra_r[successful_crossmatch]
    observations = observations[successful_crossmatch]

# Option #3: basic, assumes that ID is already equal to the source_id
else:
    observations["source_id"] = observations["id"].values
    observations.set_index("source_id", inplace=True)

# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
# Save spectra and observation table
print("\tSaving to {}/spectra_{}.fits".format(fits_save_path, label))
pu.save_fits(spectra_b, spectra_r, observations, label, path=fits_save_path)