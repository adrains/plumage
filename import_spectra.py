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
 1) All 1D fits spectra in a single folder
 2) All 1D fits spectra sorted into subfolders (e.g. nights, science program)
    within a single base folder.
These are controlled by the variables 'spectra_folder' and 'include_subfolders'
respectively.

Finally, stars will be crossmatched to their science program and a unique ID
(typically Gaia DR2) based on the ID they were observed under (typically 
2MASS for young stars, and TOI for TESS targets). This catalogue is specified
through 'cat_file'
"""
import plumage.spectra as spec
import plumage.utils as utils

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Unique label of fits file of all spec
label =  "standard"

# Base folder of 1D fits spectra
spectra_folder = "spectra/standard"

# If 1D fits spectra are within additional subfolders (e.g. nights)
include_subfolders = False

# Folder to save the new fits file to
fits_save_path = "spectra"

# Crossmatch catalogue matching observed IDs to science program
cat_type = "csv"
cat_file = "data/all_2m3_star_ids.csv"

# -----------------------------------------------------------------------------
# Import spectra, clean, find bcor, and save
# -----------------------------------------------------------------------------
# Do initial import
print("Importing Spectra...")
observations, spectra_b, spectra_r = spec.load_all_spectra(
    spectra_folder=spectra_folder,
    include_subfolders=include_subfolders
)

# Clean spectra by setting negative or zero flux values + errors to np.nan
spec.clean_spectra(spectra_b)
spec.clean_spectra(spectra_r)

# Compute barycentric correction
print("Computing barycentric velocities...")
observations["bcor"] = spec.compute_barycentric_correction(
    observations["ra"], 
    observations["dec"], 
    observations["mjd"], 
    observations["exp_time"],
    "Siding Spring Observatory",
    )

# Crossmatch IDs and science programs for targets
catalogue = utils.load_crossmatch_catalogue(cat_type, cat_file)
utils.do_id_crossmatch(observations, catalogue)

# Save spectra and observation table
utils.save_fits(spectra_b, spectra_r, observations, label, path=fits_save_path)