"""Script to import MIKE spectra and collate into a single fits file.
"""
import plumage.spectra_mike as sm
import plumage.utils_mike as um

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Spectra location on disk
spectra_folder = "spectra/mike_MK_unnormalised/"

# Location to save diagnostic plots of extracted spectra
plot_folder = "spectra/mike_MK_unnormalised/diagnostics/"

# Whether to use the flat field 'normalised' spectra
normalise_by_flat = False

# Settings for saving the resulting fits file
label = "KM_noflat"
save_folder = "spectra/"

# -----------------------------------------------------------------------------
# Importing + Saving
# -----------------------------------------------------------------------------
# Import all spectra 
blue_dict, red_dict = sm.load_all_mike_fits(
    spectra_folder=spectra_folder,
    plot_folder=plot_folder,
    normalise_by_flat=normalise_by_flat,)

# Collate these spectra into combined DataFrames and arrays
obs_dict = sm.collate_mike_obs(blue_dict, red_dict)

# Store this as a fits file
um.save_fits_from_dict(obs_dict, label=label, path=save_folder)