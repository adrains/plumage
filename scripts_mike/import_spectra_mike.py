"""Script to import MIKE spectra and collate into a single fits file used for
all future operations: <path>/mike_spectra_<label>.fits
"""
import plumage.spectra_mike as sm
import plumage.utils_mike as um
import stannon.utils as su

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Import our settings object, which stores settings detailed in a YAML file.
mike_settings = "scripts_mike/mike_reduction_settings.yml"
ms = su.load_yaml_settings(mike_settings)

# -----------------------------------------------------------------------------
# Importing + Saving
# -----------------------------------------------------------------------------
# Import all spectra 
blue_dict, red_dict = sm.load_all_mike_fits(
    spectra_folder=ms.reduction_folder,
    plot_folder=ms.reduction_diagnostic_folder,
    normalise_by_flat=ms.normalise_multifits_by_flat,)

# Collate these spectra into combined DataFrames and arrays
obs_dict = sm.collate_mike_obs(blue_dict, red_dict)

# Store this as a fits file
um.save_fits_from_dict(obs_dict, label=ms.fits_label, path=ms.fits_folder)