# PyWiFeS Reductions
WiFeS reductions are done using [a fork](https://github.com/marusazerjal/pipeline) of [PyWiFeS](https://github.com/PyWiFeS/pipelin) alongside stellar 1D spectra extraction code from the [PyWiFeS tools](https://github.com/PyWiFeS/tools) repository. Note that the steps below (and associated script names) reflect an old (but stable) version installed on the MSO servers.

## Raw --> Datacube
1. Change configBlue.py and configRed.py to have your raw and reduced data paths, plus the reduction steps you want to run.
2. Run generate_metadata_script_marusa.py as `python generate_metadata_script_marusa.py configRed/Blue.py YYYYMMDD` for both red and blue where YYYYMMDD is a folder within the root you specified in 1)
3. Check the metadata looks alright, will be in reduced_root/reduced_[br]/
4. Run reduction as `python reduce_marusa.py configRed/Blue.py YYYMMDD`. Just let it go and check back on it later in the day/the next day.

The main reduction steps we're interested in:
- p08 - Final spectra pre-flux or telluric correction (counts)
- p09 - Final flux corrected spectra without telluric correction (physical units)
- p10 - Final flux and telluric corrected spectra (physical units)

## Datacube --> 1D Spectra
For exposures with only the science target in the WiFeS IFU, reduction is simple and can be done with default parameters. For situations with a brighter (potentially saturated) companion, things get more complicated and these need to be considered on a case-by-case basis. 
1. Download https://github.com/PyWiFeS/tools and add to python path. 
2. Use the functions in [pywifes_utils.py](https://github.com/marusazerjal/pipeline/blob/master/utils/pywifes_utils.py), specifically `extract_stellar_spectra_fits_all` and `extract_stellar_spectra_fits`. You can do manual extraction, which corresponds to manual clicking or changing the box size, by changing the function parameters. This calls functions from `process_stellar` in the pywifes tools repo.
