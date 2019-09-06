"""
Observed Sample
 1 - Write a script to grab the subset of the FW input catalogue that we've 
     observed, which gives Gaia DR2, 2MASS, and ALLWISE information for all
     stars. Have a subset of columns that are useful.

Synthetic Sample
 1 - Generate a synthetic spectra equivalents for all standards
 2 - Generate a synthetic sample of young star candidates. Initially this can
     probably just ignore the effects of H-alpha emission and veiling for 
     simplicity, which for the former we can probably just mask out anyway.

Data reduction steps
 1 - Organise data/reduced data folders
 2 - Reduce data, correcting for tellurics and photometrically calibrating 
     where possible.
 3 - Figure out average SNR for each target to enable only working initially
     with the best possible sample.
 4 - RV correct all spectra and move to rest frame (using barycentric 
     correction, and presumably template matching with synthetic spectra?)

Data preparation
 1 - Normalise all spectra (either continuum normalisation, or something 
     consistent but arbitrary).
 2 - Create a mask for regions affected by tellurics, veiling, or emission due
     to accretion.

Initial (coarse) parameter estimation
 1 - Use colour relations to estimate temperatures for all stars.
 2 - Use a grid of synthetic spectra for stars with Teff > ~3500 K or so to 
     get rough parameters to use as a reference later.
  
Standard preparation
 1 - Fill in missing values for standards, or have Cannon implementation that
     can account for missing parameters.
 2 - Artifically broaden to get access to vsini dimension.
 3 - 
 
Cannon Model
 1 - Put all spectra onto a common wavelength grid
 2 - Train model on standards
 3 - Vet accuracy using leave one out approach with Cannon predicting known
     parameters, remove any standards not representative.
     
Results analysis
 1 - Accuracy in recovering labels
 2 - Temperature/gravity/metallicity sensitive spectral regions
 3 - Predicted temperatures vs empirical relations
 
 
--------------
Code structure
--------------
data
|--> standard catalogue tsvs
plumage
|--> pcannon.py - code for interfacing with the Cannon
|--> photometry.py - tools to deal with photometry
|--> spectra.py - tools for working with real spectra
|--> standards.py - tools for working with the stellar standard information
|--> synthetic.py - tools for dealing with synthetic spectra
|--> paper.py
|--> plotting.py
|--> utils.py
|--> 
"""