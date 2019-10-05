"""Script to process science spectra
"""
import numpy as np
import plumage.synthetic as synth
import plumage.spectra as spec
import plumage.plotting as pplt

# Load in science spectra
print("Importing science spectra...")
observations, spectra_b, spectra_r = spec.load_pkl_spectra(159) 

# Compute barycentric correction
print("Compute barycentric corrections...")
bcors = spec.compute_barycentric_correction(observations["ra"], 
                                            observations["dec"], 
                                            observations["mjd"], site="SSO")
observations["bcor"] = bcors

# Normalise science spectra
print("Normalise science spectra...")
spectra_b_norm = spec.normalise_spectra(spectra_b)
spectra_r_norm = spec.normalise_spectra(spectra_r)

# Make synthetic templates [requires IDL]
#ref_spec = synth.get_template_spectra(teffs, loggs, fehs)

# Load in template spectra
print("Load in synthetic templates...")
ref_params, ref_spec = synth.load_synthetic_templates(setting="R7000")  

# Normalise template spectra
print("Normalise synthetic templates...")
ref_spec_norm = spec.normalise_spectra(ref_spec)  

# Compute RVs
print("Compute RVs...")
rvs, teffs, fit_quality = spec.do_all_template_matches(spectra_r_norm, 
                                        observations, ref_params, 
                                        ref_spec_norm, print_diagnostics=True)

observations["teff_fit"] = teffs
observations["rv"] = rvs
observations["template_fit_quality"] = fit_quality

# Plot the spectra sorted by temperature
pplt.plot_teff_sorted_spectra(spectra_r_norm, observations)