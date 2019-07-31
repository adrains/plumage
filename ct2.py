from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def generate_standards(n_standards, teff_low=3000, teff_high=5000, 
                       feh_low=-0.5, feh_high=0.5, logg_scatter=0.025):
    """
    """
    teffs = np.random.randint(teff_low, teff_high, n_standards)
    
    fehs = np.random.uniform(feh_low, feh_high, n_standards)
    
    loggs = standard_logg(teffs) + np.random.normal(0, logg_scatter, n_standards)
    
    return teffs, loggs, fehs


def generate_young_stars(n_standards, teff_low=3000, teff_high=5000, 
                       feh_low=-0.5, feh_high=0.5, logg_offset=0.25, 
                       logg_scatter=0.05):
    """
    """
    teffs = np.random.randint(teff_low, teff_high, n_standards)
    
    fehs = np.random.uniform(feh_low, feh_high, n_standards)
    
    loggs = standard_logg(teffs) - np.random.normal(logg_offset, logg_scatter, 
                                                    n_standards)
    
    return teffs, loggs, fehs
    

def standard_logg(teffs):
    """Quick and dirty cool star logg calculator
    """
    loggs = teffs * -7 / 20000 + 6.15
    
    return loggs
    

def plot_combined_hr(std_teffs, std_loggs, std_fehs, ys_teffs, ys_loggs, ys_fehs):
    """
    """
    teffs = np.concatenate((std_teffs, ys_teffs))
    loggs = np.concatenate((std_loggs, ys_loggs))
    fehs = np.concatenate((std_fehs, ys_fehs))
    
    
    plt.close("all")
    sct = plt.scatter(teffs, loggs, c=fehs, marker="o")
    cb = plt.colorbar(sct)
    cb.set_label("[Fe/H]")
    
    plt.ylim([5.1,4.3])
    plt.xlim([5100,2900])
    plt.xlabel(r"T$_{\rm eff}$")
    plt.ylabel("logg")
    
    
def train_cannon(spectra):
    """
    """
    # Format everything appropriately
    teffs = np.concatenate((std_teffs, ys_teffs))
    loggs = np.concatenate((std_loggs, ys_loggs))
    fehs = np.concatenate((std_fehs, ys_fehs))
    
    inv_var = np.ones_like(spectra)
    
    training_set = pd.DataFrame(data=np.vstack((teffs, loggs, fehs)).T, 
                                columns=["teff","logg","feh"])
    
    # Make the model                            
    vectorizer = tc.vectorizer.PolynomialVectorizer(("teff", "logg", "feh"), 2)
    model = tc.CannonModel(training_set, spectra, inv_var, vectorizer=vectorizer)
    
    # Train the model
    model.train()
    
    # Test the model
    labels, cov, meta = model.test(spectra, np.ones_like(spectra))
    errors = np.abs(labels - training_set.values).std(axis=1)