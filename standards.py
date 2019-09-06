#from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import Angle

# Load in 
royas = pd.read_csv("rojas-ayala_2012.tsv", sep="\t", header=1, 
                     skiprows=0, dtype={"source_id":str, "useful":bool})
royas.set_index("source_id", inplace=True)
royas = royas[[type(ii) == str for ii in royas.index.values]]
                     
newton = pd.read_csv("newton_cpm_2014.tsv", sep="\t", header=1,  
                     skiprows=0, dtype={"source_id":str, "logg":np.float64, "useful":bool})
newton.set_index("source_id", inplace=True)
newton = newton[[type(ii) == str for ii in newton.index.values]]
                     
mann = pd.read_csv("mann_constrain_2015.tsv", sep="\t", header=1, 
                     skiprows=0, dtype={"source_id":str, "useful":bool})
mann.set_index("source_id", inplace=True)
mann = mann[[type(ii) == str for ii in mann.index.values]] 

interferometry = pd.read_csv("interferometry.tsv", sep="\t", header=1, 
                     skiprows=0, dtype={"source_id":str, "useful":bool})
interferometry.set_index("source_id", inplace=True)
interferometry = interferometry[[type(ii) == str for ii in interferometry.index.values]] 

herczeg = pd.read_csv("data/herczeg_2014_standards_gaia.tsv", sep="\t", header=1, 
                     skiprows=0, dtype={"source_id":str, "useful":bool})
herczeg.set_index("source_id", inplace=True)
herczeg = herczeg[[type(ii) == str for ii in herczeg.index.values]]

tess = pd.read_csv("data/tess_wifes.tsv", sep="\t", header=1, 
                     skiprows=0, dtype={"source_id":str, "useful":bool})
tess.set_index("source_id", inplace=True)
tess = tess[[type(ii) == str for ii in tess.index.values]] 
 

# Decide on limits, and mask
faint_mag = 10       # This gives ~15.6x the SNR of our faintest echelle target
bright_mag = 0
ra_min = 0 #10 * 15    
ra_max = 24 #5 * 15

def mask_catalogue(catalogue, faint_mag, bright_mag, ra_min, ra_max):
    """
    """
    # Make magnitude and quality mask
    mag_mask = np.logical_and(catalogue["Gmag"] < faint_mag,
                              catalogue["Gmag"] > bright_mag)  
    
    # Make coordinate mask
    ra_mask = np.logical_or(catalogue["ra"] >= ra_min, 
                             catalogue["ra"] <= ra_max)
                             
    mask = np.logical_and(mag_mask, ra_mask)
    
    final_mask = np.logical_and(mask, catalogue["useful"])
    
    return catalogue[final_mask].copy()

royas_masked = mask_catalogue(royas, faint_mag, bright_mag, ra_min, ra_max)
newton_masked = mask_catalogue(newton, faint_mag, bright_mag, ra_min, ra_max)
mann_masked = mask_catalogue(mann, faint_mag, bright_mag, ra_min, ra_max)
int_masked = mask_catalogue(interferometry, faint_mag, bright_mag, ra_min, ra_max)

unique_stars = set(np.hstack([royas_masked.index.values, 
                              newton_masked.index.values, 
                              mann_masked.index.values,
                              int_masked.index.values]))
                              
fehs = list(set(np.hstack([royas_masked["feh"].values, 
                              newton_masked["feh"].values, 
                              mann_masked["feh"].values,
                              int_masked["feh"].values])))

norm = plt.Normalize(np.nanmin(fehs), np.nanmax(fehs))
                              
print("Unique stars observable: %i" % len(unique_stars))
print("Total time for 10 min exp + 5 min overhead: %0.2f hr" 
      % (len(unique_stars)*0.25))

# Plot
plt.close("all")
plt.scatter(royas_masked["teff"], royas_masked["logg"], c=royas_masked["feh"], 
            marker="o", label="Royas-Ayala+2012", norm=norm)    
plt.scatter(newton_masked["teff"], newton_masked["logg"], norm=norm,
            c=newton_masked["feh"], marker="s", label="Newton+2014")   
plt.scatter(mann_masked["teff"], mann_masked["logg"], c=mann_masked["feh"], 
            marker="+", label="Mann+2015")  
plt.scatter(int_masked["teff"], int_masked["logg"], c=int_masked["feh"], 
            marker="^", label="Interferometry", norm=norm) 
plt.scatter(herczeg["teff"], herczeg["logg"], c=herczeg["feh"], 
            marker="*", label="Herczeg+14", norm=norm)  
cb = plt.colorbar()
cb.set_label("[Fe/H]")
plt.ylim([5.2, 0])
plt.xlim([7300,2500])
plt.xlabel(r"T$_{\rm eff}$ (K)")
plt.ylabel("logg")
plt.legend(loc="best")  



plt.figure()
norm = plt.Normalize(np.nanmin(fehs), np.nanmax(fehs))
coord_ax = plt.subplot(projection="aitoff")
rscat = coord_ax.scatter((royas_masked["ra"]-180)*np.pi/180, 
                          royas_masked["dec"]*np.pi/180, c=royas_masked["feh"], 
                          marker="o", label="Royas-Ayala+2012", norm=norm)    
nscat = coord_ax.scatter((newton_masked["ra"]-180)*np.pi/180, 
                          newton_masked["dec"]*np.pi/180, 
                          c=newton_masked["feh"], marker="s", 
                          label="Newton+2014", norm=norm)   
mscat = coord_ax.scatter((mann_masked["ra"]-180)*np.pi/180, 
                          mann_masked["dec"]*np.pi/180, c=mann_masked["feh"], 
                          marker="+", label="Mann+2015", norm=norm) 
iscat = coord_ax.scatter((int_masked["ra"]-180)*np.pi/180, 
                          int_masked["dec"]*np.pi/180, c=int_masked["feh"], 
                          marker="^", label="Interferometry", norm=norm) 
iscat = coord_ax.scatter((herczeg["ra"]-180)*np.pi/180, 
                          herczeg["dec"]*np.pi/180, c=herczeg["feh"], 
                          marker="*", label="Herczeg+14", norm=norm)   
coord_ax.set_title("On-sky positions")
coord_ax.set_xlabel("RA")
coord_ax.set_ylabel("DEC")
coord_ax.legend(loc="best")
coord_ax.grid()
cb = plt.colorbar(nscat, ax=coord_ax)
cb.set_label("[Fe/H]")


"""
plt.colorbar()
#plt.ylim([5.7, 4.6])
plt.xlabel(r"RA")
plt.ylabel("DEC")
plt.legend(loc="best")  
          """
# -----------------------------------------------------------------------------
# Plot CMD
# ----------------------------------------------------------------------------- 
ys_candidates = pd.read_csv("gaia_ys_candidates.tsv", sep="\t", header=0,
                            dtype={"source_id":str})  

royas_masked["dist"] = 1000 / royas_masked["plx"]  
royas_masked["Gmag_abs"] = royas_masked["Gmag"] - 5*np.log10(royas_masked["dist"]/10)  

newton_masked["dist"] = 1000 / newton_masked["plx"]  
newton_masked["Gmag_abs"] = newton_masked["Gmag"] - 5*np.log10(newton_masked["dist"]/10) 

mann_masked["dist"] = 1000 / mann_masked["plx"]  
mann_masked["Gmag_abs"] = mann_masked["Gmag"] - 5*np.log10(mann_masked["dist"]/10)    

int_masked["dist"] = 1000 / int_masked["plx"]  
int_masked["Gmag_abs"] = int_masked["Gmag"] - 5*np.log10(int_masked["dist"]/10)  

herczeg["dist"] = 1000 / herczeg["plx"]  
herczeg["Gmag_abs"] = herczeg["Gmag"] - 5*np.log10(herczeg["dist"]/10)  

cats = [royas_masked, newton_masked, mann_masked, int_masked, herczeg, ys_candidates] 
labels = ["Royas-Ayala+2012", "Newton+2014", "Mann+2015", "Interferometry", 
          "Herczeg+14", "YS Candidates"]
markers = ["o", "s", "+", "^", "*", "1"]

plt.figure()

for cat_i in range(0, len(cats)):
    print(labels[cat_i])
    plt.plot(cats[cat_i]["BP-RP"], cats[cat_i]["Gmag_abs"], markers[cat_i],
             label=labels[cat_i])
    print(labels[cat_i])
    if labels[cat_i] != "YS Candidates":
        for star_i in range(0, len(cats[cat_i])):
            txt = "[%s, %s]" % (cats[cat_i].iloc[star_i]["teff"],
                                cats[cat_i].iloc[star_i]["logg"])
            plt.text(cats[cat_i].iloc[star_i]["BP-RP"], 
                     cats[cat_i].iloc[star_i]["Gmag_abs"], txt, fontsize="xx-small")

plt.xlabel("Bp-Rp")
plt.ylabel("Gmag abs")
plt.legend(loc="best")
plt.ylim([14,-2])
plt.gcf().set_size_inches(16, 9)
#plt.save("plots/2.3m_ys_cmd.pdf")

#plt.show()

# -----------------------------------------------------------------------------
# Merge target lists and save to create list to be observed
# -----------------------------------------------------------------------------
def fmt_hour(ra):
    angle = Angle(ra, unit="degree")
    return angle.to_string(unit="hour", decimal=False, sep=" ")

def fmt_deg(dec):
    angle = Angle(dec, unit="degree")
    return angle.to_string(decimal=False, sep=" ") 
    
cols = ['ID', 'ra', 'dec', 'plx', 'pm_ra', 'pm_dec', 'Gmag', 'BPmag', 'RPmag', 
        'BP-RP', 'teff', 'logg', 'feh', 'standard_type']
           
standards = pd.concat([royas_masked[cols], newton_masked[cols], mann_masked[cols], 
                       int_masked[cols]])
#standards = herczeg[herczeg["Gmag"] > 11.5][cols].copy()
standards = tess[tess["useful"]][cols].copy()
standards = standards[~standards.index.duplicated(keep="first")].copy()
standards.sort_values("ra", inplace=True)

standards["ra_hr"] = standards["ra"] / 15
standards["exp_time"] = 600
standards["observed"] = ""
standards["instrument"] = "echelle"
standards["comments"] = ""
standards["taros_input"] = ['"%s" %s %s ! G=%0.2f' 
                             % (standards.iloc[ii].name, 
                                fmt_hour(standards.iloc[ii]["ra"]),
                                fmt_deg(standards.iloc[ii]["dec"]),
                                standards.iloc[ii]["Gmag"]) 
                            for ii in range(0, len(standards))]
#standards.to_csv("hstandards.tsv",  sep="\t")
standards.to_csv("tstandards.tsv",  sep="\t")
#standards.to_csv("standards.tsv",  sep="\t")