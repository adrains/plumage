"""
"""
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
# Ensure the plotting folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "paper"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# -----------------------------------------------------------------------------
# Making tables
# ----------------------------------------------------------------------------- 
def make_table_final_results():
    """Make the final results table to display the angular diameters and 
    derived fundamental parameters.
    """
    # Load in the TESS target info (TODO: function for this)
    tess_info = pd.read_csv("data/tess_info.tsv", sep="\t")    
    exp_scale = -8
    
    columns = OrderedDict([
        ("TIC", ""),
        (r"$T_{\rm eff}$", "(K)"),
        (r"$\log g", "(K)"),
        (r"[Fe/H]", "(K)"),
        (r"$f_{\rm bol}$", 
         r"(10$^{%i}\,$ergs s$^{-1}$ cm $^{-2}$)" % exp_scale),
        (r"$L$", "($L_\odot$)"),
        (r"$\theta_{\rm LD}$", "(mas)"),
        (r"$R$", "($R_\odot$)"), 
        (r"$M$", "($M_\odot$)"), 
    ])
                           
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    # Populate the table for every science target
    for star_i, star in tess_info.iterrows():
        
        # Only continue if both observed and not FP
        if star["observed"] == "yes" and star["TFOPWG Disposition"] == "FP":
            continue
        
        # Placeholder: TODO
        """

        table_row = ""
        
        # Step through column by column
        table_row += "%s & " % rutils.format_id(row["Primary"])
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["udd_final"], row["e_udd_final"])
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["ldd_final"], row["e_ldd_final"])
        table_row += r"%0.3f $\pm$ %0.3f &" % (row["r_star_final"], row["e_r_star_final"])
        
        # For fbol representation, split mantissa and exponent
        table_row += r"%5.1f $\pm$ %0.1f &" % (row["f_bol_final"] / 10**exp_scale, 
                                               row["e_f_bol_final"] / 10**exp_scale)
        table_row += r"%0.0f $\pm$ %0.0f & " % (row["teff_final"], row["e_teff_final"])
        table_row += r"%0.2f $\pm$ %0.2f " % (row["L_star_final"], row["e_L_star_final"])
        
        table_rows.append(table_row + r"\\")
        """
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_final_results.tex", table_1, fmt="%s")

        

def make_table_targets():
    """Make the table to summarise the target information.
    """
    # Load in the TESS target info (TODO: function for this)
    tess_info = pd.read_csv("data/tess_info.tsv", sep="\t")    
    exp_scale = -8
    
    columns = OrderedDict([
        ("TIC", ""),
        ("TOI", ""),
        ("2MASS", ""),
        ("Gaia DR2", ""),
        ("RA$^a$", "(hh mm ss.ss)"),
        ("DEC$^a$", "(dd mm ss.ss)"),
        ("$G$", "(mag)"), 
        ("${B_p-R_p}$", "(mag)"),
        ("Plx$^a$", "(mas)"),
        ("ruwe$^a$", ""),
        (r"N$_{\rm pc}$", "")
    ])      
    
    table_rows = []
    
    # Construct the header of the table
    table_rows.append("\\begin{landscape}")
    table_rows.append("\\begin{table}")
    table_rows.append("\\centering")
    table_rows.append("\\caption{Science targets}")
    table_rows.append("\\label{tab:science_targets}")
    
    table_rows.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    table_rows.append("\hline")
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    table_rows.append("\hline")
    
    ref_i = 0
    references = []
    
    # Populate the table for every science target
    for star_i, star in tess_info.iterrows():
        table_row = ""
        
        # Only continue if both observed and not FP
        if star["observed"] == "yes" and star["TFOPWG Disposition"] == "FP":
            continue
        
        # Format RA and DEC
        ra_hr = np.floor(star["ra"] / 15)
        ra_min = np.floor((star["ra"] / 15 - ra_hr) * 60)
        ra_sec = ((star["ra"] / 15 - ra_hr) * 60 - ra_min) * 60
        ra = "%02i %02i %05.2f" % (ra_hr, ra_min, ra_sec)
        
        dec_deg = np.floor(star["dec"])
        dec_min = np.floor((star["dec"] - dec_deg) * 60)
        dec_sec = ((star["dec"] - dec_deg) * 60 - dec_min) * 60
        dec = "%02i %02i %05.2f" % (dec_deg, dec_min, dec_sec)
        
        # Step through column by column
        table_row += "%s & " % star["TIC"]
        table_row += "%s & " % star["TOI"]
        table_row += "%s & " % star["2mass"]
        table_row += "%s & " % star["source_id"]
        table_row += "%s & " % ra
        table_row += "%s & " % dec
        table_row += "%0.2f & " % star["G_mag"]
        table_row += "%0.2f & " % star["Bp-Rp"]
        table_row += r"%0.2f $\pm$ %0.2f & " % (star["plx"], star["e_plx"])
        table_row += "%0.1f & " % star["ruwe"]
        table_row += "%i & " % star["n_pc"]

        """
        # Now do references
        refs = [star["teff_bib_ref"], star["logg_bib_ref"], 
                star["feh_bib_ref"], star["vsini_bib_ref"]]
         
        for ref in refs:   
            if ref == "":
                table_row += "-,"   
                  
            elif ref not in references:
                references.append(ref)
                ref_i = np.argwhere(np.array(references)==ref)[0][0] + 1
                table_row += "%s," % ref_i
            
            elif ref in references:
                ref_i = np.argwhere(np.array(references)==ref)[0][0] + 1
                table_row += "%s," % ref_i
        """
        # Remove the final comma and append (Replace any nans with '-')
        table_rows.append(table_row[:-1].replace("nan", "-")  + r"\\")
         
    # Finish the table
    table_rows.append("\\hline")
    table_rows.append("\\end{tabular}")
    
    # Add notes section with references
    table_rows.append("\\begin{minipage}{\linewidth}")
    table_rows.append("\\vspace{0.1cm}")
    
    table_rows.append("\\textbf{Notes:} $^a$Gaia \citet{brown_gaia_2018} - "
                      " note that Gaia parallaxes listed here have not been "
                      "corrected for the zeropoint offset, "
                      "$^b$SIMBAD, $^c$Tycho \citet{hog_tycho-2_2000}, "
                      "$^d$2MASS \citet{skrutskie_two_2006} \\\\")
    
    #for ref_i, ref in enumerate(references):
    #    table_rows.append("%i. \\citet{%s}, " % (ref_i+1, ref))
    
    # Remove last comma
    table_rows[-1] = table_rows[-1][:-1]
    
    table_rows.append("\\end{minipage}")
    table_rows.append("\\end{table}")
    table_rows.append("\\end{landscape}")
    
    # Write the table
    np.savetxt("paper/table_targets.tex", table_rows, fmt="%s")

