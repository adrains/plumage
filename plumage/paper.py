"""
"""
import os
import pandas as pd
import numpy as np
import plumage.utils as utils
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
    tess_info = pd.read_csv(
        "data/tess_info.tsv", 
        sep="\t", 
        dtype={"source_id":str, "observed":bool},
        true_values=["yes"]
    )

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

        

def make_table_targets(break_row=45):
    """Make the LaTeX table to summarise the target information.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in the TESS target info (TODO: function for this)
    tess_info = utils.load_info_cat(remove_fp=True, only_observed=True)
    exp_scale = -8
    
    cols = OrderedDict([
        ("TIC$^a$", ""),
        ("2MASS", ""),
        ("Gaia DR2$^b$", ""),
        ("RA$^b$", "(hh mm ss.ss)"),
        ("DEC$^b$", "(dd mm ss.ss)"),
        ("$G^b$", "(mag)"), 
        ("${B_p-R_p}^b$", "(mag)"),
        ("Plx$^b$", "(mas)"),
        ("ruwe$^b$", ""),
        (r"N$_{\rm pc}^c$", "")
    ])
    
    header = []
    header_1 = []
    header_2 = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    header.append("\\begin{landscape}")
    header.append("\\begin{table}")
    header.append("\\centering")
    header.append("\\label{tab:science_targets}")
    
    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(3, "\\caption{Science targets}")

    header_2 = header.copy()
    header_2.insert(3, "\\contcaption{Science targets}")
    
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
        ra = "{:02.0f} {:02.0f} {:05.2f}".format(ra_hr, ra_min, ra_sec)
        
        dec_deg = np.floor(star["dec"])
        dec_min = np.floor((star["dec"] - dec_deg) * 60)
        dec_sec = ((star["dec"] - dec_deg) * 60 - dec_min) * 60
        dec = "{:+02.0f} {:02.0f} {:05.2f}".format(dec_deg, dec_min, dec_sec)
        
        # Step through column by column
        table_row += "%s & " % star["TIC"]
        #table_row += "%s & " % star["TOI"]
        table_row += "%s & " % star["2mass"]
        table_row += "%s & " % star["source_id"]
        table_row += "%s & " % ra
        table_row += "%s & " % dec
        table_row += "%0.2f & " % star["G_mag"]
        table_row += "%0.2f & " % star["Bp-Rp"]
        table_row += r"%0.2f $\pm$ %0.2f & " % (star["plx"], star["e_plx"])
        table_row += "%0.1f & " % star["ruwe"]
        table_row += "%i " % star["n_pc"]

        # Replace any nans with '-'
        table_rows.append(table_row.replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")
    
    # Add notes section with references
    notes.append("\\begin{minipage}{\linewidth}")
    notes.append("\\vspace{0.1cm}")
    
    notes.append("\\textbf{Notes:} $^a$TESS Input Catalogue ID "
                 "\citep{stassun_tess_2018, stassun_revised_2019},"
                 "$^b$Gaia \citet{brown_gaia_2018} - "
                 " note that Gaia parallaxes listed here have not been "
                 "corrected for the zeropoint offset, "
                 "$^c$Number of candidate planets, NASA Exoplanet Follow-up "
                 "Observing Program for TESS \\\\")
    
    notes.append("\\end{minipage}")
    notes.append("\\end{table}")
    notes.append("\\end{landscape}")
    
    # Write the tables
    table_1 = header_1 + table_rows[:break_row] + footer + notes
    table_2 = header_2 + table_rows[break_row:] + footer + notes

    # Write the table
    np.savetxt("paper/table_targets_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_targets_2.tex", table_2, fmt="%s")


def make_table_observations(break_row=60):
    """Make the LaTeX table to summarise the observations.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in the TESS target info (TODO: function for this)
    observations = utils.load_fits_obs_table("TESS", path="spectra")
    toi_info = utils.load_exofop_toi_cat()
    
    cols = OrderedDict([
        ("TIC", ""),
        ("UT Date", ""), 
        (r"$X$", ""), 
        ("exp", "(sec)"), 
        ("RV", r"(km$\,$s$^{-1}$)"), 
    ])      
    
    header = []
    header_1 = []
    header_2 = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    #header.append("\\begin{landscape}")
    header.append("\\begin{table}")
    header.append("\\centering")
    header.append("\\label{tab:observing_log}")
    
    header.append("\\begin{tabular}{%s}" % ("c"*(2+len(cols))))
    header.append("\hline")
    
    
    header.append((("%s & "*len(cols))) % tuple(cols.keys()))
    header.append(r"\multicolumn{2}{c}{SNR} \\")
    header.append((("%s & "*len(cols))) % tuple(cols.values()))
    header.append(r"(B) & (R) \\")
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(3, "\\caption{Observing log}")

    header_2 = header.copy()
    header_2.insert(3, "\\contcaption{Observing log}")
    
    # Populate the table for every science target
    for star_i, star in observations.iterrows():
        table_row = ""

        # Get the TIC ID from the TOI ID
        toi = float(star["id"].replace("TOI", ""))
        tic = toi_info.loc[toi]["TIC"]
        
        # Step through column by column
        table_row += "%s & " % tic
        table_row += "%s & " % star["date"].split("T")[0]
        table_row += "%0.1f & " % star["airmass"]
        table_row += "%0.0f & " % star["exp_time"]
        table_row += r"%0.2f $\pm$ %0.2f & " % (star["rv"], star["e_rv"])
        table_row += "%0.0f & " % star["snr_b"]
        table_row += "%0.0f " % star["snr_r"]

        # Replace any nans with '-'
        table_rows.append(table_row.replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")
    
    #table_rows.append("\\end{minipage}")
    footer.append("\\end{table}")
    #footer.append("\\end{landscape}")
    
    table_1 = header_1 + table_rows[:break_row] + footer
    table_2 = header_2 + table_rows[break_row:] + footer

    # Write the table
    np.savetxt("paper/table_observations_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_observations_2.tex", table_2, fmt="%s")

