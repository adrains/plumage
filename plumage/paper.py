"""
"""
import os
import pandas as pd
import numpy as np
import plumage.utils as utils
import astropy.constants as const 
from collections import OrderedDict
# Ensure the plotting folder exists to save to
here_path = os.path.dirname(__file__)
plot_dir = os.path.abspath(os.path.join(here_path, "..", "paper"))

if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# -----------------------------------------------------------------------------
# Making tables
# ----------------------------------------------------------------------------- 
def make_table_final_results(
    break_row=45, 
    logg_col="logg_m19",
    exp_scale=-12):
    """Make the final results table to display the angular diameters and 
    derived fundamental parameters.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in observations, TIC, and TOI info
    observations = utils.load_fits_obs_table("TESS", path="spectra")
    tic_info = utils.load_info_cat(remove_fp=True, only_observed=True)  
    toi_info = utils.load_exofop_toi_cat()

    comb_info = observations.join(tic_info, "source_id", rsuffix="_info")
    comb_info.sort_values("teff_synth", inplace=True)
    
    cols = OrderedDict([
        ("TIC", ""),
        (r"$T_{\rm eff}$", "(K)"),
        (r"$\log g$", ""),
        (r"[Fe/H]", ""),
        (r"$M$", "($M_\odot$)"),
        (r"$R$", "($R_\odot$)"),
        (r"$f_{\rm bol}$", 
         r"(10$^{%i}\,$ergs s$^{-1}$ cm $^{-2}$)" % exp_scale),
        (r"$L$", "($L_\odot$)"),
    ])
                           
    header = []
    header_1 = []
    header_2 = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    header.append("\\begin{table*}")
    header.append("\\centering")
    header.append("\\label{tab:final_results}")

    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(3, "\\caption{Final results}")

    header_2 = header.copy()
    header_2.insert(3, "\\contcaption{Final results}")

    # Populate the table for every science target
    for star_i, star in comb_info.iterrows():
        
        table_row = ""
        
        # Step through column by column
        table_row += "%s & " % star["TIC"]

        # Fitted spectroscopic params
        table_row += r"{:0.0f} $\pm$ {:0.0f} & ".format(
            star["teff_synth"], star["e_teff_synth"])

        table_row += r"{:0.2f} $\pm$ {:0.2f} &".format(
            star[logg_col], star["e_{}".format(logg_col)])

        table_row += r"{:0.2f} $\pm$ {:0.2f} &".format(
            star["feh_synth"], star["e_feh_synth"])

        table_row += r"{:0.3f} $\pm$ {:0.2f} &".format(
            star["mass_m19"], star["e_mass_m19"])

        table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
            star["radius"], star["e_radius"])

        # For fbol representation, split mantissa and exponent
        table_row += r"{:5.1f} $\pm$ {:0.1f} &".format(
            star["f_bol_avg"] / 10**exp_scale, 
            star["e_f_bol_avg"] / 10**exp_scale)
        
        table_row += r"{:0.3f} $\pm$ {:0.3f} ".format(
            star["lum"], star["e_lum"])

        table_rows.append(table_row + r"\\")
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    footer.append("\\end{table*}")
    
    # Write the tables
    table_1 = header_1 + table_rows[:break_row] + footer + notes
    table_2 = header_2 + table_rows[break_row:] + footer + notes
    
    np.savetxt("paper/table_final_results_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_final_results_2.tex", table_2, fmt="%s")

        

def make_table_targets(break_row=45):
    """Make the LaTeX table to summarise the target information.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in the TESS target info (TODO: function for this)
    tess_info = utils.load_info_cat(remove_fp=True, only_observed=True, 
        in_paper=True)
    tess_info.sort_values("G_mag", inplace=True)
    
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
        table_row += "%s & " % star_i
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


def make_table_observations(observations, info_cat, label, break_row=60):
    """Make the LaTeX table to summarise the observations.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Table join
    comb_info = observations.join(info_cat, "source_id", rsuffix="_info", how="inner")
    
    # Pick ID column based on label
    if label == "tess":
        id_col = "TIC"
        id_label = "TIC"
    else:
        id_col = "source_id"
        id_label = "Gaia DR2"

    cols = OrderedDict([
        (id_label, ""),
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
    for source_id, star in comb_info.iterrows():
        table_row = ""
        
        # Step through column by column
        if label == "tess":
           table_row += "%s & " % star[id_col]
        else: 
            table_row += "%s & " % source_id
        table_row += "%s & " % star["date"].split("T")[0][2:]
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
    np.savetxt("paper/table_observations_{}_1.tex".format(label), table_1, fmt="%s")
    np.savetxt("paper/table_observations_{}_2.tex".format(label), table_2, fmt="%s")


def make_table_planet_params(break_row=60,):
    """Make table of final planet parameters.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in TOI results
    toi_results = utils.load_fits_table("TRANSIT_FITS", label="tess", path="spectra")
    toi_results.sort_values("TOI", inplace=True)

    cols = OrderedDict([
        ("TOI", ""),
        ("TIC", ""),
        ("Period", "(days)"),
        (r" $a/R_*$ ", r""),     # prior
        (r"$R_p/R_*$", r""),     # fit params
        (r"$a/R_*$", r""),       # fit params
        (r"$i$", r"($^\circ$)"), # fit params
        (r"$R_p$", r"($R_E$)"),  # physical
        (r"$a$", "(au)"),        # physical
    ])

    header = []
    header_1 = []
    header_2 = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    header.append("\\begin{table*}")
    header.append("\\centering")
    header.append("\\label{tab:planet_params}")

    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")

    mc = (r"\multicolumn{3}{c}{TESS} & Prior & \multicolumn{3}{c}{Fit}"
          r" & \multicolumn{2}{c}{Physical} \\")

    header.append(mc)
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(3, "\\caption{Planet params}")

    header_2 = header.copy()
    header_2.insert(3, "\\contcaption{Planet params}")

    # Populate the table for every science target
    for toi, star in toi_results.iterrows():
        table_row = ""
        
        # Step through column by column
        table_row += "{:0.2f} & ".format(toi)
        table_row += "{:0.0f} & ".format(star["TIC"])

        # Period
        table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
                star["Period (days)"], star["Period error"])

        # a/R* (prior)
        table_row += r"{:0.4f} $\pm$ {:0.4f} &".format(
                star["sma_rstar"], star["e_sma_rstar"])

        # Rp/R*
        table_row += r"{:0.4f} $\pm$ {:0.4f} &".format(
                star["rp_rstar_fit"], star["e_rp_rstar_fit"])

        # a/R*
        table_row += r"{:0.4f} $\pm$ {:0.4f} &".format(
                star["sma_rstar_fit"], star["e_sma_rstar_fit"])

        # Inclination
        table_row += r"{:0.2f} $\pm$ {:0.2f} &".format(
            star["inclination_fit"], star["e_inclination_fit"])

        # Rp (physical)
        table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
                star["rp_fit"], star["e_rp_fit"])

        # SMA (physical)
        table_row += r"{:0.3f} $\pm$ {:0.3f}".format(
            star["sma"]/const.au.si.value, 
            star["e_sma"]/const.au.si.value)

        table_rows.append(table_row + r"\\")
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    footer.append("\\end{table*}")
    
    # Write the tables
    table_1 = header_1 + table_rows[:break_row] + footer + notes
    table_2 = header_2 + table_rows[break_row:] + footer + notes
    
    np.savetxt("paper/table_planet_params_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_planet_params_2.tex", table_2, fmt="%s")


def make_table_planet_lit_comp(confirmed_planet_tab="data/known_planets.tsv",):
    """
    """
    # Import literature data of confirmed planets
    cp_cat = pd.read_csv(confirmed_planet_tab, delimiter="\t", index_col="TOI")
    cp_cat.sort_values("TOI", inplace=True)

    # Column names and associated units
    columns = OrderedDict([("TOI", ""), 
                           ("TIC", ""),
                           ("Name", ""),
                           ("$R_P/R_*$", ""),
                           ("$a/R_*$", ""),
                           ("i", "\degree"), 
                           ("$R_P$", "$R_E$"), 
                           ("Reference", ""),])
    
    table_rows = []
    
    # Construct the header of the table
    #table_rows.append("\\begin{landscape}")
    table_rows.append("\\begin{table*}")
    table_rows.append("\\centering")
    table_rows.append("\\caption{Literature planet params}")
    table_rows.append("\\label{tab:planet_lit_params}")
    
    table_rows.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    table_rows.append("\hline")
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    table_rows.append("\hline")
    
    ref_i = 0
    references = []
    
    # Populate the table for every science target
    for toi, planet_info in cp_cat.iterrows():
        table_row = ""
        
        # Only continue if we have transit data on the planet
        if np.isnan(planet_info["rp_rstar"]):
            continue
        
        # Step through column by column
        table_row += "{:0.2f} & ".format(toi)

        table_row += "{:0.0f} & ".format(planet_info["TIC"])

        table_row += "{} & ".format(planet_info["name"])

        table_row += r"${:0.5f}^{{+{:0.5f}}}_{{-{:0.5f}}}$ & ".format(
            planet_info["rp_rstar"], planet_info["e_rp_rstar_pos"], 
            planet_info["e_rp_rstar_neg"],)
        
        table_row += r"${:0.2f}^{{+{:0.2f}}}_{{-{:0.2f}}}$ & ".format(
            planet_info["a_rstar"], planet_info["e_a_rstar_pos"], 
            planet_info["e_a_rstar_neg"],)

        table_row += r"${:0.2f}^{{+{:0.2f}}}_{{-{:0.2f}}}$ & ".format(
            planet_info["i"], planet_info["e_i_pos"], planet_info["e_i_neg"],)

        table_row += r"${:0.3f}^{{+{:0.3f}}}_{{-{:0.3f}}}$ & ".format(
            planet_info["rp"], planet_info["e_rp_pos"], 
            planet_info["e_rp_neg"],)

        table_row += "\\citet{{{}}}".format(planet_info["bib_ref"])
        
        table_rows.append(table_row  + r"\\")

    # Finish the table
    table_rows.append("\\hline")
    table_rows.append("\\end{tabular}")
    table_rows.append("\\end{table*}")
    
    # Write the table
    np.savetxt("paper/table_planet_lit_params.tex", table_rows, fmt="%s")


def make_table_ld_coeff():
    """
    """
    pass

def make_table_fbol():
    """
    """
    pass