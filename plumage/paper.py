"""
"""
import os
import pandas as pd
import numpy as np
import plumage.transits as transit
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
    label="tess",
    logg_col="logg_synth",
    exp_scale=-12,
    info_cat=None,
    do_activity_crossmatch=False,):
    """Make the final results table to display the angular diameters and 
    derived fundamental parameters.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in observations, TIC, and TOI info
    observations = utils.load_fits_table("OBS_TAB", label, path="spectra")
    
    # Crossmatch with activity
    if do_activity_crossmatch:
        observations = utils.merge_activity_table_with_obs(
            observations,
            label,
            fix_missing_source_id=True)

    if info_cat is None:
        info_cat = utils.load_info_cat(
            remove_fp=True, 
            in_paper=True,
            only_observed=True,
            use_mann_code_for_masses=True,)

    if label == "tess":
        id_col = "TIC"
        caption = "Final results for TESS candidate exoplanet hosts"
    else:
        id_col = "Gaia DR2"
        caption = "Final results for cool dwarf standards"

    comb_info = observations.join(info_cat, "source_id", rsuffix="_info", how="inner")
    comb_info.sort_values("teff_synth", inplace=True)
    
    cols = OrderedDict([
        ("TOI", ""),
        (id_col, ""),
        (r"$T_{\rm eff}$", "(K)"),
        (r"$\log g$", ""),
        (r"[Fe/H]", ""),
        (r"$M$", "($M_\odot$)"),
        (r"$R$", "($R_\odot$)"),
        (r"$m_{\rm bol}$", ""),
        (r"$f_{\rm bol}$", 
         r"{{\footnotesize(10$^{%i}\,$ergs s$^{-1}$ cm $^{-2}$)}}" % exp_scale),
        #(r"$L$", "($L_\odot$)"),
        (r"EW(H$\alpha$)", r"\SI{}{\angstrom}"),
        (r"$\log R^{'}_{\rm HK}$", ""),

    ])
    
    # Pop activity keys if making standards
    if label != "tess":
        cols.pop("TOI")
        cols.pop(r"EW(H$\alpha$)")
        cols.pop(r"$\log R^{'}_{\rm HK}$")

    header = []
    header_1 = []
    header_2 = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    header.append("\\begin{table*}")
    header.append("\\centering")

    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(2, "\\label{{tab:final_results_{}}}".format(label))
    header_1.insert(2, "\\caption{{{}}}".format(caption))

    header_2 = header.copy()
    header_2.insert(2, "\\contcaption{{{}}}".format(caption))

    # Populate the table for every science target
    for star_i, star in comb_info.iterrows():
        
        # Check to make sure this star has results
        if np.isnan(star["teff_synth"]):
            print("Skipping", star_i)
            continue

        table_row = ""
        
        # Step through column by column
        if label == "tess":
            table_row += "%s & " % int(star["TOI"])
            table_row += "%s & " % star["TIC"]
        else:
            table_row += "%s & " % star.name

        # Temperature
        table_row += r"{:0.0f} $\pm$ {:0.0f} & ".format(
            star["teff_synth"], star["e_teff_synth"])

        table_row += r"{:0.2f} $\pm$ {:0.2f} &".format(
            star[logg_col], star["e_{}".format(logg_col)])

        # Photometric [Fe/H]
        if np.isnan(star["phot_feh"]):
            table_row += "- &"
        else:
            table_row += r"{:0.2f} &".format(star["phot_feh"])

        # Mass and radius
        table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
            star["mass_m19"], star["e_mass_m19"])

        table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
            star["radius"], star["e_radius"])

        # Bolometric magnitude
        table_row += r"{:0.2f} $\pm$ {:0.2f} &".format(
            star["mbol_synth"], star["e_mbol_synth"])

        # For fbol representation, split mantissa and exponent
        table_row += r"{:5.1f} $\pm$ {:0.1f} &".format(
            star["fbol"] / 10**exp_scale, 
            star["e_fbol"] / 10**exp_scale)
        
        # Luminosity
        #table_row += r"{:0.3f} $\pm$ {:0.3f} & ".format(
            #star["L_star"], star["e_L_star"])

        # H alpha and logR`HK
        if do_activity_crossmatch and label == "tess":
            table_row += r"{:0.2f} & ".format(star["EW(Ha)"])
            table_row += r"{:0.2f} ".format(star["logR'HK"])
        elif label == "tess":
            table_row += r"{:0.2f} & ".format(np.nan)
            table_row += r"{:0.2f} ".format(np.nan)
        # Not writing activity columns, remove &
        else:
            table_row = table_row[:-2]

        # Get rid of nans, append
        table_rows.append(table_row.replace("nan", "-") + r"\\")

    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    footer.append("\\end{table*}")
    
    # Write the table/s
    break_rows = np.arange(break_row, len(comb_info), break_row)
    low_row = 0
    
    for table_i, break_row in enumerate(break_rows):
        if table_i == 0:
            header = header_1
        else:
            header = header_2

        table_x = header + table_rows[low_row:break_row] + footer + notes
        np.savetxt("paper/table_final_results_{}_{:0.0f}.tex".format(
            label, table_i), table_x, fmt="%s")
        low_row = break_row

    # Do final part table
    if low_row < len(comb_info):
        table_i += 1
        table_x = header_2 + table_rows[low_row:] + footer + notes
        np.savetxt("paper/table_final_results_{}_{:0.0f}.tex".format(
            label, table_i), table_x, fmt="%s")

def make_table_targets(break_row=45, tess_info=None,):
    """Make the LaTeX table to summarise the target information.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in the TESS target info
    if tess_info is None:
        tess_info = utils.load_info_cat(remove_fp=True, only_observed=True, 
            in_paper=True)
    tess_info.sort_values("G_mag", inplace=True)
    
    cols = OrderedDict([
        ("TOI$^a$", ""),
        ("TIC$^b$", ""),
        ("2MASS$^c$", ""),
        ("Gaia DR2$^d$", ""),
        ("RA$^d$", "(hh mm ss.ss)"),
        ("DEC$^d$", "(dd mm ss.ss)"),
        ("$G^d$", "(mag)"), 
        ("${B_p-R_p}^d$", "(mag)"),
        ("Plx$^d$", "(mas)"),
        ("ruwe$^d$", ""),
        ("E($B-V$)",""),
        (r"N$_{\rm pc}^e$", "")
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
    
    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(3, "\\caption{Science targets}")
    header_1.insert(4, "\\label{tab:science_targets}")

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
        table_row += "%s & " % int(star["TOI"])
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
        table_row += "%0.2f & " % star["ebv"]
        table_row += "%i " % star["n_pc"]

        # Replace any nans with '-'
        table_rows.append(table_row.replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")
    footer_2 = footer.copy()
    
    # Add notes section with references
    notes.append("\\begin{minipage}{\linewidth}")
    notes.append("\\vspace{0.1cm}")
    
    notes.append("\\textbf{Notes:} $^a$ TESS Object of Interest ID, "
                 "$^b$ TESS Input Catalogue ID "
                 "\citep{stassun_tess_2018, stassun_revised_2019},"
                 "$^c$2MASS \citep{skrutskie_two_2006}, "
                 "$^c$Gaia \citep{brown_gaia_2018} - "
                 " note that Gaia parallaxes listed here have been "
                 "corrected for the zeropoint offset, "
                 "$^d$Number of candidate planets, NASA Exoplanet Follow-up "
                 "Observing Program for TESS \\\\")
    
    notes.append("\\end{minipage}")
    notes.append("\\end{table}")
    notes.append("\\end{landscape}")

    footer_2.append("\\end{table}")
    footer_2.append("\\end{landscape}")
    
    # Write the tables
    table_1 = header_1 + table_rows[:break_row] + footer + notes
    table_2 = header_2 + table_rows[break_row:] + footer_2

    # Write the table
    np.savetxt("paper/table_targets_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_targets_2.tex", table_2, fmt="%s")


def make_table_observations(observations, info_cat, label, break_row=60, 
    add_rv_systematic=True, rv_syst=4.5,):
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
        table_kind = "*"
        caption = "Observing log for TESS candidate exoplanet host stars"
    else:
        id_col = "source_id"
        id_label = "Gaia DR2"
        table_kind = "*"
        caption = "Observing log for cool dwarf standards"

    cols = OrderedDict([
        (id_label, ""),
        ("UT Date", ""), 
        (r"airmass", ""), 
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
    header.append("\\begin{{table{}}}".format(table_kind))
    header.append("\\centering")
    
    header.append("\\begin{tabular}{%s}" % ("c"*(2+len(cols))))
    header.append("\hline")
    
    
    header.append((("%s & "*len(cols))) % tuple(cols.keys()))
    header.append(r"\multicolumn{2}{c}{SNR} \\")
    header.append((("%s & "*len(cols))) % tuple(cols.values()))
    header.append(r"(B) & (R) \\")
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(2, "\\caption{{{}}}".format(caption))
    header_1.insert(3, "\\label{{tab:observing_log_{}}}".format(label))

    header_2 = header.copy()
    header_2.insert(2, "\\contcaption{{{}}}".format(caption))
    
    # Populate the table for every science target
    for source_id, star in comb_info.iterrows():
        table_row = ""
        
        # Add the RV systematic
        if add_rv_systematic:
            e_rv = np.sqrt(star["e_rv"]**2 + rv_syst**2)
        else:
            e_rv = star["e_rv"]

        # Step through column by column
        if label == "tess":
           table_row += "%s & " % star[id_col]
        else: 
            table_row += "%s & " % source_id
        table_row += "%s & " % star["date"].split("T")[0][2:]
        table_row += "%0.1f & " % star["airmass"]
        table_row += "%0.0f & " % star["exp_time"]
        table_row += r"%0.2f $\pm$ %0.2f & " % (star["rv"], e_rv)
        table_row += "%0.0f & " % star["snr_b"]
        table_row += "%0.0f " % star["snr_r"]

        # Replace any nans with '-'
        table_rows.append(table_row.replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")
    
    #table_rows.append("\\end{minipage}")
    footer.append("\\end{{table{}}}".format(table_kind))
    #footer.append("\\end{landscape}")
    
    # Write the table/s
    break_rows = np.arange(break_row, len(comb_info), break_row)
    low_row = 0
    
    for table_i, break_row in enumerate(break_rows):
        if table_i == 0:
            header = header_1
        else:
            header = header_2
        table_x = header + table_rows[low_row:break_row] + footer + notes
        np.savetxt("paper/table_observations_{}_{:0.0f}.tex".format(
            label, table_i), table_x, fmt="%s")
        low_row = break_row

    # Do final part table
    if low_row < len(comb_info):
        table_i += 1
        table_x = header_2 + table_rows[low_row:] + footer + notes
        np.savetxt("paper/table_observations_{}_{:0.0f}.tex".format(
            label, table_i), table_x, fmt="%s")

    #table_1 = header_1 + table_rows[:break_row] + footer
    #table_2 = header_2 + table_rows[break_row:] + footer

    # Write the table
    #np.savetxt("paper/table_observations_{}_1.tex".format(label), table_1, fmt="%s")
    #np.savetxt("paper/table_observations_{}_2.tex".format(label), table_2, fmt="%s")


def make_table_planet_params(
    break_row=60, 
    force_high_uncertainty_to_nan=False,
    default_i_uncertainty=5,):
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
        ("Sector/s", ""),       # TESS sectors
        ("Period", "(days)"),
        #(r" $a/R_*$ ", r""),    # prior
        (r"$R_p/R_{\star}$", r""),     # fit params
        (r"$a/R_{\star}$", r""),       # fit params
        (r"$e$ flag", ""),       # sma/r* mismatch flag
        (r"$i$", r"($^{\circ}$)"), # fit params
        (r"$R_p$", r"($R_{\oplus}$)"),  # physical
        #(r"$a$", "(au)"),        # physical
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

    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")

    #mc = (r"\multicolumn{3}{c}{TESS} & Prior & \multicolumn{3}{c}{Fit}"
    #      r" & \multicolumn{2}{c}{Physical} \\")

    #header.append(mc)
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(
        2, 
        "\\caption{Final results for TESS candidate exoplanets}")
    header_1.insert(3, "\\label{tab:planet_params}")

    header_2 = header.copy()
    header_2.insert(
        2, 
        "\\contcaption{Final results for TESS candidate exoplanets}")

    # Set unreasonably high uncertainties to nans TODO
    if force_high_uncertainty_to_nan:
        bad_std_mask = np.logical_or(
            toi_results["e_rp_rstar_fit"] > 10,
            toi_results["e_inclination_fit"] > 10)
        toi_results.loc[bad_std_mask, "e_sma_rstar_fit"] = np.nan
        toi_results.loc[bad_std_mask, "e_rp_rstar_fit"] = np.nan
        toi_results.loc[bad_std_mask, "e_inclination_fit"] = np.nan
        toi_results.loc[bad_std_mask, "e_rp_fit"] = np.nan

    # Populate the table for every science target
    for toi, star in toi_results.iterrows():
        # Skip this TOI if it has been marked as excluded
        if star["excluded"] or star["sectors"] == "NA":
            continue

        table_row = ""
        
        # Step through column by column
        # If star has a placeholder TOI ID, use proper name
        if star["proper_name"] != "N/A":
            table_row += "{} & ".format(star["proper_name"].split("-")[-1])
        else:
            table_row += "{:0.2f} & ".format(toi)

        table_row += "{:0.0f} & ".format(star["TIC"])

        # Number of sectors
        table_row += "{} & ".format(star["sectors"])

        # Use ExoFop period if we didn't fit for it
        if np.isnan(star["period_fit"]):
            #table_row += r"{:0.5f} $\pm$ {:0.5f} &".format(
            #        star["Period (days)"], star["Period error"])
            table_row += r"{:0.6f} &".format(star["Period (days)"])
        # Otherwise use the fitted period
        else:
            table_row += r"{:0.5f} $\dagger$ &".format(star["period_fit"])
            
        # a/R* (prior)
        #table_row += r"{:0.4f} $\pm$ {:0.4f} &".format(
        #        star["sma_rstar"], star["e_sma_rstar"])

        # Rp/R*
        table_row += r"{:0.4f} $\pm$ {:0.4f} &".format(
                star["rp_rstar_fit"], star["e_rp_rstar_fit"])

        # a/R*
        table_row += r"{:0.2f} $\pm$ {:0.2f} &".format(
                star["sma_rstar_fit"], star["e_sma_rstar_fit"])

        # Flag
        table_row += "{:0.0f} & ".format(star["sma_rstar_mismatch_flag"])

        # Inclination
        if star["e_inclination_fit"] < 10:
            table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
                star["inclination_fit"], star["e_inclination_fit"])
        else:
            table_row += r"{:0.3f} $\pm$ {:0.3f} &".format(
                star["inclination_fit"], default_i_uncertainty)

        # Rp (physical)
        table_row += r"{:0.2f} $\pm$ {:0.2f}".format(
                star["rp_fit"], star["e_rp_fit"])

        # SMA (physical)
        #table_row += r"{:0.3f} $\pm$ {:0.3f}".format(
        #    star["sma"]/const.au.si.value, 
        #    star["e_sma"]/const.au.si.value)

        table_rows.append(table_row + r"\\")
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    footer_2 = footer.copy()
    footer_2.append("\\end{table*}")

    # Add notes section
    notes.append("\\begin{minipage}{\linewidth}")
    notes.append("\\vspace{0.1cm}")
    
    n_bad_i = np.sum(toi_results["e_inclination_fit"] > 10)

    notes.append("\\textbf{{Notes:}} Periods denoted by $\dagger$ are not as "
        "reported by ExoFOP, and have been refitted here. These are "
        "overwhelmingly systems with TESS extended mission data, thus having "
        "longer time baselines with which to constrain orbital periods. Our "
        "fitted periods however are generally consistent within uncertainties "
        "of their ExoFOP values, and as such we do not report new "
        "uncertainties here. Additionally, our least squares fits to {:0.0f}"
        " of our light curves proved insenstive to non-edge-on inclinations."
        " As such, we report conservative uncertanties of $\pm{:0.0f}$\degree~"
        " for these planets.\\\\".format(n_bad_i, default_i_uncertainty))

    notes.append("\\end{minipage}")
    notes.append("\\end{table*}")
    
    # Write the tables
    table_1 = header_1 + table_rows[:break_row] + footer + notes
    table_2 = header_2 + table_rows[break_row:] + footer_2
    
    np.savetxt("paper/table_planet_params_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_planet_params_2.tex", table_2, fmt="%s")


def make_table_planet_lit_comp(confirmed_planet_tab="data/known_planets.tsv",):
    """
    """
    # Import literature data of confirmed planets
    cp_cat = pd.read_csv(confirmed_planet_tab, delimiter="\t", index_col="TOI")
    cp_cat.sort_values("TOI", inplace=True)

    cp_cat = cp_cat[cp_cat["in_paper"]]

    # Column names and associated units
    columns = OrderedDict([("TOI", ""), 
                           ("TIC", ""),
                           ("Name", ""),
                           ("$R_P/R_*$", ""),
                           ("$a/R_*$", ""),
                           ("i", "\degree"), 
                           ("$R_P$", "$R_\oplus$"), 
                           ("Reference", ""),])
    
    table_rows = []
    
    # Construct the header of the table
    #table_rows.append("\\begin{landscape}")
    table_rows.append("\\begin{table*}")
    table_rows.append("\\centering")
    table_rows.append(
        "\\caption{Summary of literature properties for already confirmed planets}")
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
        if planet_info["placeholder_toi"]:
            table_row += "- & "
        else:
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


def make_table_ld_coeff(
    break_row=64, 
    label="tess",
    info_cat=None,):
    """Make the final results table to display the angular diameters and 
    derived fundamental parameters.

    Parameters
    ----------
    break_row: int
        Which row to break table 1 at and start table 2.
    """
    # Load in observations, TIC, and TOI info
    observations = utils.load_fits_table("OBS_TAB", label, path="spectra")
    
    if info_cat is None:
        info_cat = utils.load_info_cat(
            remove_fp=True, 
            in_paper=True,
            only_observed=True,
            use_mann_code_for_masses=True,)

    if label == "tess":
        id_col = "TIC"
    else:
        id_col = "Gaia DR2"

    comb_info = observations.join(info_cat, "source_id", rsuffix="_info", how="inner")
    comb_info.sort_values("teff_synth", inplace=True)
    
    cols = OrderedDict([
        (id_col, ""),
        (r"$a_1$", ""),
        (r"$a_2$", ""),
        (r"$a_3$", ""),
        (r"$a_4$", ""),
        
    ])
                           
    header = []
    header_1 = []
    header_2 = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    header.append("\\begin{table}")
    header.append("\\centering")
    #header.append("\\label{tab:ld_coeff}")

    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    caption = ("Nonlinear limb darkening coefficients from "
               "\citet{claret_limb_2017}")
    
    header_1.insert(2, "\\caption{{{}}}".format(caption))
    header_1.insert(3, "\\label{tab:ld_coeff}")

    header_2 = header.copy()
    header_2.insert(2, "\\contcaption{Limb darkening coefficients}")

    # Populate the table for every science target
    for star_i, star in comb_info.iterrows():
        
        table_row = ""
        
        # Step through column by column
        if label == "tess":
            table_row += "%s & " % star["TIC"]
        else:
            table_row += "%s & " % star.name

        # Coefficients
        table_row += r"{:0.3f} & ".format(star["ldc_a1"])
        table_row += r"{:0.3f} & ".format(star["ldc_a2"])
        table_row += r"{:0.3f} & ".format(star["ldc_a3"])
        table_row += r"{:0.3f} ".format(star["ldc_a4"])

        table_rows.append(table_row + r"\\")

    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    footer.append("\\end{table}")
    
    # Write the table/s
    break_rows = np.arange(break_row, len(comb_info), break_row)
    low_row = 0
    
    for table_i, break_row in enumerate(break_rows):
        if table_i == 0:
            header = header_1
        else:
            header = header_2
        table_x = header + table_rows[low_row:break_row] + footer + notes
        np.savetxt("paper/table_ldc_{}_{:0.0f}.tex".format(
            label, table_i), table_x, fmt="%s")
        low_row = break_row

    # Do final part table
    if low_row < len(comb_info):
        table_i += 1
        table_x = header_2 + table_rows[low_row:] + footer + notes
        np.savetxt("paper/table_ldc_{}_{:0.0f}.tex".format(
            label, table_i), table_x, fmt="%s")
