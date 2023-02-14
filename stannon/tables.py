"""Functions to generate LaTeX tables.
"""
import numpy as np
import pandas as pd
from collections import OrderedDict

label_source_refs = {
    "VF05":"valenti_spectroscopic_2005",
    "Sou06":"sousa_spectroscopic_2006",
    "vB09":"van_belle_directly_2009",
    "D09":"demory_mass-radius_2009",
    "B12":"boyajian_stellar_2012-1",
    "RA12":"rojas-ayala_metallicity_2012",
    "vB12":"von_braun_gj_2012",
    "G14":"gaidos_trumpeting_2014",
    "vB14":"von_braun_stellar_2014",
    "M15":"mann_how_2015",
    "T15":"terrien_near-infrared_2015",
    "M18":"montes_calibrating_2018",
    "R19":"rabus_discontinuity_2019",
    "R21":"rains_characterization_2021",
    "A12":"adibekyan_chemical_2012",
    "C01":"cayrel_de_strobel_catalogue_2001",
    "M13":"mann_prospecting_2013",
}

def make_table_benchmark_overview(
    obs_tab,
    labels_adopt,
    sigmas_adopt,
    labels_fit,
    label_sources,
    abundance_labels=[],
    break_row=61,
    synth_logg_col="logg_synth",
    aberrant_logg_threshold=0.15,):
    """Make a LaTeX table of our adopted benchmark stellar parameters, the
    source/s of those values, as well as the systematic corrected results for
    the benchmark set.
    """
    info_cols = [
        "Star",
        "Gaia DR3",
        r"$BP-RP$",
        r"$BP$",
        r"SNR$_{\rm B}$",
        r"SNR$_{\rm R}$",
    ]

    info_units = ["", "", "", "", "", "",]

    param_cols = [
        r"$T_{\rm eff}$",
        r"$\log g$",
        "[Fe/H]",
    ]
    
    param_units = ["(K)", "(dex)", "(dex)",]

    # Account for abundances if we're using them - add each to dictionary
    for abundance in abundance_labels:
        abund = "[{}/H]".format(abundance.split("_")[0])
        param_cols.append(abund)
        param_units.append("(dex)")

    # Combine columns and units
    col_names = info_cols + param_cols + ["References"] + param_cols
    col_units = info_units + param_units + [""] + param_units

    header = []
    table_rows = []
    footer = []
    notes = []
    
    # Keeping track of references
    references = []

    # Construct the header of the table
    #header.append("\\begin{landscape}")
    header.append("\\begin{table*}")
    header.append("\\centering")
    
    col_format = ("c"*len(info_cols) + "|" + "c"*(len(param_cols)+1) + "|" 
        + "c"*len(param_cols))

    header.append(r"\resizebox{\textwidth}{!}{%")
    header.append("\\begin{tabular}{%s}" % col_format)
    header.append("\hline")
    header.append((
        r"\multicolumn{{{:0.0f}}}{{c}}{{}} & "
        r"\multicolumn{{{:0.0f}}}{{c}}{{Adopted Parameters}} & "
        r"\multicolumn{{{:0.0f}}}{{c}}{{Fitted Parameters}} \\").format(
            len(info_cols), labels_adopt.shape[1]+1, labels_adopt.shape[1]))
    header.append((("%s & "*len(col_names))[:-2] + r"\\") % tuple(col_names))
    header.append((("%s & "*len(col_units))[:-2] + r"\\") % tuple(col_units))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(2, "\\caption{Benchmark Stars}")

    header_2 = header.copy()
    header_2.insert(2, "\\contcaption{Benchmark Stars}")

    # Sort by BP-RP
    ii = np.argsort(obs_tab["BP_RP_dr3"])
    sorted_tab = obs_tab.iloc[ii]
    sorted_labels_adopt = labels_adopt[ii]
    sorted_sigmas_adopt = sigmas_adopt[ii]
    sorted_labels_fit = labels_fit[ii]
    sorted_label_sources = label_sources[ii]

    # Populate the table for every science target
    for star_i, (source_id, star) in enumerate(sorted_tab.iterrows()):
        table_row = ""
        
        # Star ID/s
        table_row += "{} & ".format(star["simbad_name"])
        table_row += "{} & ".format(source_id)

        # Colour
        table_row += "{:0.2f} & ".format(star["BP_RP_dr3"])

        # Magnitude
        table_row += "{:0.2f} & ".format(star["BP_mag_dr3"])

        # SNR
        table_row += "{:0.0f} & ".format(star["snr_b"])
        table_row += "{:0.0f} & ".format(star["snr_r"])

        # Adopted Parameters (with references and uncertainties)
        # ------------------------------------------------------
        # Teff
        table_row += r"${:0.0f}\pm{:0.0f}$ & ".format(
            sorted_labels_adopt[star_i, 0], sorted_sigmas_adopt[star_i, 0])
        
        # Logg
        table_row += r"${:0.2f}\pm{:0.2f}$ & ".format(
            sorted_labels_adopt[star_i, 1], sorted_sigmas_adopt[star_i, 1])

        # [Fe/H]
        if sorted_label_sources[star_i][2] != "":
            table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                sorted_labels_adopt[star_i, 2], sorted_sigmas_adopt[star_i, 2])
        else:
            table_row += r"- & "

        # Abundances
        for abund_i in range(len(abundance_labels)):
            # Get label index
            label_i = 3 + abund_i

            if sorted_label_sources[star_i][label_i] != "":
                table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                    sorted_labels_adopt[star_i, label_i], 
                    sorted_sigmas_adopt[star_i, label_i])
            else:
                table_row += r"- & "

        # Now do references
        refs = sorted_label_sources[star_i]

        # TODO HACK: Delete
        refs = [ref.replace("TW", "M13") for ref in refs]

        for ref in refs:
            if ref == "":
                table_row += "-,"
                  
            elif ref not in references:
                references.append(ref)
                table_row += "{},".format(ref)
            
            elif ref in references:
                table_row += "{},".format(ref)

        # Remove last comma, add &
        table_row = table_row[:-1] + " & "

        # Fitted Parameters
        # -----------------
        # Teff
        table_row += r"${:0.0f}$ & ".format(sorted_labels_fit[star_i, 0])

        # logg - making sure to flag the star if it has an aberrant logg
        delta_logg = np.abs(
            sorted_labels_fit[star_i, 1] - star[synth_logg_col])
        
        if delta_logg > aberrant_logg_threshold:
            table_row += r"${:0.2f} $\dagger & ".format(
                sorted_labels_fit[star_i, 1])
        else:
            table_row += r"${:0.2f}$ & ".format(
                sorted_labels_fit[star_i, 1])

        # [Fe/H]
        table_row += r"${:+0.2f}$ & ".format(
            sorted_labels_fit[star_i, 2])

        # Abundances
        for abund_i in range(len(abundance_labels)):
            # Get label index
            label_i = 3 + abund_i

            table_row += r"${:+0.2f}$ & ".format(
                sorted_labels_fit[star_i, label_i],)

        # Replace any nans with '-', remove final space and &
        table_rows.append(table_row[:-2].replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}}")

    # Now add the separate info for the two tables
    footer_1 = footer.copy()
    footer_1.append("\\label{tab:benchmark_parameters}")

    footer_2 = footer.copy()
    
    # Add notes section with references
    notes.append("\\begin{minipage}{\linewidth}")
    notes.append("\\vspace{0.1cm}")
    
    notes.append("\\textbf{References:}")
    notes_references = ""

    for ref in references:
        if ref in label_source_refs:
            bib_ref = "\\citet{{{}}}".format(label_source_refs[ref])
        else:
            bib_ref = "-"
        notes_references += "{}: {}, ".format(ref, bib_ref)
    
    # Remove last comma
    notes_references = notes_references[:-2]
    notes.append(notes_references)
    
    notes.append("\\end{minipage}")
    notes.append("\\end{table*}")
    
    # Write the table/s
    break_rows = np.arange(break_row, len(obs_tab), break_row)
    low_row = 0
    
    for table_i, break_row in enumerate(break_rows):
        if table_i == 0:
            header = header_1
            footer = footer_1
        else:
            header = header_2
            footer = footer_2
        table_x = header + table_rows[low_row:break_row] + footer + notes
        np.savetxt(
            "paper/table_benchmark_params_{:0.0f}.tex".format(table_i),
            table_x,
            fmt="%s")
        low_row = break_row

    # Do final part table
    if low_row < len(obs_tab):
        table_i += 1
        table_x = header_2 + table_rows[low_row:] + footer_2 + notes
        np.savetxt(
            "paper/table_benchmark_params_{:0.0f}.tex".format(table_i),
            table_x,
            fmt="%s")


def make_table_parameter_fit_results(
    obs_tab,
    label_fits,
    e_label_fits,
    abundance_labels=[],
    break_row=61,
    star_label=("Star", "simbad_name"),
    table_label="benchmark",
    caption="",
    synth_logg_col="logg_synth",
    aberrant_logg_threshold=0.15,):
    """Make a LaTeX table of our Cannon fitted stellar parameters.
    """
    cols = OrderedDict([
        (star_label[0], ""),
        ("Gaia DR3", ""),
        (r"$B_P$", ""),
        (r"$B_P-R_P$", ""),
        (r"SNR$_B$", ""),
        (r"SNR$_R$", ""),
        (r"$T_{\rm eff}$", "(K)"),
        (r"$\log g$", "(dex)"),
        ("[Fe/H]", "(dex)"),
    ])

    # Account for abundances if we're using them - add each to dictionary
    for abundance in abundance_labels:
        abund = "[{}/H]".format(abundance.split("_")[0])
        cols[abund] = "(dex)"
    
    header = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    #header.append("\\begin{landscape}")
    header.append("\\begin{table*}")
    header.append("\\centering")
    
    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(2, "\\caption{{{}}}".format(caption))

    header_2 = header.copy()
    header_2.insert(2, "\\contcaption{{{}}}".format(caption))

    # Sort by Bp-Rp
    ii = np.argsort(obs_tab["BP_RP_dr3"].values)[::-1]
    sorted_tab = obs_tab.iloc[ii]
    sorted_labels = label_fits[ii]
    sorted_e_labels = e_label_fits[ii]

    # Populate the table for every science target
    for star_i, (source_id, star) in enumerate(sorted_tab.iterrows()):
        table_row = ""
        
        # Star ID/s
        table_row += "{} & ".format(star[star_label[1]])
        table_row += "{} & ".format(source_id)

        # Magnitude
        table_row += "{:0.2f} & ".format(star["BP_mag_dr3"])

        # Colour
        table_row += "{:0.2f} & ".format(star["BP_RP_dr3"])

        # SNR
        table_row += "{:0.0f} & ".format(star["snr_b"])
        table_row += "{:0.0f} & ".format(star["snr_r"])

        # Teff
        table_row += r"${:0.0f}\pm{:0.0f}$ & ".format(
            sorted_labels[star_i, 0], sorted_e_labels[star_i, 0])

        # logg - making sure to flag the star if it has an aberrant logg
        delta_logg = np.abs(
            sorted_labels[star_i, 1] - star[synth_logg_col])
        
        if delta_logg > aberrant_logg_threshold:
            table_row += r"${:0.2f}\pm{:0.2f}$ $\dagger$ & ".format(
                sorted_labels[star_i, 1], sorted_e_labels[star_i, 1])
        else:
            table_row += r"${:0.2f}\pm{:0.2f}$ & ".format(
                sorted_labels[star_i, 1], sorted_e_labels[star_i, 1])

        # [Fe/H]
        table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
            sorted_labels[star_i, 2], sorted_e_labels[star_i, 2])

        # Abundances
        for abund_i in range(len(abundance_labels)):
            # Get label index
            label_i = 3 + abund_i

            table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                sorted_labels[star_i, label_i], 
                sorted_e_labels[star_i, label_i])

        # Replace any nans with '-', remove final '&'
        table_rows.append(table_row[:-2].replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")

    # Now add the separate info for the two tables
    footer_1 = footer.copy()
    footer_1.append("\\label{{tab:{}_parameters}}".format(table_label))

    footer_2 = footer.copy()
    
    notes.append("\\end{table*}")

    # Write the table/s
    break_rows = np.arange(break_row, len(obs_tab), break_row)
    low_row = 0
    
    for table_i, break_row in enumerate(break_rows):
        if table_i == 0:
            header = header_1
            footer = footer_1
        else:
            header = header_2
            footer = footer_2
        table_x = header + table_rows[low_row:break_row] + footer + notes
        np.savetxt(
            "paper/table_param_fit_{}_{:0.0f}.tex".format(table_label, table_i),
            table_x,
            fmt="%s")
        low_row = break_row

    # Do final part table
    if low_row < len(obs_tab):
        table_i += 1
        table_x = header_2 + table_rows[low_row:] + footer_2 + notes
        np.savetxt(
            "paper/table_param_fit_{}_{:0.0f}.tex".format(table_label, table_i),
            table_x,
            fmt="%s")