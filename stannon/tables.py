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
    "vb12":"von_braun_gj_2012",
    "G14":"gaidos_trumpeting_2014",
    "vB14":"von_braun_stellar_2014",
    "M15":"mann_how_2015",
    "T15":"terrien_near-infrared_2015",
    "M18":"montes_calibrating_2018",
    "R19":"rabus_discontinuity_2019",
    "R21":"rains_characterization_2021",
}

def make_table_benchmark_overview(
    obs_tab,
    labels,
    e_labels,
    label_sources,
    abundance_labels=[],
    break_row=61,):
    """Make a LaTeX table of our adopted benchmark stellar parameters and the
    source/s of those values
    """
    cols = OrderedDict([
        ("Star", ""),
        ("Gaia DR2", ""),
        (r"$B_P$", ""),
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

    # Add last entry in the OrderedDict
    cols["References"] = ""
    
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
    
    header.append("\\begin{tabular}{%s}" % ("c"*len(cols)))
    header.append("\hline")
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.keys()))
    header.append((("%s & "*len(cols))[:-2] + r"\\") % tuple(cols.values()))
    header.append("\hline")
    
    # Now add the separate info for the two tables
    header_1 = header.copy()
    header_1.insert(2, "\\caption{Benchmark Stars}")

    header_2 = header.copy()
    header_2.insert(2, "\\contcaption{Benchmark Stars}")

    # Sort by Teff
    ii = np.argsort(labels[:,0])
    sorted_tab = obs_tab.iloc[ii]
    sorted_labels = labels[ii]
    sorted_e_labels = e_labels[ii]
    sorted_label_sources = label_sources[ii]

    # Populate the table for every science target
    for star_i, (source_id, star) in enumerate(sorted_tab.iterrows()):
        table_row = ""
        
        # Star ID/s
        table_row += "{} & ".format(star["simbad_name"])
        table_row += "{} & ".format(source_id)

        # Magnitude
        table_row += "{:0.2f} & ".format(star["Bp_mag"])

        # SNR
        table_row += "{:0.0f} & ".format(star["snr_b"])
        table_row += "{:0.0f} & ".format(star["snr_r"])

        # Teff
        table_row += r"${:0.0f}\pm{:0.0f}$ & ".format(
            sorted_labels[star_i, 0], sorted_e_labels[star_i, 0])

        # Logg
        table_row += r"${:0.2f}\pm{:0.2f}$ & ".format(
            sorted_labels[star_i, 1], sorted_e_labels[star_i, 1])

        # [Fe/H]
        if sorted_label_sources[star_i][2] != "":
            table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                sorted_labels[star_i, 2], sorted_e_labels[star_i, 2])
        else:
            table_row += r"- & "

        # Abundances
        for abund_i in range(len(abundance_labels)):
            # Get label index
            label_i = 3 + abund_i

            if sorted_label_sources[star_i][label_i] != "":
                table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                    sorted_labels[star_i, label_i], 
                    sorted_e_labels[star_i, label_i])
            else:
                table_row += r"- & "

        # Now do references
        refs = sorted_label_sources[star_i]

        for ref in refs:
            if ref == "":
                table_row += "-,"
                  
            elif ref not in references:
                references.append(ref)
                table_row += "{},".format(ref)
            
            elif ref in references:
                table_row += "{},".format(ref)

        # Replace any nans with '-', remove final comma
        table_rows.append(table_row[:-1].replace("nan", "-")  + r"\\")
         
    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}")

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
        ("Gaia DR2", ""),
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
    ii = np.argsort(obs_tab["Bp-Rp"].values)[::-1]
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
        table_row += "{:0.2f} & ".format(star["Bp_mag"])

        # Colour
        table_row += "{:0.2f} & ".format(star["Bp-Rp"])

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