"""Functions to generate LaTeX tables.
"""
import os
import warnings
import numpy as np
from collections import OrderedDict
from stannon.vectorizer import PolynomialVectorizer

def make_table_sample_summary(
    obs_tab,
    labels,
    references,
    reference_dict,
    ref_this_work,
    table_folder="paper",):
    """Creates a table summarising the literature sources for all benchmarks,
    broken up by label. The table has columns [label, source, median_sigma, 
    n_with, n_without, n_adopted].

    Parameters
    ----------
    obs_tab: pandas DataFrame
        DataFrame containing label information.

    labels: str list
        List of labels, where the first two are assuming to be 'teff' and 
        'logg', and all others are chemical labels starting with 'Fe_H'.

    references: str list
        List of literature source abbreviations used to indicate which sample
        each label comes from in the column 'label_adopt_sigma_<label>'.

    references_dict: dict
        Dictionary mapping these reference abbreviations to the unique 
        identifier used when citing the reference in LaTeX (e.g. mann_how_2015)

    ref_this_work: str list
        List of label sources considered 'This Work' to be replaced with 'TW'.

    table_folder: str, default: 'paper'
        Folder to save the resulting .tex file table to.
    """
    # Grab total benchmarks for convenience
    n_benchmarks = len(obs_tab)

    col_names = [
        "Label",
        "Sample",
        r"Median $\sigma_{\rm label}$",
        r"$N_{\rm with}$",
        r"$N_{\rm without}$",
        r"$N_{\rm adopted}$"
    ]

    header = []
    table_rows = []
    footer = []

    # Construct the header of the table
    header.append("\\begin{table}")
    header.append("\\centering")
    header.append("\\caption{Benchmark sample summary}")

    col_format = "cccccc"

    header.append(r"\resizebox{\columnwidth}{!}{%")
    header.append("\\begin{tabular}{%s}" % col_format)
    header.append("\hline")
    header.append((("%s & "*len(col_names)) + r"\\") % tuple(col_names))

    # Remove extra &
    header[-1] = header[-1].replace("& \\", "\\")

    header.append("\hline")

    # Ensure we're only working with those stars selected as benchmarks
    is_cannon_benchmark = obs_tab["is_cannon_benchmark"].values
    benchmarks = obs_tab[is_cannon_benchmark]

    # Initialise table rows:
    table_rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # ---------------------------------------------------------------------
        # Teff
        # --------------------------------------------------------------------
        # All teffs
        median_teff_sigma = np.median(benchmarks["label_adopt_sigma_teff"])
        teff_str = \
            r"$T_{{\rm eff}}$ & All & {:0.0f}\,K & {:d} & {:d} & {:d} \\"
        teff_row = teff_str.format(
                median_teff_sigma,              # median sigma
                n_benchmarks,                   # with
                0,                              # without
                n_benchmarks,)                  # adopted

        # Interferometry
        has_interferometry = ~np.isnan(benchmarks["teff_int"].values)
        median_teff_int_sigma = np.median(
            benchmarks[has_interferometry]["label_adopt_sigma_teff"])
        teff_int_str = \
            r"& Interferometry & {:0.0f}\,K & {:d} & {:d} & {:d} \\"
        teff_int_row = teff_int_str.format(
                median_teff_int_sigma,          # median sigma
                np.sum(has_interferometry),     # with
                np.sum(~has_interferometry),    # without
                np.sum(has_interferometry),)    # adopted

        # Mann+15 + Kesseli+19 (Empirical Relations)
        has_m15_er = ~np.isnan(benchmarks["teff_M15_BP_RP_feh"].values)
        adopted_m15_er = benchmarks["label_source_teff"].values == "M15er"
        median_teff_m15_er_sigma = np.median(
            benchmarks[adopted_m15_er]["label_adopt_sigma_teff"])
        
        ref = r"M (\citealt{mann_how_2015}, \citealt{kesseli_radii_2019})"

        teff_m15_er_row = \
            r"& {} & {:0.0f}\,K & {:d} & {:d} & {:d} \\".format(
                ref,                            # reference
                median_teff_m15_er_sigma,       # median sigma
                np.sum(has_m15_er),             # with
                np.sum(~has_m15_er),            # without
                np.sum(adopted_m15_er),)        # adopted
        
        # Casagrande+21
        has_c21 = ~np.isnan(benchmarks["teff_C21_BP_RP_logg_feh"].values)
        adopted_c21 = benchmarks["label_source_teff"].values == "C21"
        median_teff_c21_sigma = \
            np.median(benchmarks[adopted_c21]["label_adopt_sigma_teff"])
        
        ref = r"K \citep{casagrande_galah_2021}"

        teff_c21_row = \
            r"& {} & {:0.0f}\,K & {:d} & {:d} & {:d} \\".format(
                ref,                            # reference
                median_teff_c21_sigma,          # median sigma
                np.sum(has_c21),                # with
                np.sum(~has_c21),               # without
                np.sum(adopted_c21),)           # adopted
        
        # Add Teff rows
        table_rows += \
            [teff_row, teff_int_row, teff_m15_er_row, teff_c21_row, "\\hline",]

        # ---------------------------------------------------------------------
        # logg
        # ---------------------------------------------------------------------
        # All loggs
        median_logg_sigma = np.median(benchmarks["label_adopt_sigma_logg"])
        logg_row = \
            r"$\log g$ & All & {:0.2f}\,dex & {:d} & {:d} & {:d}\\".format(
            median_logg_sigma,              # median sigma
            n_benchmarks,                   # with
            0,                              # without
            n_benchmarks,)                  # adopted

        # Add logg row
        table_rows += [logg_row, "\\hline",]

        # ---------------------------------------------------------------------
        # [Fe/H] and [X/Fe]
        # ---------------------------------------------------------------------
        for label in labels[2:]:
            chem_row_fmt = r"& {} & {:0.2f}\,dex & {:d} & {:d} & {:d} \\"

            median_chem_sigma = np.nanmedian(
                benchmarks["label_adopt_sigma_{}".format(label)])
            
            chem_row = \
                r"{} & All & {:0.2f}\,dex & {:d} & {:d} & {:d}\\".format(
                    "[{}]".format(label.replace("_", "/")), # Species
                    median_chem_sigma,                      # median sigma
                    n_benchmarks,                           # with
                    0,                                      # without
                    n_benchmarks,)                          # adopted
            
            # Add chemistry summary row
            table_rows.append(chem_row)
            
            for chem_ref in references:
                # Check whether this is a valid label from this source
                chem_col = "{}_{}".format(label,chem_ref)

                if chem_col not in obs_tab.columns.values:
                    continue

                has_ref = ~np.isnan(benchmarks[chem_col].values)

                label_source_col = "label_source_{}".format(label)
                adopted_ref = benchmarks[label_source_col].values == chem_ref

                # If we've adopted zero, skip
                if np.sum(adopted_ref) == 0:
                    continue

                label_sigma_col = "label_adopt_sigma_{}".format(label)
                median_chem_sigma = np.nanmedian(
                    benchmarks[adopted_ref][label_sigma_col].values)
                
                if chem_ref in ref_this_work:
                    citation = "TW"
                else:
                    citation = r"\citet{{{}}}".format(reference_dict[chem_ref])

                chem_ref_row = chem_row_fmt.format(
                        citation,                       # label
                        median_chem_sigma,              # median sigma
                        np.sum(has_ref),                # with
                        np.sum(~has_ref),               # without
                        np.sum(adopted_ref))            # Adopted
                
                # Add chemistry row
                table_rows.append(chem_ref_row)

            table_rows.append("\\hline")
    
    # -------------------------------------------------------------------------
    # Wrapping up
    # -------------------------------------------------------------------------
    # Delete nan values
    for row_i, row in enumerate(table_rows):
        table_rows[row_i] = row.replace("+nan\\,dex", "-")

    # Finish the table
    footer.append("\\end{tabular}}")
    footer.append("\\label{tab:benchmark_sample_summary}")
    footer.append("\\end{table}")

    table = header + table_rows + footer

    # Save table
    if not os.path.isdir(table_folder):
        os.mkdir(table_folder)

    table_fn = os.path.join(table_folder, "table_benchmark_sample_summary.tex")

    np.savetxt(
        fname=table_fn,
        X=table,
        fmt="%s",)


def make_table_benchmark_overview(
    benchmark_df,
    cannon_df,
    label_names,
    references_dict,
    ref_this_work,
    abundance_labels=[],
    break_row=90,
    table_folder="paper",):
    """Make a LaTeX table of our adopted benchmark stellar parameters, the
    source/s of those values, as well as the systematic corrected results for
    the benchmark set.

    Parameters
    ----------
    benchmark_df: pandas DataFrame
        DataFrame of observed + literature information for each star.

    cannon_df: pandas DataFrame
        DataFrame containing the parameter fit information for each star.

    label_names: str list
        List of non-abundance labels. TODO: unify.

    references_dict: dict
        Dictionary mapping these reference abbreviations to the unique 
        identifier used when citing the reference in LaTeX (e.g. mann_how_2015)

    ref_this_work: str list
        List of label sources considered 'This Work' to be replaced with 'TW'.

    abundance_labels: str list
        List of abundance labels.

    break_row: int, default: 90
        Row to break into separate tables.

    table_folder: str, default: 'paper'
        Folder to save the resulting .tex file table to.
    """
    # Temporary merge
    obs_tab = benchmark_df.join(
        cannon_df, "source_id_dr3", rsuffix="_sec").copy()

    # Grab label source columns
    label_source_cols = \
        ["label_source_{}".format(label) for label in label_names]

    n_labels = len(label_names)

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
        abund = "[{}/{}]".format(*tuple(abundance.split("_")))
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
            len(info_cols), n_labels+1, n_labels))
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
            star["label_adopt_teff"], star["label_adopt_sigma_teff"])
        
        # Logg
        table_row += r"${:0.2f}\pm{:0.2f}$ & ".format(
            star["label_adopt_logg"], star["label_adopt_sigma_logg"])

        # [Fe/H]
        if star["label_source_Fe_H"] != "":
            table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                star["label_adopt_Fe_H"], star["label_adopt_sigma_Fe_H"])
        else:
            table_row += r"- & "

        # Abundances
        for abund_i, abund in enumerate(abundance_labels):
            if star["label_source_{}".format(abund)] != "":
                table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                    star["label_adopt_{}".format(abund)],
                    star["label_adopt_sigma_{}".format(abund)])
            else:
                table_row += r"- & "

        # Now do references
        refs = star[label_source_cols].values

        # Note the source as being 'This Work' where appropriate.
        refs = ["TW" if ref in ref_this_work else ref for ref in refs]

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
        table_row += r"${:0.0f}\pm{:0.0f}$ & ".format(
            star["teff_cannon_value"],
            star["teff_cannon_sigma_total"])

        # logg - making sure to flag the star if it has an aberrant logg
        if star["logg_aberrant"]:
            table_row += r"${:0.2f}\pm{:0.2f}$ $\dagger$ & ".format(
                star["logg_cannon_value"],
                star["logg_cannon_sigma_total"])
        else:
            table_row += r"${:0.2f}\pm{:0.2f}$ & ".format(
                star["logg_cannon_value"],
                star["logg_cannon_sigma_total"])

        # [Fe/H]
        table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
            star["Fe_H_cannon_value"],
            star["Fe_H_cannon_sigma_total"])

        # Abundances
        for abund_i, abund in enumerate(abundance_labels):
            # Get label index
            label_i = 3 + abund_i

            table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                star["{}_cannon_value".format(abund)],
                star["{}_cannon_sigma_total".format(abund)])

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
        if ref in references_dict:
            bib_ref = "\\citet{{{}}}".format(references_dict[ref])
        elif ref == "TW":
            bib_ref = "This Work"
        else:
            bib_ref = "-"
        notes_references += "{}: {}, ".format(ref, bib_ref)
    
    # Remove last comma
    notes_references = notes_references[:-2]
    notes.append(notes_references)

    # Add note about dagger symbol
    dagger_note = (
        "\\textbf{Notes:} $\\dagger$: $|\\Delta\\log g|$ aberrant by $> 0.075"
        "\\,$dex compared to literature value.")
    notes.append(dagger_note)
        
    
    notes.append("\\end{minipage}")
    notes.append("\\end{table*}")
    
    # Write the table/s, breaking (if necessary) on break_row
    if break_row > len(obs_tab):
        break_rows = np.array([len(obs_tab)])
    
    else:
        break_rows = np.arange(break_row, len(obs_tab), break_row)

    low_row = 0
    
    # Save table
    if not os.path.isdir(table_folder):
        os.mkdir(table_folder)

    for table_i, break_row in enumerate(break_rows):
        if table_i == 0:
            header = header_1
            footer = footer_1
        else:
            header = header_2
            footer = footer_2
        table_x = header + table_rows[low_row:break_row] + footer + notes

        table_fn = os.path.join(
            table_folder,
            "table_benchmark_params_{:0.0f}.tex".format(table_i))

        np.savetxt(
            fname=table_fn,
            X=table_x,
            fmt="%s",)

        low_row = break_row

    # Do final part table
    if low_row < len(obs_tab):
        table_i += 1
        table_x = header_2 + table_rows[low_row:] + footer_2 + notes

        table_fn = os.path.join(
            table_folder,
            "table_benchmark_params_{:0.0f}.tex".format(table_i))

        np.savetxt(
            fname=table_fn,
            X=table_x,
            fmt="%s",)


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
    aberrant_logg_threshold=0.15,
    bp_mag_col="BP_mag_dr3",
    bp_rp_col="BP_RP_dr3",):
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
        abund = "[{}/Fe]".format(abundance.split("_")[0])
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
    ii = np.argsort(obs_tab[bp_rp_col].values)[::-1]
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
        table_row += "{:0.2f} & ".format(star[bp_mag_col])

        # Colour
        table_row += "{:0.2f} & ".format(star[bp_rp_col])

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
        

def make_theta_comparison_table(
    sms,
    sm_labels,
    br_cutoff_wl=5400,
    print_table=False,):
    """Creates a LaTeX table of standard deviations for each theta coefficient
    for a set of Cannon models for blue, red, and all wavelengths. Table is
    saved to paper/table_theta_std_comp.tex.

    Parameters
    ----------
    sms: list of Stannon objects
        List of trained Cannon models.

    sm_labels: list of str
        List of model names corresponding to smns.

    br_cutoff_wl: float, default: 5400
        Cutoff between blue and red wavelengths.

    print_table: boolean, default: False
        Whether to print a version of the table to the terminal.
    """
    SUPPORTED_ABUNDANCES = ["Ti"]

    header = []
    table_rows = []
    footer = []

    # Construct the header of the table
    header.append("\\begin{table}")
    header.append("\\centering")
    
    col_format = ("cccc")

    header.append("\\begin{tabular}{%s}" % col_format)
    header.insert(2, r"\caption{$\theta$ coeff comparison}")

    # Loop over all provided models
    for model_i, (sm, label) in enumerate(zip(sms, sm_labels)):
        # Setup and format coefficients
        vectorizer = PolynomialVectorizer(sm.label_names, 2)
        theta_lvec = vectorizer.get_human_readable_label_vector()
        
        theta_lvec = theta_lvec.replace("teff", r"$T_{\rm eff}$")
        theta_lvec = theta_lvec.replace("logg", r"$\log g$")
        theta_lvec = theta_lvec.replace("feh", r"${\rm[Fe/H]}$")

        # Math terms
        theta_lvec = theta_lvec.replace("*", r"$\,\times\,$")
        theta_lvec = theta_lvec.replace("^2", r"$^2$")
        
        # Abundances
        theta_lvec = theta_lvec.replace("_Fe", r"/Fe]}$")
        for x in SUPPORTED_ABUNDANCES:
            theta_lvec = theta_lvec.replace(x, r"{}\rm[{}".format("${",x))

        theta_lvec = theta_lvec.split(" + ")

        # Create mask to separate blue and red spectra
        red_mask = sm.masked_wl > br_cutoff_wl

        table_rows.append("\hline")

        table_rows.append((
            r"$\theta_{{\rm {}}}$ & ".format(label) +
            r"\multicolumn{3}{c}{$\sigma_{\theta_N}$}\\"))
        table_rows.append(r"& B3000 & R7000 & All \\")
        table_rows.append("\hline")

        # Loop over all coefficients
        for coeff_i in range(len(theta_lvec)):
            # Grab std of all, blue, and red spectra
            coeff_all = np.std(sm.theta[:,coeff_i])
            coeff_b = np.std(sm.theta[~red_mask, coeff_i])
            coeff_r = np.std(sm.theta[red_mask, coeff_i])
            
            # Create the row
            table_rows.append(
                r"{} & {:0.3f} & {:0.3f} & {:0.3f} \\".format(
                    str(theta_lvec[coeff_i]), coeff_b, coeff_r, coeff_all))
        
        table_rows.append("\hline")

    # Finish the table
    footer.append("\\end{tabular}")
    footer.append(r"\label{tab:theta_std_comp}")
    footer.append("\\end{table}")

    table = header + table_rows + footer
    np.savetxt("paper/table_theta_std_comp.tex", table, fmt="%s")

    if print_table:
        for sm, label in zip(sms, sm_labels):
            print("\n{:>40}".format(
                "theta coeff std for {} label model".format(sm.L)))
            print("-"*40)
            vectorizer = PolynomialVectorizer(sm.label_names, 2)
            theta_lvec = \
                vectorizer.get_human_readable_label_vector().split(" + ")

            print("{:>10}{:>10}{:>10}{:>10}".format("coeff", "b", "r", "all"))
            for coeff_i in range(len(theta_lvec)):
                coeff_all = np.std(sm.theta[:,coeff_i])
                coeff_b = np.std(sm.theta[~red_mask, coeff_i])
                coeff_r = np.std(sm.theta[red_mask, coeff_i])
                print("{:>10}{:10.3f}{:10.3f}{:10.3f}".format(
                    theta_lvec[coeff_i], coeff_b, coeff_r, coeff_all))