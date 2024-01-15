"""Functions to generate LaTeX tables.
"""
import numpy as np
import stannon.parameters as params
from collections import OrderedDict
from stannon.vectorizer import PolynomialVectorizer

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
    "B16":"brewer_spectral_2016",
    "RB20":"rice_stellar_2020",
    "Sou08":"sousa_spectroscopic_2008",
}

def make_table_sample_summary(obs_tab,):
    """Creates a table summarising where each set of labels comes from, and
    how many are adopted from each sample (e.g. Mann+15).
    """
    col_names = [
        "Label",
        "Sample",
        r"Median $\sigma_{\rm label}$",
        r"Offset",
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

    col_format = "ccccccc"

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

    teff_syst = 0
    logg_syst = 0

    # -------------------------------------------------------------------------
    # Teff
    # -------------------------------------------------------------------------
    # All teffs
    has_default_teff = ~benchmarks["label_nondefault_teff"].values
    median_teff_sigma = \
        np.median(benchmarks[~has_default_teff]["label_adopt_sigma_teff"])
    teff_row = \
        r"$T_{{\rm eff}}$ & All & {:0.0f}\,K & - & {:d} & {:d} & {:d} \\".format(
            median_teff_sigma,              # median sigma
            np.sum(~has_default_teff),      # with
            np.sum(has_default_teff),       # without
            np.sum(~has_default_teff),)     # adopted

    # Interferometry
    has_interferometry = ~np.isnan(benchmarks["teff_int"].values)
    median_teff_int_sigma = \
        np.median(benchmarks[has_interferometry]["label_adopt_sigma_teff"])
    teff_int_row = \
        r"& Interferometry & {:0.0f}\,K & - & {:d} & {:d} & {:d} \\".format(
            median_teff_int_sigma,          # median sigma
            np.sum(has_interferometry),     # with
            np.sum(~has_interferometry),    # without
            np.sum(has_interferometry),)    # adopted

    # Rains+21
    has_r21 = ~np.isnan(benchmarks["teff_synth"].values)
    adopted_21 = benchmarks["label_source_teff"].values == "R21"
    median_teff_r21_sigma = \
        np.median(benchmarks[adopted_21]["label_adopt_sigma_teff"])
    teff_r21_row = \
        r"& Rains+21 & {:0.0f}\,K & - & {:d} & {:d} & {:d} \\".format(
            median_teff_r21_sigma,          # median sigma
            np.sum(has_r21),                # with
            np.sum(~has_r21),               # without
            np.sum(adopted_21),)            # adopted

    # -------------------------------------------------------------------------
    # logg
    # -------------------------------------------------------------------------
    # All loggs
    has_default_logg = ~benchmarks["label_nondefault_logg"].values
    median_logg_sigma = \
        np.median(benchmarks[~has_default_logg]["label_adopt_sigma_logg"])
    logg_row = \
        r"$\log g$ & All & {:0.2f}\,dex & - & {:d} & {:d} & {:d}\\".format(
        median_logg_sigma,              # median sigma
        np.sum(~has_default_logg),      # with
        np.sum(has_default_logg),       # without
        np.sum(~has_default_logg))      # adopted

    # Rains+21
    has_r21 = ~np.isnan(benchmarks["teff_synth"].values)
    adopted_r21 = benchmarks["label_source_logg"].values == "R21"
    median_logg_r21_sigma = \
        np.median(benchmarks[adopted_r21]["label_adopt_sigma_logg"])
    logg_r21_row = \
        r"& Rains+21 & {:0.2f}\,dex & - & {:d} & {:d} & {:d} \\".format(
        median_logg_r21_sigma,          # median sigma
        np.sum(has_r21),                # with
        np.sum(~has_r21),               # without
        np.sum(adopted_r21),)           # adopted

    # -------------------------------------------------------------------------
    # [Fe/H]
    # -------------------------------------------------------------------------
    feh_row_fmt = \
        r"& {} & {:0.2f}\,dex & {:+0.2f}\,dex & {:d} & {:d} & {:d} \\"

    has_default_feh = ~benchmarks["label_nondefault_feh"].values
    median_feh_sigma = \
        np.nanmedian(
            benchmarks[~has_default_feh]["label_adopt_sigma_feh"].values)
    feh_row = \
        r"[Fe/H] & All & {:0.2f}\,dex & - & {:d} & {:d} & {:d}\\".format(
            median_feh_sigma,               # median sigma
            np.sum(~has_default_feh),       # with
            np.sum(has_default_feh),        # without
            np.sum(~has_default_feh))       # adopted

    # Brewer+2016
    has_b16 = ~np.isnan(benchmarks["Fe_H_b16"].values)
    adopted_b16 = benchmarks["label_source_feh"].values == "B16"
    median_feh_b16_sigma = \
        np.nanmedian(benchmarks[adopted_b16]["label_adopt_sigma_feh"].values)
    feh_b16_row = feh_row_fmt.format(
            "Brewer+2016",                  # label
            median_feh_b16_sigma,           # median sigma
            params.FEH_OFFSETS["B16"],      # offset
            np.sum(has_b16),                # with
            np.sum(~has_b16),               # without
            np.sum(adopted_b16))            # Adopted

    # Rice & Brewer 2020
    has_rb20 = ~np.isnan(benchmarks["Fe_H_rb20"].values)
    adopted_rb20 = benchmarks["label_source_feh"].values == "RB20"
    median_feh_rb20_sigma = \
        np.nanmedian(benchmarks[adopted_b16]["label_adopt_sigma_feh"].values)
    feh_rb20_row = feh_row_fmt.format(
            "Rice \& Brewer 2020",           # label
            median_feh_rb20_sigma,           # median sigma
            params.FEH_OFFSETS["RB20"],      # offset
            np.sum(has_rb20),                # with
            np.sum(~has_rb20),               # without
            np.sum(adopted_rb20))            # Adopted

    # Valenti Fischer 2005
    has_vf05 = ~np.isnan(benchmarks["Fe_H_vf05"].values)
    adopted_vf05 = benchmarks["label_source_feh"].values == "VF05"
    median_feh_vf05_sigma = \
        np.nanmedian(benchmarks[adopted_vf05]["label_adopt_sigma_feh"].values)
    feh_vf05_row = feh_row_fmt.format(
            "Valenti \& Fischer 2005",       # label
            median_feh_vf05_sigma,           # median sigma
            params.FEH_OFFSETS["VF05"],      # offset
            np.sum(has_vf05),                # with
            np.sum(~has_vf05),               # without
            np.sum(adopted_vf05))            # Adopted
    
    # Montes+2018
    has_m18 = ~np.isnan(benchmarks["Fe_H_lit_m18"].values)
    adopted_m18 = benchmarks["label_source_feh"].values == "M18"
    median_feh_m18_sigma = \
        np.nanmedian(benchmarks[adopted_m18]["label_adopt_sigma_feh"].values)
    feh_m18_row = feh_row_fmt.format(
            "Montes+2018",                  # label
            median_feh_m18_sigma,           # median sigma
            params.FEH_OFFSETS["M18"],      # offset
            np.sum(has_m18),                # with
            np.sum(~has_m18),               # without
            np.sum(adopted_m18))            # Adopted

    # Sousa+2008 - TODO incomplete crossmatch ATM, can't compute offset
    adopted_s08 = benchmarks["label_source_feh"].values == "Sou08"
    median_feh_s08_sigma = \
        np.nanmedian(benchmarks[adopted_s08]["label_adopt_sigma_feh"].values)
    feh_s08_row = \
        r"& {} & {:0.2f}\,dex & - & - & - & {:d} \\".format(
            "Sousa+2008",                   # label
            median_feh_s08_sigma,           # median sigma
            np.sum(adopted_s08))            # Adopted
    
    # Mann+2015
    has_m15 = ~np.isnan(benchmarks["feh_m15"].values)
    adopted_m15 = benchmarks["label_source_feh"].values == "M15"
    median_feh_m15_sigma = \
        np.nanmedian(benchmarks[adopted_m15]["label_adopt_sigma_feh"].values)
    feh_m15_row = feh_row_fmt.format(
            "Mann+2015",                    # label
            median_feh_m15_sigma,           # median sigma
            params.FEH_OFFSETS["M13"],      # offset
            np.sum(has_m15),                # with
            np.sum(~has_m15),               # without
            np.sum(adopted_m15))            # Adopted

    # Rojas-Ayala+2012
    has_ra12 = ~np.isnan(benchmarks["feh_ra12"].values)
    adopted_ra12 = benchmarks["label_source_feh"].values == "RA12"
    median_feh_ra12_sigma = \
        np.nanmedian(benchmarks[adopted_ra12]["label_adopt_sigma_feh"].values)
    feh_ra12_row = feh_row_fmt.format(
            "Rojas-Ayala+2012",             # label
            median_feh_ra12_sigma,          # median sigma
            params.FEH_OFFSETS["RA12"],     # offset
            np.sum(has_ra12),               # with
            np.sum(~has_ra12),              # without
            np.sum(adopted_ra12))           # adopted

    # Other NIR
    has_other = ~np.isnan(benchmarks["feh_nir"].values)
    adopted_other = np.logical_or(
        benchmarks["label_source_feh"].values == "G14",
        benchmarks["label_source_feh"].values == "T15")
    median_feh_other_sigma = \
        np.nanmedian(benchmarks[adopted_other]["label_adopt_sigma_feh"].values)
    feh_other_row = \
        r"& Other NIR & {:0.2f}\,dex & - & - & - & {:d} \\".format(
            median_feh_other_sigma,          # median sigma
            np.sum(adopted_other))           # adopted

    # Photometric
    has_photometric = ~np.isnan(benchmarks["phot_feh"].values)
    adopted_photometric = benchmarks["label_source_feh"].values == "R21"
    median_feh_photometric_sigma = \
        np.nanmedian(
            benchmarks[adopted_photometric]["label_adopt_sigma_feh"].values)
    feh_photometric_row = feh_row_fmt.format(
            "Photometric",                   # label
            median_feh_photometric_sigma,    # median sigma
            params.FEH_OFFSETS["R21"],       # offset
            np.sum(has_photometric),         # with
            np.sum(~has_photometric),        # without
            np.sum(adopted_photometric))     # adopted

    # -------------------------------------------------------------------------
    # [Ti/Fe]
    # -------------------------------------------------------------------------
    has_ti_fe = np.isfinite(benchmarks["label_adopt_Ti_Fe"].values)
    median_ti_sigma = np.nanmedian(benchmarks["label_adopt_sigma_Ti_Fe"].values)
    ti_row = \
        r"[Ti/Fe] & All & {:0.2f}\,dex & - & {:d} & {:d} & {:d} \\".format(
            median_ti_sigma,
            np.sum(has_ti_fe), 
            np.sum(~has_ti_fe),
            np.sum(has_ti_fe))

    # Brewer+2016
    has_tih_b16 = ~np.isnan(benchmarks["Ti_H_b16"].values)
    adopted_tih_b16 = benchmarks["label_source_Ti_Fe"].values == "B16"
    median_tih_b16_sigma = \
        np.median(benchmarks[adopted_tih_b16]["label_adopt_sigma_Ti_Fe"])
    ti_b16_row = feh_row_fmt.format(
            "Brewer+2016",                 # label
            median_tih_b16_sigma,          # median sigma
            params.TIH_OFFSETS["B16"],     # offset
            np.sum(has_tih_b16),           # with
            np.sum(~has_tih_b16),          # without
            np.sum(adopted_tih_b16),)      # adopted
    
    # Rice & Brewer 2020
    has_tih_rb20 = ~np.isnan(benchmarks["Ti_H_rb20"].values)
    adopted_tih_rb20 = benchmarks["label_source_Ti_Fe"].values == "RB20"
    median_tih_rb20_sigma = \
        np.median(benchmarks[adopted_tih_rb20]["label_adopt_sigma_Ti_Fe"])
    ti_rb20_row = feh_row_fmt.format(
            "Rice \& Brewer 2020",          # label
            median_tih_rb20_sigma,          # median sigma
            params.TIH_OFFSETS["RB20"],     # offset
            np.sum(has_tih_rb20),           # with
            np.sum(~has_tih_rb20),          # without
            np.sum(adopted_tih_rb20),)      # adopted

    # Valenti Fischer 2005
    has_tih_vf05 = ~np.isnan(benchmarks["Ti_H_vf05"].values)
    adopted_tih_vf05 = benchmarks["label_source_Ti_Fe"].values == "VF05"
    median_tih_vf05_sigma = \
        np.median(benchmarks[adopted_tih_vf05]["label_adopt_sigma_Ti_Fe"])
    ti_vf05_row = feh_row_fmt.format(
            "Valenti \& Fischer 2005",      # label
            median_tih_vf05_sigma,          # median sigma
            params.TIH_OFFSETS["VF05"],     # offset
            np.sum(has_tih_vf05),           # with
            np.sum(~has_tih_vf05),          # without
            np.sum(adopted_tih_vf05),)      # adopted

    # Montes+2018
    has_tih_m18 = ~np.isnan(benchmarks["Ti_H_m18"].values)
    adopted_tih_m18 = benchmarks["label_source_Ti_Fe"].values == "M18"
    median_tih_m18_sigma = \
        np.median(benchmarks[adopted_tih_m18]["label_adopt_sigma_Ti_Fe"])
    ti_m18_row = feh_row_fmt.format(
            "Montes+2018",                  # label
            median_tih_m18_sigma,           # median sigma
            params.TIH_OFFSETS["M18"],      # offset
            np.sum(has_tih_m18),            # with
            np.sum(~has_tih_m18),           # without
            np.sum(adopted_tih_m18),)       # adopted
    
    # Adibekyan+2012 (TODO: incomplete cross-match)
    has_tih_a12 = ~np.isnan(benchmarks["TiI_H_a12"].values)
    adopted_tih_a12 = benchmarks["label_source_Ti_Fe"].values == "A12"
    median_tih_a12_sigma = \
        np.median(benchmarks[adopted_tih_a12]["label_adopt_sigma_Ti_Fe"])
    ti_a12_row = \
        r"& {} & {:0.2f}\,dex & - & - & - & {:d} \\".format(
            "Adibekyan+2012",               # label
            median_tih_a12_sigma,           # median sigma
            #params.TIH_OFFSETS["A12"],      # offset
            #np.sum(has_tih_a12),            # with
            #np.sum(~has_tih_a12),           # without
            np.sum(adopted_tih_a12),)       # adopted
    
    # Empirical Relation
    has_ti_fe_monty = ~np.isnan(benchmarks["Ti_Fe_monty"].values)
    adopted_ti_fe_monty = benchmarks["label_source_Ti_Fe"].values == "R22a"
    median_ti_fe_monty_sigma = \
        np.median(benchmarks[adopted_tih_m18]["label_adopt_sigma_Ti_Fe"])
    ti_monty_row = feh_row_fmt.format(
            "This Work",                        # label
            median_ti_fe_monty_sigma,           # median sigma
            params.Ti_Fe_OFFSETS["Monty"],        # offset
            np.sum(has_ti_fe_monty),            # with
            np.sum(~has_ti_fe_monty),           # without
            np.sum(adopted_ti_fe_monty),)       # adopted
    
    # Put all rows together
    table_rows = [
        teff_row,
        teff_int_row,
        teff_r21_row,
        "\hline",
        logg_row,
        logg_r21_row,
        "\hline",
        feh_row,
        feh_b16_row,
        feh_rb20_row,
        feh_vf05_row,
        feh_m18_row,
        feh_s08_row,
        feh_m15_row,
        feh_ra12_row,
        feh_other_row,
        feh_photometric_row,
        "\hline",
        ti_row,
        ti_b16_row,
        ti_rb20_row,
        ti_vf05_row,
        ti_m18_row,
        ti_a12_row,
        ti_monty_row,]
    
    # Delete nan values
    for row_i, row in enumerate(table_rows):
        table_rows[row_i] = row.replace("+nan\\,dex", "-")

    # Finish the table
    footer.append("\\hline")
    footer.append("\\end{tabular}}")
    footer.append("\\label{tab:benchmark_sample_summary}")
    footer.append("\\end{table}")

    table = header + table_rows + footer

    np.savetxt(
        fname="paper/table_benchmark_sample_summary.tex",
        X=table,
        fmt="%s",)


def make_table_benchmark_overview(
    obs_tab,
    label_names,
    abundance_labels=[],
    break_row=61,):
    """Make a LaTeX table of our adopted benchmark stellar parameters, the
    source/s of those values, as well as the systematic corrected results for
    the benchmark set.
    """
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
        if star["label_source_feh"] != "":
            table_row += r"${:+0.2f}\pm{:0.2f}$ & ".format(
                star["label_adopt_feh"], star["label_adopt_sigma_feh"])
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

        # TODO HACK: Delete
        refs = [ref.replace("R22a", "TW") for ref in refs]

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
            star["feh_cannon_value"],
            star["feh_cannon_sigma_total"])

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
        if ref in label_source_refs:
            bib_ref = "\\citet{{{}}}".format(label_source_refs[ref])
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