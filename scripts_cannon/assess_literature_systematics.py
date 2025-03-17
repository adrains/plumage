"""Script to assess systematics and overlap between different literature
chemistry benchmarks.
"""
import numpy as np
import pandas as pd

samples = {
    "VF05":"data/VF05.tsv",             # Const sigma for: all
    "A12":"data/A12_gaia_all.tsv",      # Const sigma for: Fe
    "RA12":"data/RA12_dr3_all.tsv",     # Individual sigmas
    "M15":"data/mann15_all_dr3.tsv",    # Individual sigmas
    "B16":"data/B16_dr3_all.tsv",       # Const sigma for: all
    "M18":"data/montes18_prim.tsv",     # Individual sigmas
    "L18":"data/luck18_all_dr3.tsv",    # Individual sigmas
    "RB20":"data/RB20_dr3_all.tsv",     # Const sigma for: all
}

species_all = {
    "VF05":["M", "Na", "Si", "Ti", "Fe", "Ni"],
    "A12":["Na", "Mg", "Al", "Si", "Ca", "Ti", "Cr", "Ni", "Mn", "Fe", "Co", 
           "Sc", "Mn", "V"],
    "RA12":["M", "Fe"],
    "M15":["Fe"],
    "B16":["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "V", "Cr", "Mn",
           "Fe", "Ni", "Y"],
    "M18":["Fe", "Na", "Mg", "Al", "Si", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
           "Co", "Ni"],
    # Note: ignoring Eu since it lacks an uncertainty
    "L18":["Na", "Mg", "Al", "Si", "S", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
           "Fe", "Co", "Ni", "Cu", "Zn", "Sr", "Y", "Zr", "Ba", "La", "Ce",
           "Nd", "Sm",], # "Eu",]
    "RB20":["C", "N", "O", "Na", "Mg", "Al", "Si", "Ca", "Ti", "V", "Cr", "Mn",
            "Fe", "Ni", "Y"],
}

sigmas = {
    "VF05":{"M":0.029,
            "Na":0.032,
            "Si":0.019,
            "Ti":0.046,
            "Fe":0.03,
            "Ni":0.03,},
    "A12":{"Fe":0.03,},     # From paper, pg. 2, section 2.
    "B16":{"C":0.026,
            "N":0.042,
            "O":0.036,
            "Na":0.014,
            "Mg":0.012,
            "Al":0.028,
            "Si":0.008,
            "Ca":0.014,
            "Ti":0.012,
            "V":0.034,
            "Cr":0.014,
            "Mn":0.020,
            "Fe":0.010,
            "Ni":0.012,
            "Y":0.03,},
                          # limits
    "RB20":{"C":0.05,     # −0.60–0.64
            "N":0.09,     # −0.86–0.84
            "O":0.07,     # −0.36–0.77
            "Na":0.06,    # −1.09–0.78
            "Mg":0.03,    # −0.70–0.54
            "Al":0.05,    # −0.66–0.58
            "Si":0.03,    # −0.65–0.57
            "Ca":0.03,    # −0.73–0.54
            "Ti":0.03,    # −0.71–0.52
            "V":0.04,     # −0.85–0.46
            "Cr":0.03,    # −1.07–0.52
            "Mn":0.05,    # −1.40–0.66
            "Fe":0.02,    # −0.99–0.57
            "Ni":0.03,    # −0.97–0.63
            "Y":0.07,}    # −0.87–1.35
}


#------------------------------------------------------------------------------
# Initial Import + Unique IDs
#------------------------------------------------------------------------------
all_ids = []
dataframes = {}

for ref in samples.keys():
    df = pd.read_csv(
        samples[ref],
        delimiter="\t",
        dtype={"source_id":str, "source_id_dr3":str},)
    df.rename(columns={"source_id":"source_id_dr3"}, inplace=True)
    df.set_index("source_id_dr3", inplace=True)

    all_ids += df.index.values.tolist()
    dataframes[ref] = df

# Collect unique IDs, remove nans and '-' values
unique_ids = set(all_ids)
unique_ids.remove("-")
unique_ids.remove(np.nan)

# Create new DataFrame to be used for crossmatching to later
df_comb = pd.DataFrame(data=unique_ids, columns="source_id_dr3")
df_comb.set_index("source_id_dr3", inplace=True)

#------------------------------------------------------------------------------
# Collate columns of interest
#------------------------------------------------------------------------------
dataframes_cut = {}

#=========================================
# Valenti & Fischer 2005
#=========================================
VF05 = dataframes["VF05"]
n_VF05 = len(VF05)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["VF05"]]
abund_cols_new = ["{}_H_VF05".format(ss) for ss in species_all["VF05"]]

VF05.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["VF05"]:
    e_species = np.full(n_VF05, sigmas["VF05"][species])
    sigma_col_new = "e_{}_H_VF05".format(species)
    VF05[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["VF05"] = VF05[cols].copy()

#=========================================
# Adibekyan+2012
#=========================================
A12 = dataframes["A12"]

# Some columns have been corrected for a temperature systematic (see S3.2). 
# Swap the column names such that the *corrected* columns match the standard
# naming format, and the uncorrected columns have a subscript 'uc'.
species_to_correct = ["Na", "Al", "ScI", "TiI", "V", "CrII"]

for species in species_to_correct:
    col_base = "[{}/H]".format(species)
    col_corr = "[{}/H]c".format(species)
    col_uncorr = "[{}/H]uc".format(species)
    A12.rename(columns={col_base:col_uncorr}, inplace=True)
    A12.rename(columns={col_corr:col_base}, inplace=True)

# We need to combine the ionisation states and uncertainties for some species
species_to_combine = ["Sc", "Ti", "Cr"]

for species in species_to_combine:
    n_X_I = A12["o_[{}I/H]".format(species)].values
    X_I_log10 = A12["[{}I/H]".format(species)].values
    e_X_I_log10 = A12["e_[{}I/H]".format(species)].values

    X_I = 10**X_I_log10
    e_X_I = X_I * np.log(10) * e_X_I_log10
    
    n_X_II = A12["o_[{}II/H]".format(species)].values
    X_II_log10 = A12["[{}II/H]".format(species)].values
    e_X_II_log10 = A12["e_[{}II/H]".format(species)].values

    X_II = 10**X_II_log10
    e_X_II = X_II * np.log(10) * e_X_II_log10

    # Compute combined abundance and propagate uncertainty
    #   A = ((n1 * A1) + (n2 * A2)) / (n1 + n2)
    X_H = ((n_X_I * X_I) + (n_X_II * X_II)) / (n_X_I + n_X_II)
    e_X_H = np.sqrt(
        (n_X_I / (n_X_I + n_X_II) * e_X_I)**2
        + (n_X_II / (n_X_I + n_X_II) * e_X_II)**2)
    
    X_H_log10 = np.log10(X_H)
    e_X_H_log10 = e_X_H / (X_H * np.log(10))

    # Add back to dataframe
    A12["[{}/H]".format(species)] = X_H_log10
    A12["e_[{}/H]".format(species)] = e_X_H_log10

# Add in [Fe/H] uncertainty, (sigma = 0.03, from paper, pg. 2, section 2).
A12["e_[Fe/H]"] = 0.03

# Now that we have a standard set of columns, grab and rename
abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["A12"]]
abund_cols_new = ["{}_H_A12".format(ss) for ss in species_all["A12"]]
A12.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["A12"]]
sigma_cols_new = ["e_{}_H_A12".format(ss) for ss in species_all["A12"]]
A12.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["A12"] = A12[cols].copy()

#=========================================
# Rojas-Ayala+2012
#=========================================
RA12 = dataframes["RA12"]

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["RA12"]]
abund_cols_new = ["{}_H_RA12".format(ss) for ss in species_all["RA12"]]
RA12.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["RA12"]]
sigma_cols_new = ["e_{}_H_RA12".format(ss) for ss in species_all["RA12"]]
RA12.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["RA12"] = RA12[cols].copy()

#=========================================
# Mann+2015
#=========================================
M15 = dataframes["M15"]

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["M15"]]
abund_cols_new = ["{}_H_M15".format(ss) for ss in species_all["M15"]]
M15.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["M15"]]
sigma_cols_new = ["e_{}_H_M15".format(ss) for ss in species_all["M15"]]
M15.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["M15"] = M15[cols].copy()

#=========================================
# Brewer+2016
#=========================================
B16 = dataframes["B16"]
n_B16 = len(B16)

abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["B16"]]
abund_cols_new = ["{}_H_B16".format(ss) for ss in species_all["B16"]]

B16.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["B16"]:
    e_species = np.full(n_B16, sigmas["B16"][species])
    sigma_col_new = "e_{}_H_B16".format(species)
    B16[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["B16"] = B16[cols].copy()

#=========================================
# Montes+2018
#=========================================
M18 = dataframes["M18"]

abund_cols_old = ["{}_H".format(ss) for ss in species_all["M18"]]
abund_cols_new = ["{}_H_M18".format(ss) for ss in species_all["M18"]]
M18.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e{}_H".format(ss) for ss in species_all["M18"]]
sigma_cols_new = ["e_{}_H_M18".format(ss) for ss in species_all["M18"]]
M18.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["M18"] = M18[cols].copy()

#=========================================
# Luck 18
#=========================================
# Before doing anything, we need to compute the solar reference values
L18_sun = pd.read_csv("data/L18_solar.tsv", delimiter="\t", index_col="ID")

x_h_solar = {}

for species in species_all["L18"]:
    abund_X_I_col = "log{}I".format(species)
    n_lines_X_I_col = "o_log{}I".format(species)

    abund_X_II_col = "log{}II".format(species)
    n_lines_X_II_col = "o_log{}II".format(species)

    # Don't have ionised state
    if abund_X_II_col not in L18_sun.columns.values:
        x_h_solar[species] = L18_sun.loc["Sun"][abund_X_I_col]

    # Don't have ground state
    elif abund_X_I_col not in L18_sun.columns.values:
        x_h_solar[species] = L18_sun.loc["Sun"][abund_X_II_col]

    # Otherwise have both ionisation states do a weighted average to for [X/H]
    else:
        X_I = L18_sun.loc["Sun"][abund_X_I_col]
        n_X_I = L18_sun.loc["Sun"][n_lines_X_I_col]

        X_II = L18_sun.loc["Sun"][abund_X_II_col]
        n_X_II = L18_sun.loc["Sun"][n_lines_X_II_col]

        X_H = ((n_X_I * X_I) + (n_X_II * X_II)) / (n_X_I + n_X_II)

        x_h_solar[species] = X_H

#=========================================
# Now we can continue to our sample proper
L18 = dataframes["L18"]
n_L18 = len(L18)

# We need to combine the uncertainties for some species across ionisation
# states. Unlike with A12, however, we already have [X/H], just not e_[X/H].
species_to_combine = ["Si", "Ca", "Sc", "Ti", "V", "Cr", "Fe", "Y", "Zr",]

for species in species_to_combine:
    n_X_I = L18["o_log{}I".format(species)].values
    X_I_log10 = L18["log{}I".format(species)].values
    e_X_I_log10 = L18["e_log{}I".format(species)].values

    X_I = 10**X_I_log10
    e_X_I = X_I * np.log(10) * e_X_I_log10

    # Divide by solar value
    #X_I /= 10**x_h_solar[species]
    #e_X_I /= 10**x_h_solar[species]
    
    n_X_II = L18["o_log{}II".format(species)].values
    X_II_log10 = L18["log{}II".format(species)].values
    e_X_II_log10 = L18["e_log{}II".format(species)].values

    X_II = 10**X_II_log10
    e_X_II = X_II * np.log(10) * e_X_II_log10

    # Divide by solar value
    #X_II /= 10**x_h_solar[species]
    #e_X_II /= 10**x_h_solar[species]

    # Compute combined abundance and propagate uncertainty
    #   A = ((n1 * A1) + (n2 * A2)) / (n1 + n2)
    X_H = ((n_X_I * X_I) + (n_X_II * X_II)) / (n_X_I + n_X_II)
    e_X_H = np.sqrt(
        (n_X_I / (n_X_I + n_X_II) * e_X_I)**2
        + (n_X_II / (n_X_I + n_X_II) * e_X_II)**2)
    
    X_H_log10 = np.log10(X_H)
    e_X_H_log10 = e_X_H / (X_H * np.log(10))

    # Add back to dataframe
    L18["[{}/H]_calc".format(species)] = X_H_log10
    L18["e_[{}/H]".format(species)] = e_X_H_log10

# Rename columns from only a single ionisation state for consistency
single_state_species = [
    "NaI", "MgI", "AlI", "SI", "MnI", "CoI", "NiI", "CuI", "ZnI", "SrI",
    "BaII", "LaII", "CeII", "NdII", "SmII",]# "EuII"]

for species in single_state_species:
    abund_col = "[{}/H]".format(species)
    sigma_col = "e_log{}".format(species)

    abund_col_new = abund_col.replace("I", "")
    sigma_col_new = "e_{}".format(abund_col_new)

    L18.rename(
        columns={abund_col:abund_col_new, sigma_col:sigma_col_new},
        inplace=True)

# Now that we have a standard set of columns, grab and rename
abund_cols_old = ["[{}/H]".format(ss) for ss in species_all["L18"]]
abund_cols_new = ["{}_H_L18".format(ss) for ss in species_all["L18"]]
L18.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_old = ["e_[{}/H]".format(ss) for ss in species_all["L18"]]
sigma_cols_new = ["e_{}_H_L18".format(ss) for ss in species_all["L18"]]
L18.rename(columns=dict(zip(sigma_cols_old, sigma_cols_new)), inplace=True)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["L18"] = L18[cols].copy()

#=========================================
# Rice & Brewer 2020
#=========================================
RB20 = dataframes["RB20"]
n_RB20 = len(RB20)

abund_cols_old = ["{}/H".format(ss) for ss in species_all["RB20"]]
abund_cols_new = ["{}_H_RB20".format(ss) for ss in species_all["RB20"]]

RB20.rename(columns=dict(zip(abund_cols_old, abund_cols_new)), inplace=True)

sigma_cols_new = []

# Add sigma columns one at a time
for species in species_all["RB20"]:
    e_species = np.full(n_RB20, sigmas["RB20"][species])
    sigma_col_new = "e_{}_H_RB20".format(species)
    RB20[sigma_col_new] = e_species
    sigma_cols_new.append(sigma_col_new)

# Create new subset dataframe of just abundances and uncertainties
cols = [val for pair in zip(abund_cols_new, sigma_cols_new) for val in pair]
dataframes_cut["RB20"] = RB20[cols].copy()

#------------------------------------------------------------------------------
# Join all separate tables
#------------------------------------------------------------------------------
for ref in dataframes_cut.keys():
    df_comb = df_comb.join(dataframes_cut[ref], "source_id_dr3",)
