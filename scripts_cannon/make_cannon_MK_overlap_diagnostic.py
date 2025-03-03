"""Script to plot a comparison of the label recovery performance of two Cannon
models trained on an overlapping set of benchmark stars plus a model trained on
the entire sample.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import plumage.utils as pu
import stannon.utils as su
import matplotlib.ticker as plticker

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def plot_cannon_overlap_diagnostics(
    obs_join,
    df_M,
    df_K,
    df_MK,
    model_names,
    labels,
    labels_str,
    labels_cb,
    units,
    axis_ticks,
    plot_folder="plots/",):
    """Function to plot a comparison of the label recovery performance of two 
    Cannon models trained on an overlapping set of benchmark stars alongside
    a model trained on the combined sample. The plot is a 2D grid of subplots,
    with n_rows equal to 3 (model A, model B, combined model) and n_cols equal
    to 1+ n_labels (CMD of each sample + one column for each label).

    Parameters
    ----------
    obs_join: pandas DataFrame
        Dataframe containing literature info and adopted labels for each star.

    df_M, df_K, df_MK: pandas DataFrame
        Results DataFrames for the two overlapping and combined Cannon models
        respectively.

    model_names: str list
        List of model names for each Cannon model, of length n_label.
    
    labels, labels_str: str list
        List of labels used in each Cannon model and the LaTeX representation
        of them respectively, of length n_label.

    labels_cb: str list
        Labels to plot on the colour bar of each residual plot (e.g. 'feh' when
        plotting 'teff' residuals), of length n_label.

    units: str list
        Units corresponding to each label, of length n_label.

    axis_ticks: dict of float arrays
        Dictionary with keys equal to labels, mapped to a length 4 float array
        storing the axis ticks [x_major, x_minor, y_major, y_minor].

    plot_folder: str, default: "plots/"
        Folder to save plots to. By default just a subdirectory called plots.
    """
    N_LABELS = len(labels)

    # Initialise plot, 3 rows (M, K, MK), and n_labels + 1 (for CMD) columns
    plt.close("all")
    fig, axes = plt.subplots(
        nrows=3, ncols=N_LABELS+1, figsize=(14,6), sharex='col')
    
    fig.subplots_adjust(
        left=0.075,
        bottom=0.075,
        right=0.98,
        top=0.95,
        hspace=0.1,
        wspace=0.4)

    # Initialise masks
    adopted_M = df_M["adopted_benchmark"].values
    adopted_K = df_K["adopted_benchmark"].values
    adopted_MK = df_MK["adopted_benchmark"].values
    adopted_overlap = np.logical_and(adopted_M, adopted_K)

    models = (df_M, df_K, df_MK)
    masks = (adopted_M, adopted_K, adopted_MK)

    #--------------------------------------------------------------------------
    # Plot CMD for each model
    #--------------------------------------------------------------------------
    # Here we want to plot a colour-magnitude diagram of the benchmark sample
    # used to train this particular model and highlight the overlaping sample.
    for df_i, (df, adopted_mask)  in enumerate(zip(models, masks)):
        N_STAR = np.sum(adopted_mask)

        # Plot benchmarks used for this model
        scatter = axes[df_i,0].scatter(
            obs_join["BP_RP_dr3"].values[adopted_mask],
            obs_join["K_mag_abs"].values[adopted_mask],
            zorder=1,
            c=obs_join["label_adopt_feh"].values[adopted_mask],
            label="{} ({})".format(model_names[df_i], N_STAR),
            alpha=0.9,
            cmap="viridis",)

        # Setup colourbar + ticks
        cb = fig.colorbar(scatter, ax=axes[df_i,0])
        tick_locator = plticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        
        # Only label the topmost colourbar
        if df_i == 0:
            cb.ax.set_title("[Fe/H]")

        # Outline overlap sample
        _ = axes[df_i,0].scatter(
            obs_join["BP_RP_dr3"].values[adopted_overlap],
            obs_join["K_mag_abs"].values[adopted_overlap],
            marker="o",
            c=obs_join["label_adopt_feh"].values[adopted_overlap],
            edgecolor="k",
            linewidths=1.2,
            zorder=1,
            label="{} ({})".format("Overlap", np.sum(adopted_overlap)),)
        
        # Faintly plot the unused sample in the background
        unused = np.logical_and(
            obs_join["is_cannon_benchmark"].values, ~adopted_overlap)

        _ = axes[df_i,0].scatter(
            obs_join["BP_RP_dr3"].values[unused],
            obs_join["K_mag_abs"].values[unused],
            marker="o",
            edgecolor="k",
            facecolor="None",
            linewidths=0.1,
            zorder=-1,)

        axes[df_i,0].legend(loc="best", fontsize="x-small")

        axes[df_i,0].set_ylabel(r"$M_{K_S}$", fontsize="large")

        # X/Y Ticks
        axes[df_i,0].xaxis.set_major_locator(plticker.MultipleLocator(base=1))
        axes[df_i,0].xaxis.set_minor_locator(plticker.MultipleLocator(base=.5))
        axes[df_i,0].yaxis.set_major_locator(plticker.MultipleLocator(base=1))
        axes[df_i,0].yaxis.set_minor_locator(plticker.MultipleLocator(base=.5))

    axes[df_i,0].set_xlabel(r"$BP-RP$", fontsize="large")
    
    # Flip magnitude axis, lock y for all axes
    ymin, ymax = axes[2,0].get_ylim()
    axes[0,0].set_ylim((ymax, ymin))
    axes[1,0].set_ylim((ymax, ymin))
    axes[2,0].set_ylim((ymax, ymin))

    #--------------------------------------------------------------------------
    # Plot residuals for each model+label combo
    #--------------------------------------------------------------------------
    # Here we want to plot the cross-validation residuals for each label.
    models_overlap = (
        df_M[adopted_overlap], df_K[adopted_overlap], df_MK[adopted_overlap])

    info_overlap = obs_join[adopted_overlap]

    # Loop over each label and model
    for label_i, label in enumerate(labels):
        for df_i, df  in enumerate(models_overlap):
            # Grab axis handle for convenience
            axis = axes[df_i, label_i+1]

            # Construct our column names
            label_adopt = "label_adopt_{}".format(label)
            label_cv = "label_cv_{}".format(label)

            # Plot horizontal line at resid = 0
            axis.hlines(
                y=0,
                xmin=info_overlap[label_adopt].min(),
                xmax=info_overlap[label_adopt].max(),
                color="k",
                linestyle="--",)
            
            # Select the colour bar/map, which will be [Fe/H] for all labels
            # except for [Fe/H] itself where we'll use Teff
            colour = (info_overlap["label_adopt_feh"].values if label != "feh"
                      else info_overlap["label_adopt_teff"].values)
            cmap = "viridis" if label != "feh" else "magma"

            # Compute and plot residuals as resid = lit - pred
            resid = info_overlap[label_adopt] - df[label_cv]

            cax = axis.scatter(
                info_overlap[label_adopt], resid, c=colour, cmap=cmap)
            cb = fig.colorbar(cax, ax=axis)
            
            # Annotate systematic + std of the residuals
            resid_med = np.median(resid)
            resid_std = np.std(resid)
            txt = r"${:0.2f} \pm {:0.2f}\,${}".format(
                resid_med, resid_std, units[label_i])
            
            axis.text(
                x=0.5,
                y=0.25,
                s=txt,
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
                bbox=dict(facecolor="grey", edgecolor="None", alpha=0.5),)
            
            # Only have x labels on the bottom
            if df_i == 2:
                axis.set_xlabel(labels_str[label_i])

            # Only label the topmost colourbar
            if df_i == 0:
                cb.ax.set_title(labels_cb[label_i])

            # Y label
            axis.set_ylabel(
                r"$\Delta${}".format(labels_str[label_i]), labelpad=0.0)

            # X/Y Ticks
            axis.xaxis.set_major_locator(plticker.MultipleLocator(
                base=axis_ticks[label][0]))
            axis.xaxis.set_minor_locator(plticker.MultipleLocator(
                base=axis_ticks[label][1]))
            axis.yaxis.set_major_locator(plticker.MultipleLocator(
                base=axis_ticks[label][2]))
            axis.yaxis.set_minor_locator(plticker.MultipleLocator(
                base=axis_ticks[label][3]))

    # Label each row with the name of the model
    for model_i, name in enumerate(model_names):
        axes[model_i,0].annotate(
            name, 
            xy=(0, 0.5),
            xytext=(-axes[model_i,0].yaxis.labelpad - 50, 0),
            xycoords='axes fraction',
            textcoords='offset points',
            size='large',
            ha='center',
            va='baseline')
        
    # Save plot
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)

    plot_fn = os.path.join(plot_folder, "cannon_model_comp_MK")

    plt.savefig("{}.pdf".format(plot_fn))
    plt.savefig("{}.png".format(plot_fn), dpi=300)

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
cannon_settings_yaml = "scripts_cannon/cannon_settings.yml"
cs = su.load_cannon_settings(cannon_settings_yaml)

# Model settings for each of our 3 different Cannon models. Note that they 
# share n_px and n_label.
n_px = 5024
n_label = 3

n_star_MK = 195
n_star_K = 87
n_star_M = 142

# Names of each of the three models
model_names = ("M", "K", "M+K")

# Labels + axis info for plotting later. These should have length >= n_label.
labels = ("teff", "logg", "feh")
labels_str = (r"$T_{\rm eff}$", r"$\log g$", "[Fe/H]")
labels_cb = ("[Fe/H]", "[Fe/H]", r"$T_{\rm eff}$")

units = ("K", "dex", "dex")

axis_ticks = {
    "teff":(200,100,50,25),
    "logg":(0.1,0.05,0.02,0.01),
    "feh":(0.5,0.25,0.1,0.05),}

#------------------------------------------------------------------------------
# Imports and setup
#------------------------------------------------------------------------------
# Import literature info for each benchmark
obs_join = pu.load_fits_table("CANNON_INFO", cs.std_label)

# Import results dataframes for MK, M, and K cases
df_M = pu.load_fits_table(
    extension="CANNON_MODEL",
    label=cs.std_label,
    path=cs.model_save_path,
    ext_label="{}_{}L_{}P_{}S".format("M", n_label, n_px, n_star_M))

df_K = pu.load_fits_table(
    extension="CANNON_MODEL",
    label=cs.std_label,
    path=cs.model_save_path,
    ext_label="{}_{}L_{}P_{}S".format("K", n_label, n_px, n_star_K))

df_MK = pu.load_fits_table(
    extension="CANNON_MODEL",
    label=cs.std_label,
    path=cs.model_save_path,
    ext_label="{}_{}L_{}P_{}S".format("MK", n_label, n_px, n_star_MK))

#------------------------------------------------------------------------------
# Plotting
#------------------------------------------------------------------------------
plot_cannon_overlap_diagnostics(
    obs_join=obs_join,
    df_M=df_M,
    df_K=df_K,
    df_MK=df_MK,
    model_names=model_names,
    labels=labels,
    labels_str=labels_str,
    labels_cb=labels_cb,
    units=units,
    axis_ticks=axis_ticks,
    plot_folder="paper",)