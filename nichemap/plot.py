import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke
from skimage.measure import regionprops


def _sanitize_filename(name):
    """Convert a plot title into a safe file name."""

    return name.replace(":", "").replace(" ", "_").replace("/", "_")


def _save_figure(fig, out_dir, name, dpi=600):
    """Save figure if output directory is provided."""

    if not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{_sanitize_filename(name)}.png")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {save_path}")


def _set_plot_style():
    """Apply a consistent plotting style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )


def visualize_and_export_peaks(
    smooth_map_peak,
    peak_coords,
    xedges,
    yedges,
    title="Figure 2A: Detected niche peaks",
    cmap="inferno",
    marker_color="cyan",
    figsize=(16, 8),
    out_dir=None,
):
    """Visualize detected peaks and return watershed markers."""

    markers = np.zeros_like(smooth_map_peak, dtype=int)
    for i, (row, col) in enumerate(peak_coords, start=1):
        markers[row, col] = i

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    peak_x = [x_centers[row] for row, col in peak_coords]
    peak_y = [y_centers[col] for row, col in peak_coords]

    _set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=600)

    im = ax.imshow(
        smooth_map_peak.T,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        interpolation="none",
    )

    ax.scatter(
        peak_x,
        peak_y,
        s=35,
        c=marker_color,
        edgecolors="black",
        linewidths=0.8,
        label="Seed Peaks",
        zorder=10,
    )

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Smoothed Intensity", fontsize=11, fontweight="bold")
    cbar.outline.set_linewidth(0.8)

    ax.set_title(
        f"{title} (n={len(peak_coords)})",
        fontsize=14,
        fontweight="bold",
        pad=15,
        loc="left",
    )

    if len(peak_coords) > 0:
        ax.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)

    plt.tight_layout()
    _save_figure(fig, out_dir, title)
    # plt.show()

    return markers, peak_x, peak_y


def plot_peak_positions_on_scatter(
    x_col,
    y_col,
    peak_coords,
    xedges,
    yedges,
    title="Figure 2B: Peak positions on spatial map",
    marker_color="#FF3333",
    figsize=(12, 10),
    out_dir=None,
):
    """Plot detected peak positions on the raw spatial scatter."""

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    peak_x = [x_centers[row] for row, col in peak_coords]
    peak_y = [y_centers[col] for row, col in peak_coords]

    _set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    ax.scatter(
        x_col,
        y_col,
        s=1,
        c="#EAEAEA",
        edgecolors="none",
        label="All Cells/Spots",
        rasterized=True,
        zorder=1,
    )

    ax.scatter(
        peak_x,
        peak_y,
        s=80,
        c=marker_color,
        edgecolors="black",
        linewidths=0.8,
        label="Detected Peaks",
        zorder=10,
    )

    text_effect = [withStroke(linewidth=2.5, foreground="white")]
    for i, (px, py) in enumerate(zip(peak_x, peak_y), start=1):
        ax.text(
            px,
            py - 100,
            str(i),
            color="black",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            path_effects=text_effect,
            zorder=11,
        )

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{title}\n$n = {len(peak_coords)}$ peaks identified",
        fontsize=14,
        fontweight="bold",
        pad=15,
        loc="left",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=2)

    plt.tight_layout()
    _save_figure(fig, out_dir, title)
    # plt.show()

    return peak_x, peak_y


def plot_expansion_mask(
    smooth_map_expand,
    candidate_mask,
    peak_x,
    peak_y,
    xedges,
    yedges,
    title="Figure 3A: Candidate expansion region",
    figsize=(12, 10),
    out_dir=None,
):
    """Visualize expansion intensity, candidate mask, and peak labels."""

    _set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    im = ax.imshow(
        smooth_map_expand.T,
        origin="lower",
        cmap="inferno",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        interpolation="none",
        zorder=1,
    )

    masked_data = np.ma.masked_where(~candidate_mask, candidate_mask)
    ax.imshow(
        masked_data.T,
        origin="lower",
        cmap="Greens",
        alpha=0.4,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        interpolation="none",
        zorder=2,
    )

    text_effect = [withStroke(linewidth=2.5, foreground="black")]
    for i, (px, py) in enumerate(zip(peak_x, peak_y), start=1):
        ax.text(
            px,
            py,
            str(i),
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            path_effects=text_effect,
            zorder=11,
        )

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Expansion Map Intensity", fontsize=11, fontweight="bold")
    cbar.outline.set_linewidth(0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, loc="left")

    plt.tight_layout()
    _save_figure(fig, out_dir, title)
    # plt.show()


def plot_niche_map(
    smooth_map_expand,
    labels_final,
    peak_x,
    peak_y,
    xedges,
    yedges,
    title="Figure 4A: Spatial Niche Segmentation",
    cmap_base="Greys_r",
    cmap_labels="Set3",
    scale_bar_um=None,
    figsize=(12, 10),
    out_dir=None,
):
    """Plot final niche segmentation over the expansion map."""

    _set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    vmin, vmax = np.percentile(smooth_map_expand, [1, 99])

    im = ax.imshow(
        smooth_map_expand.T,
        origin="lower",
        cmap=cmap_base,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        zorder=1,
    )

    masked_labels = np.ma.masked_where(labels_final == 0, labels_final)
    ax.imshow(
        masked_labels.T,
        origin="lower",
        cmap=cmap_labels,
        alpha=0.35,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        interpolation="none",
        zorder=2,
    )

    for niche_id in np.unique(labels_final):
        if niche_id == 0:
            continue

        region_mask = labels_final == niche_id
        if np.sum(region_mask) <= 4:
            continue

        ax.contour(
            region_mask.T.astype(float),
            levels=[0.5],
            colors="#00FFFF",
            linewidths=1.0,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            zorder=3,
        )

    ax.scatter(
        peak_x,
        peak_y,
        s=25,
        c="white",
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
    )

    text_effect = [withStroke(linewidth=2.5, foreground="black")]
    for prop in regionprops(labels_final):
        niche_id = prop.label
        if niche_id == 0:
            continue

        row, col = prop.centroid
        idx_row = int(np.clip(round(row), 0, len(x_centers) - 1))
        idx_col = int(np.clip(round(col), 0, len(y_centers) - 1))

        ax.text(
            x_centers[idx_row],
            y_centers[idx_col],
            str(niche_id),
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            path_effects=text_effect,
            zorder=11,
        )

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    n_niches = np.sum(np.unique(labels_final) != 0)
    ax.set_title(
        f"{title}\n$n = {n_niches}$ niche regions identified",
        fontsize=14,
        fontweight="bold",
        pad=15,
        loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.outline.set_linewidth(0.8)
    cbar.set_label("Expansion Signal Intensity", fontsize=11, fontweight="bold")

    if scale_bar_um:
        try:
            from matplotlib_scalebar.scalebar import ScaleBar

            scalebar = ScaleBar(
                1,
                "um",
                length_fraction=0.15,
                location="lower right",
                box_alpha=0,
                color="black",
                font_properties={"size": 10, "weight": "bold"},
            )
            ax.add_artist(scalebar)
        except ImportError:
            print(
                "matplotlib-scalebar is not installed. "
                "Run `pip install matplotlib-scalebar` to enable scale bars."
            )

    plt.tight_layout()
    _save_figure(fig, out_dir, title)
    # plt.show()


def plot_cell_level_niches(
    adata,
    labels_final,
    xedges,
    yedges,
    niche_column="ECM_niche_id",
    coords_columns=("x_centroid", "y_centroid"),
    title="Figure 4B: Cell-level niche assignment",
    s_bg=1,
    s_fg=3,
    cmap="Set3",
    boundary_color="#00FFFF",
    figsize=(12, 12),
    out_dir=None,
    verbose=True,
):
    """Plot cell-level niche assignments with region boundaries and labels."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    x_col, y_col = coords_columns
    adata.obs[niche_column] = adata.obs[niche_column].astype(int)

    mask_bg = (adata.obs[niche_column] == 0) | (adata.obs[niche_column].isna())
    mask_fg = ~mask_bg

    ax.scatter(
        adata.obs.loc[mask_bg, x_col],
        adata.obs.loc[mask_bg, y_col],
        s=s_bg,
        c="#EAEAEA",
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    ax.scatter(
        adata.obs.loc[mask_fg, x_col],
        adata.obs.loc[mask_fg, y_col],
        s=s_fg,
        c=adata.obs.loc[mask_fg, niche_column],
        cmap=cmap,
        edgecolors="none",
        alpha=0.6,
        rasterized=True,
        zorder=5,
    )

    if verbose:
        print("Computing vector boundaries for niche sketching...")

    for niche_id in np.unique(labels_final):
        if niche_id == 0:
            continue

        region_mask = labels_final == niche_id
        if np.sum(region_mask) < 9:
            continue

        ax.contour(
            region_mask.T.astype(float),
            levels=[0.5],
            colors=boundary_color,
            linewidths=0.9,
            linestyles="solid",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            zorder=7,
        )

    if verbose:
        print("Computing niche centroids for in situ labeling...")

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    text_effect = [withStroke(linewidth=2.0, foreground="black")]

    for prop in regionprops(labels_final):
        niche_id = prop.label
        if niche_id == 0:
            continue

        row, col = prop.centroid
        idx_row = int(np.clip(round(row), 0, len(x_centers) - 1))
        idx_col = int(np.clip(round(col), 0, len(y_centers) - 1))

        ax.text(
            x_centers[idx_row],
            y_centers[idx_col],
            str(niche_id),
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            path_effects=text_effect,
            zorder=15,
        )

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    n_niches = np.sum(np.unique(labels_final) != 0)
    ax.set_title(
        f"{title}\n$n = {n_niches}$ niche regions validated at cell level",
        fontsize=14,
        fontweight="bold",
        pad=20,
        loc="left",
    )

    plt.tight_layout()
    _save_figure(fig, out_dir, title)
    # plt.show()


def plot_spatial_score(
    adata,
    score_name="ECM_score_mean",
    x_col="x_centroid",
    y_col="y_centroid",
    s=2,
    cmap="RdBu_r",
    figsize=(16, 8),
    out_dir=None,
):
    """Plot spatial distribution of a cell-level score."""

    _set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    scatter = ax.scatter(
        adata.obs[x_col],
        adata.obs[y_col],
        c=adata.obs[score_name],
        s=s,
        cmap=cmap,
        edgecolors="none",
        rasterized=True,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(score_name, fontsize=11, fontweight="bold")
    cbar.outline.set_linewidth(0.8)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(
        f"Spatial distribution of {score_name}",
        fontsize=14,
        fontweight="bold",
        pad=15,
        loc="left",
    )
    ax.set_xlabel("x (μm)", fontsize=11)
    ax.set_ylabel("y (μm)", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()
    _save_figure(fig, out_dir, f"Spatial_{score_name}")
    # plt.show()


def plot_grid_map(
    map_data,
    xedges,
    yedges,
    cmap="inferno",
    title="Step 1A: Raw ECM grid map",
    cbar_label="Mean ECM score per grid",
    figsize=(16, 8),
    out_dir=None,
):
    """Plot a 2D heatmap for grid-based data."""

    _set_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    im = ax.imshow(
        map_data.T,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        interpolation="none",
    )

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")
    cbar.outline.set_linewidth(0.8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, loc="left")

    plt.tight_layout()
    _save_figure(fig, out_dir, title)
    # plt.show()