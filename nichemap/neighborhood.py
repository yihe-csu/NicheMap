import base64
import io
from pathlib import Path
import time

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_category(adata, col):
    """Summarize category counts and percentages from ``adata.obs[col]``."""

    counts = adata.obs[col].value_counts()
    df = counts.rename("count").to_frame()
    df["percentage"] = df["count"] / df["count"].sum() * 100
    df["percentage"] = df["percentage"].round(2)
    df.index.name = None
    return df


def style_summary_table(df, cmap="Blues"):
    """Return a styled category summary table for notebook display."""

    return (
        df.style.format({"count": "{:,}", "percentage": "{:.2f}%"})
        .background_gradient(subset=["count"], cmap=cmap)
        .bar(
            subset=["percentage"],
            color="#E5E7EB",
            vmin=0,
            vmax=df["percentage"].max(),
        )
        .set_properties(
            **{
                "font-size": "13px",
                "text-align": "right",
                "padding": "6px 10px",
                "border": "0px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "13px"),
                        ("font-weight", "600"),
                        ("text-align", "right"),
                        ("background-color", "#F8FAFC"),
                        ("color", "#111827"),
                        ("border", "0px"),
                        ("padding", "6px 10px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "0px"),
                        ("border-bottom", "1px solid #F1F5F9"),
                    ],
                },
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("font-size", "15px"),
                        ("font-weight", "600"),
                        ("text-align", "left"),
                        ("color", "#111827"),
                        ("padding", "6px 0px 10px 0px"),
                    ],
                },
            ]
        )
    )


def spatial_plot_to_base64(
    adata,
    color_col,
    spatial_key="spatial",
    point_size=1.4,
    width=6.5,
    height=5.2,
):
    """Render a spatial category map as a base64 PNG data URL."""

    coords = adata.obsm[spatial_key]
    labels = adata.obs[color_col].astype("category")
    categories = labels.cat.categories.tolist()

    if f"{color_col}_colors" in adata.uns:
        colors = list(adata.uns[f"{color_col}_colors"])
    else:
        colors = plt.cm.tab20.colors

    color_map = {
        category: colors[i % len(colors)] for i, category in enumerate(categories)
    }

    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for category in categories:
        mask = labels == category
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            c=[color_map[category]],
            linewidths=0,
            alpha=0.9,
            label=category,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(
        f"Spatial map colored by {color_col}",
        fontsize=11,
        fontweight="600",
        pad=8,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        frameon=False,
        fontsize=7,
        markerscale=4,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0,
    )

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def annotation_summary_html(
    adata,
    cell_type_col,
    structure_col,
    spatial_key="spatial",
    point_size=1.25,
    width=6.2,
    height=4.6,
):
    """Build an HTML summary of cell types, structures, and spatial layout."""

    cell_type_summary = summarize_category(adata, cell_type_col)
    structure_summary = summarize_category(adata, structure_col)
    cell_html = (
        style_summary_table(cell_type_summary, cmap="Blues")
        .set_caption("Cell-type composition")
        .to_html()
    )
    structure_html = (
        style_summary_table(structure_summary, cmap="Greens")
        .set_caption("Spatial structure composition")
        .to_html()
    )
    spatial_img = spatial_plot_to_base64(
        adata,
        color_col=structure_col,
        spatial_key=spatial_key,
        point_size=point_size,
        width=width,
        height=height,
    )

    return f"""
    <div style="
        display:flex;
        gap:12px;
        align-items:flex-start;
        justify-content:flex-start;
        max-width:none;
    ">
        <div style="width:430px; flex-shrink:0;">
            {cell_html}
        </div>

        <div style="
            width:720px;
            display:flex;
            flex-direction:column;
            gap:12px;
        ">
            <div>
                {structure_html}
            </div>

            <div style="
                background:#FFFFFF;
                border:1px solid #E5E7EB;
                border-radius:10px;
                padding:12px 14px;
                box-shadow:0 1px 2px rgba(0,0,0,0.04);
            ">
                <div style="
                    font-size:15px;
                    font-weight:600;
                    color:#111827;
                    margin-bottom:8px;
                ">
                    Spatial overview
                </div>

                <img src="{spatial_img}" style="
                    width:100%;
                    height:auto;
                    display:block;
                ">
            </div>
        </div>
    </div>
    """


def display_annotation_summary(
    adata,
    cell_type_col,
    structure_col,
    spatial_key="spatial",
    point_size=1.25,
    width=6.2,
    height=4.6,
):
    """Display dataset annotation summaries in a Jupyter notebook."""

    from IPython.display import HTML, Markdown, display

    cell_type_summary = summarize_category(adata, cell_type_col)
    structure_summary = summarize_category(adata, structure_col)

    display(
        Markdown(
            f"""
### Dataset annotation summary

- **Total cells:** {adata.n_obs:,}
- **Cell-type categories:** {cell_type_summary.shape[0]}
- **Spatial structure categories:** {structure_summary.shape[0]}
"""
        )
    )
    display(
        HTML(
            annotation_summary_html(
                adata,
                cell_type_col=cell_type_col,
                structure_col=structure_col,
                spatial_key=spatial_key,
                point_size=point_size,
                width=width,
                height=height,
            )
        )
    )

    return {
        "cell_type_summary": cell_type_summary,
        "structure_summary": structure_summary,
    }


def default_cell_type_palette():
    """Return the default cell-type color palette."""

    return [
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
        "#B09C85",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#E69F00",
        "#009E73",
        "#F0E442",
        "#8DD3C7",
        "#BEBADA",
        "#FB8072",
        "#80B1D3",
        "#FDB462",
        "#B3DE69",
        "#FCCDE5",
        "#BC80BD",
        "#FFED6F",
        "#CCEBC5",
        "#D9D9D9",
    ]


def default_structure_colors():
    """Return generic colors for structure cores, neighbor ring, and background."""

    return {
        "Structure": default_cell_type_palette(),
        "Neighbor_Ring": "#F39B7F",
        "Background": "#F3F4F6",
    }


def set_plot_style():
    """Apply consistent figure style settings."""

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.linewidth"] = 1.0


def _despine(ax):
    try:
        import seaborn as sns

        sns.despine(ax=ax, top=True, right=True, trim=False)
    except ImportError:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def normalize_hops(hops):
    """Normalize a single hop integer or iterable of hops to a list of integers."""

    if isinstance(hops, (int, np.integer)):
        return [int(hops)]

    normalized = [int(hop) for hop in hops]
    if not normalized:
        raise ValueError("hops must contain at least one hop value.")

    return normalized


def get_spatial_adjacency(
    adata,
    spatial_key="spatial",
    connectivity_key="spatial_connectivities",
    n_neighbors=6,
):
    """Return a binary spatial adjacency matrix from an AnnData object."""

    if connectivity_key not in adata.obsp:
        import scanpy as sc

        sc.pp.neighbors(
            adata,
            use_rep=spatial_key,
            n_neighbors=n_neighbors,
            key_added="spatial",
        )

    return (adata.obsp[connectivity_key] > 0).astype(float)


def get_neighbor_mask(adata, adjacency, target_region, hop, structure_col):
    """Return target and hop-neighbor masks for one structure label."""

    target_mask = (adata.obs[structure_col] == target_region).to_numpy()
    expanded = target_mask.astype(float)

    for _ in range(hop):
        expanded = adjacency.dot(expanded)

    neighbor_mask = (expanded > 0) & (~target_mask)
    return target_mask, neighbor_mask


def resolve_cell_types(adata, cell_type_col, selected_cell_types=None):
    """Return selected cell types, or all cell types ordered by abundance."""

    if selected_cell_types is None:
        return adata.obs[cell_type_col].value_counts().index.tolist()

    return list(selected_cell_types)


def calculate_cell_type_proportions(
    adata,
    adjacency,
    target_regions,
    hop,
    cell_type_col,
    structure_col,
    selected_cell_types=None,
):
    """Calculate cell-type proportions in hop-neighbor regions."""

    cell_types = resolve_cell_types(adata, cell_type_col, selected_cell_types)
    results = {}
    summaries = []

    for target in target_regions:
        target_mask, neighbor_mask = get_neighbor_mask(
            adata,
            adjacency,
            target,
            hop,
            structure_col,
        )
        neighbor_cell_types = adata.obs.loc[neighbor_mask, cell_type_col]

        if selected_cell_types is not None:
            neighbor_cell_types = neighbor_cell_types[
                neighbor_cell_types.isin(cell_types)
            ]

        counts = neighbor_cell_types.value_counts().reindex(cell_types).fillna(0)
        total = counts.sum()
        proportions = counts / total * 100 if total > 0 else counts.astype(float)
        results[target] = proportions

        summaries.append(
            {
                "target_region": target,
                "hop": hop,
                "core_cells": int(target_mask.sum()),
                "neighbor_cells": int(neighbor_mask.sum()),
                "counted_cells": int(total),
            }
        )

    return pd.DataFrame(results).fillna(0), pd.DataFrame(summaries)


def result_prefix(selected_cell_types):
    """Return output prefix for all-cell or targeted-cell analysis."""

    return "Targeted" if selected_cell_types is not None else "All"


def save_proportions(df, output_dir, hop, selected_cell_types=None):
    """Save a proportions table and return the written path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"{result_prefix(selected_cell_types)}_CellType_Proportions_{hop}hop.csv"
    )
    df.to_csv(path)
    return path


def color_sequence(items, palette=None):
    """Cycle a color palette over an arbitrary sequence of labels."""

    palette = palette or default_cell_type_palette()
    return [palette[i % len(palette)] for i, _ in enumerate(items)]


def plot_targeted_bar(
    df_props,
    hop,
    output_dir,
    cell_types=None,
    palette=None,
    label_threshold=3.0,
    title=None,
    filename_prefix=None,
):
    """Plot stacked cell-type proportions for one hop distance."""

    set_plot_style()

    if cell_types is None:
        cell_types = df_props.index.tolist()

    df_plot = df_props.reindex(index=cell_types).fillna(0).T
    n_regions = max(1, len(df_plot.index))
    fig_width = max(3.5, 2.7 + 0.8 * n_regions)

    fig, ax = plt.subplots(figsize=(fig_width, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = color_sequence(df_plot.columns, palette)
    df_plot.plot(
        kind="bar",
        stacked=True,
        color=colors,
        ax=ax,
        width=0.6 if n_regions == 1 else 0.72,
        edgecolor="white",
        linewidth=0.6,
    )

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.2f}%" if bar.get_height() > label_threshold else ""
            for bar in container
        ]
        ax.bar_label(
            container,
            labels=labels,
            label_type="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylabel("Proportion of neighboring cells (%)", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.0)
    _despine(ax)

    legend = ax.legend(
        title="Cell type",
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        ncol=1,
        handlelength=1.2,
        handleheight=1.2,
    )
    legend._legend_box.align = "left"

    if title is None:
        title = f"Cell-type composition\n({hop}-hop neighbors)"
    ax.set_title(title, fontsize=11, pad=12)

    if filename_prefix is None:
        filename_prefix = "CellType"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"Barplot_{filename_prefix}_Microenvironment_{hop}hop.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def structure_palette(structure_colors=None):
    """Return the generic structure color palette."""

    colors = structure_colors or default_structure_colors()
    palette = colors.get("Structure", default_cell_type_palette())
    if isinstance(palette, str):
        return [palette]

    return list(palette)


def structure_color_map(target_regions, structure_colors=None):
    """Assign generic structure colors to target regions in order."""

    palette = structure_palette(structure_colors)
    return {target: palette[i % len(palette)] for i, target in enumerate(target_regions)}


def plot_spatial_topology(
    adata,
    adjacency,
    target_regions,
    hop,
    output_dir,
    structure_col,
    spatial_key="spatial",
    structure_colors=None,
):
    """Plot target structures and their hop-neighbor microenvironments."""

    colors = structure_colors or default_structure_colors()
    target_colors = structure_color_map(target_regions, structure_colors)
    coords = adata.obsm[spatial_key]
    paths = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in target_regions:
        target_mask, neighbor_mask = get_neighbor_mask(
            adata,
            adjacency,
            target,
            hop,
            structure_col=structure_col,
        )
        background_mask = (~target_mask) & (~neighbor_mask)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.scatter(
            coords[background_mask, 0],
            coords[background_mask, 1],
            c=colors["Background"],
            s=3,
            edgecolors="none",
            alpha=0.4,
            zorder=1,
        )
        ax.scatter(
            coords[neighbor_mask, 0],
            coords[neighbor_mask, 1],
            c=colors["Neighbor_Ring"],
            s=3,
            edgecolors="none",
            alpha=0.85,
            zorder=2,
            label=f"{hop}-hop Microenvironment",
        )
        ax.scatter(
            coords[target_mask, 0],
            coords[target_mask, 1],
            c=target_colors[target],
            s=3,
            edgecolors="white",
            linewidths=0.1,
            alpha=0.95,
            zorder=3,
            label=f"Core Structure ({target})",
        )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_axis_off()
        ax.legend(loc="upper right", frameon=False, fontsize=11, markerscale=2.0)
        ax.set_title(
            f"Spatial Topology:\n{target} and its Microenvironment",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        path = output_dir / f"Spatial_Topology_{target}_Microenvironment_{hop}hop.png"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def plot_gradient_trend(
    neighborhood_results,
    target_regions,
    hops,
    output_dir,
    cell_types=None,
    palette=None,
):
    """Plot cell-type composition trends across multiple hop distances."""

    set_plot_style()
    paths = []

    if len(hops) < 2:
        return paths

    if cell_types is None:
        cell_types = neighborhood_results[hops[0]].index.tolist()

    colors = color_sequence(cell_types, palette)
    color_map = dict(zip(cell_types, colors))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in target_regions:
        df_plot = pd.DataFrame(
            {f"{hop}-hop": neighborhood_results[hop][target] for hop in hops}
        ).reindex(cell_types).fillna(0)

        fig, ax = plt.subplots(figsize=(4.0, 5.0), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        x_positions = np.arange(len(hops))
        bar_width = 0.5
        values = df_plot.to_numpy()
        tops = df_plot.cumsum(axis=0).to_numpy()
        bottoms = tops - values

        for i, cell_type in enumerate(cell_types):
            ax.bar(
                x_positions,
                values[i],
                bottom=bottoms[i],
                width=bar_width,
                color=color_map[cell_type],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

        for i, cell_type in enumerate(cell_types):
            color = color_map[cell_type]
            for j in range(len(hops) - 1):
                x0 = x_positions[j] + bar_width / 2
                x1 = x_positions[j + 1] - bar_width / 2
                y0_bottom, y0_top = bottoms[i, j], tops[i, j]
                y1_bottom, y1_top = bottoms[i, j + 1], tops[i, j + 1]

                ax.fill_between(
                    [x0, x1],
                    [y0_bottom, y1_bottom],
                    [y0_top, y1_top],
                    color=color,
                    alpha=0.25,
                    edgecolor="none",
                    zorder=1,
                )
                ax.plot(
                    [x0, x1],
                    [y0_top, y1_top],
                    color=color,
                    alpha=0.6,
                    linewidth=1.0,
                    zorder=2,
                )
                ax.plot(
                    [x0, x1],
                    [y0_bottom, y1_bottom],
                    color=color,
                    alpha=0.6,
                    linewidth=1.0,
                    zorder=2,
                )

        ax.set_ylabel("Proportion of neighboring cells (%)", fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{hop}-hop" for hop in hops],
            fontsize=11,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.0)
        _despine(ax)

        custom_handles = [
            mpatches.Patch(color=color_map[cell_type], label=cell_type)
            for cell_type in cell_types
        ]
        legend = ax.legend(
            handles=custom_handles,
            title="Cell type",
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            handlelength=1.2,
            handleheight=1.2,
        )
        legend._legend_box.align = "left"

        ax.set_title(
            f"Spatial Gradient of Niche Composition\n(Target: {target})",
            fontsize=12,
            pad=15,
        )

        path = output_dir / f"Gradient_Trend_{target}_Microenvironment.png"
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close(fig)
        paths.append(path)

    return paths


def display_saved_figures(paths):
    """Display saved image files inline when running in a Jupyter notebook."""

    try:
        from IPython.display import Image, display
    except ImportError:
        return

    for path in paths:
        if Path(path).exists():
            display(Image(filename=str(path)))

def print_nichemap_banner(version="0.1.0"):
    print(
f"""
NicheMap v{version}
Spatial Niche and Neighborhood Analysis Toolkit
Cell Types → Structures → Spatial Niches
{'='*70}
"""
    )


def run_cell_type_neighborhood_analysis(
    adata,
    target_regions,
    hops,
    structure_col,
    cell_type_col,
    output_dir,
    selected_cell_types=None,
    spatial_key="spatial",
    connectivity_key="spatial_connectivities",
    n_neighbors=6,
    make_plots=True,
    display_plots=True,
    show_progress=True,
    progress_desc="Neighborhood analysis",
):
    """Run cell-type neighborhood analysis and optionally export figures."""

    if show_progress:
        print_nichemap_banner()
        print(f"Dataset      : {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"Structures   : {target_regions}")
        print(f"Hops         : {hops}")

        if selected_cell_types is None:
            print("Cell types   : All")
        else:
            print(f"Cell types   : {len(selected_cell_types)} selected")

        print("-" * 70)
    hops = normalize_hops(hops)

    if show_progress:
        print(">>> 1. Building spatial adjacency matrix...")
        start_time = time.time()
    adjacency = get_spatial_adjacency(
        adata,
        spatial_key=spatial_key,
        connectivity_key=connectivity_key,
        n_neighbors=n_neighbors,
    )
    if show_progress:
        elapsed = time.time() - start_time
        print(f"Spatial adjacency matrix is ready. Time used: {elapsed:.2f} seconds")

    cell_types = resolve_cell_types(adata, cell_type_col, selected_cell_types)
    prefix = result_prefix(selected_cell_types)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    neighborhood_results = {}
    summary_tables = []
    csv_paths = []
    figure_paths = []

    iterator = hops
    progress_bar = None
    if show_progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(hops, desc=progress_desc, unit="hop")
        iterator = progress_bar

    for hop in iterator:
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"{hop}-hop proportions")

        df_props, summary = calculate_cell_type_proportions(
            adata,
            adjacency,
            target_regions,
            hop,
            cell_type_col=cell_type_col,
            structure_col=structure_col,
            selected_cell_types=selected_cell_types,
        )
        neighborhood_results[hop] = df_props
        summary_tables.append(summary)
        csv_paths.append(save_proportions(df_props, output_dir, hop, selected_cell_types))

        if make_plots:
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"{hop}-hop figures")

            figure_paths.append(
                plot_targeted_bar(
                    df_props,
                    hop,
                    output_dir,
                    cell_types=cell_types,
                    title=f"{prefix} cell-type composition\n({hop}-hop neighbors)",
                    filename_prefix=prefix,
                )
            )
            figure_paths.extend(
                plot_spatial_topology(
                    adata,
                    adjacency,
                    target_regions,
                    hop,
                    output_dir,
                    structure_col=structure_col,
                    spatial_key=spatial_key,
                )
            )

    if make_plots and len(hops) > 1:
        if progress_bar is not None:
            progress_bar.set_postfix_str("gradient trend")

        figure_paths.extend(
            plot_gradient_trend(
                neighborhood_results,
                target_regions,
                hops,
                output_dir,
                cell_types=cell_types,
            )
        )

    if make_plots and display_plots:
        if progress_bar is not None:
            progress_bar.set_postfix_str("display figures")

        display_saved_figures(figure_paths)

    summary = pd.concat(summary_tables, ignore_index=True)
    return {
        "hops": hops,
        "adjacency": adjacency,
        "cell_types": cell_types,
        "proportions": neighborhood_results,
        "summary": summary,
        "csv_paths": csv_paths,
        "figure_paths": figure_paths,
    }


__all__ = [
    "annotation_summary_html",
    "calculate_cell_type_proportions",
    "color_sequence",
    "default_cell_type_palette",
    "default_structure_colors",
    "display_annotation_summary",
    "display_saved_figures",
    "get_neighbor_mask",
    "get_spatial_adjacency",
    "normalize_hops",
    "plot_gradient_trend",
    "plot_spatial_topology",
    "plot_targeted_bar",
    "resolve_cell_types",
    "result_prefix",
    "run_cell_type_neighborhood_analysis",
    "save_proportions",
    "set_plot_style",
    "spatial_plot_to_base64",
    "structure_color_map",
    "structure_palette",
    "style_summary_table",
    "summarize_category",
]
