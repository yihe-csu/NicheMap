import os
import sys

import matplotlib.pyplot as plt
import nichemap
import numpy as np
import pandas as pd
import scanpy as sc


sys.path.append(os.path.abspath("C://Users//heyi//Desktop/NicheMap-main"))


# -----------------------------------------------------------------------------
# 1. Define input paths and analysis parameters
# -----------------------------------------------------------------------------
base_dir = r"F:\spatial_data_lung\SSc_1_1_2_raw"
anno_file = r"F:\spatial_data_lung\ssc112_annotation_map.csv"
gene_list = r"F:\spatial_data_lung\marker_genes\ECM-gene.csv"

score_id = "ECM_score"
bins = 300
peak_intensity = 1.5
exp_intensity = 1.0

out_dir = rf"F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\{score_id}"
os.makedirs(out_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Load Xenium data
# -----------------------------------------------------------------------------
adata = nichemap.preprocess.load_xenium_data(
    base_dir=base_dir,
    anno_file=anno_file,
)
print(adata)


# -----------------------------------------------------------------------------
# 3. Normalize data and calculate gene signature score
# -----------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

ecm_genes_list = nichemap.preprocess.calculate_gene_signature_score(
    adata=adata,
    csv_path=gene_list,
    score_id=score_id,
    gene_column="Gene Symbol",
)

nichemap.plot.plot_spatial_score(
    adata,
    score_name=score_id,
    out_dir=out_dir,
)


# -----------------------------------------------------------------------------
# 4. Build spatial grid and generate score maps
# -----------------------------------------------------------------------------
sigma_peak = 2

mean_map, smooth_map_peak, counts, xedges, yedges = nichemap.utils.generate_mean_grid_map(
    adata,
    score_id=score_id,
    bins=bins,
    sigma_peak=sigma_peak,
)

nichemap.plot.plot_grid_map(
    mean_map,
    xedges,
    yedges,
    cmap="inferno",
    title=f"Figure 1A: Raw {score_id} grid map",
    cbar_label=f"Mean {score_id} per grid",
    out_dir=out_dir,
)

nichemap.plot.plot_grid_map(
    counts,
    xedges,
    yedges,
    cmap="viridis",
    title="Figure 1B: Grid cell density",
    cbar_label="Cell count per grid",
    out_dir=out_dir,
)


# -----------------------------------------------------------------------------
# 5. Detect niche seed peaks
# -----------------------------------------------------------------------------
_, _, peak_coords, results_df = nichemap.utils.find_peaks(
    smooth_map_peak,
    mode="heuristic",
    intensity_sigma=peak_intensity,
    use_otsu_base=True,
)

markers, peak_x, peak_y = nichemap.plot.visualize_and_export_peaks(
    smooth_map_peak,
    peak_coords,
    xedges,
    yedges,
    out_dir=out_dir,
)

x = adata.obs["x_centroid"].values
y = adata.obs["y_centroid"].values

peak_x, peak_y = nichemap.plot.plot_peak_positions_on_scatter(
    x_col=x,
    y_col=y,
    peak_coords=peak_coords,
    xedges=xedges,
    yedges=yedges,
    out_dir=out_dir,
)


# -----------------------------------------------------------------------------
# 6. Create expansion mask and segment niche regions
# -----------------------------------------------------------------------------
smooth_map_exp, niche_mask = nichemap.utils.create_expansion_mask(
    mean_map,
    sigma=2,
    mode="heuristic",
    expansion_sigma=exp_intensity,
    use_otsu_base=True,
)

nichemap.plot.plot_expansion_mask(
    smooth_map_exp,
    niche_mask,
    peak_x,
    peak_y,
    xedges,
    yedges,
    out_dir=out_dir,
)

niche_labels, niche_ids = nichemap.utils.segment_niche_regions(
    smooth_map_exp,
    markers,
    niche_mask,
)

nichemap.plot.plot_niche_map(
    smooth_map_exp,
    niche_labels,
    peak_x,
    peak_y,
    xedges,
    yedges,
    cmap_base="magma",
    cmap_labels="Set3",
    out_dir=out_dir,
)


# -----------------------------------------------------------------------------
# 7. Map niche labels back to cells
# -----------------------------------------------------------------------------
nichemap.utils.map_niche_to_cells(
    adata,
    niche_labels,
    xedges,
    yedges,
    x_col="x_centroid",
    y_col="y_centroid",
    output_col=f"{score_id}_niche_id",
    verbose=True,
)

nichemap.plot.plot_cell_level_niches(
    adata,
    niche_labels,
    xedges,
    yedges,
    niche_column=f"{score_id}_niche_id",
    coords_columns=("x_centroid", "y_centroid"),
    s_bg=1,
    s_fg=3,
    cmap="tab20",
    boundary_color="cyan",
    out_dir=out_dir,
    verbose=True,
)


# -----------------------------------------------------------------------------
# 8. Export results
# -----------------------------------------------------------------------------
nichemap.utils.export_niche_results(
    adata=adata,
    out_dir=out_dir,
    niche_column=f"{score_id}_niche_id",
    file_prefix="SSc_1_1_2",
    export_csv=True,
    export_h5ad=True,
)