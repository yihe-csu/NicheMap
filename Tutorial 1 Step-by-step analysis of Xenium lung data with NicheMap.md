# Tutorial 1: Step-by-step analysis of Xenium lung data with NicheMap

This tutorial demonstrates how to use `NicheMap` step by step on Xenium lung data.

In this example, we use a fibrosis-related gene set (`ECM_score`) to identify spatial niche regions from a Xenium dataset.

---

## 1. Import packages

```python
import os
import sys

sys.path.append(os.path.abspath("C://Users//heyi//Desktop/NicheMap-main"))

import matplotlib.pyplot as plt
import nichemap
import numpy as np
import pandas as pd
import scanpy as sc
```

## 2. Define input paths and analysis parameters

```python
base_dir = r"F:\spatial_data_lung\SSc_1_1_2_raw"
anno_file = r"F:\spatial_data_lung\ssc112_annotation_map.csv"
gene_list = r"F:\spatial_data_lung\marker_genes\ECM-gene.csv"

score_id = "ECM_score"
bins = 300
peak_intensity = 1.5
exp_intensity = 1.0

out_dir = rf"F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\{score_id}"
os.makedirs(out_dir, exist_ok=True)
```

## 3. Load Xenium data

```py
adata = nichemap.preprocess.load_xenium_data(
    base_dir=base_dir,
    anno_file=anno_file,
)
print(adata)
```

```python
Loading expression matrix from: F:\spatial_data_lung\SSc_1_1_2_raw\cell_feature_matrix
Parsing spatial polygons from: F:\spatial_data_lung\SSc_1_1_2_raw\cells.zarr
Merging annotation file: F:\spatial_data_lung\ssc112_annotation_map.csv
Finished. Annotated cells: 82224
AnnData object with n_obs × n_vars = 82224 × 541
    obs: 'x_centroid', 'y_centroid', 'annotation'
    var: 'gene_name', 'gene_id'
    obsm: 'spatial'
```

## 4. Normalize data and calculate gene signature score

```python
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
```

```python
[Score] ECM_score
Genes in CSV: 183
Valid genes: 17
Stored in adata.obs['ECM_score']
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Spatial_ECM_score.png
```

## 5. Build spatial grid and generate score maps

```python
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
```

```python
[Grid] ECM_score
Shape: (300, 300), Non-empty: 31560
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_1A_Raw_ECM_score_grid_map.png
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_1B_Grid_cell_density.png
```

## 6. Detect niche seed peaks

```python
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
```

```python
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_2A_Detected_niche_peaks.png
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_2B_Peak_positions_on_spatial_map.png
```

## 7. Create expansion mask and segment niche regions

```python
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
```

```
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_3A_Candidate_expansion_region.png
[Watershed] Niche count: 14
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_4A_Spatial_Niche_Segmentation.png
```

## 8. Map niche labels back to cells

```python
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
```

```python
[Assign] ECM_score_niche_id
ECM_score_niche_id
0     57110
9      6523
3      4624
2      3423
14     3045
Name: count, dtype: int64
Computing vector boundaries for niche sketching...
Computing niche centroids for in situ labeling...
Figure saved to F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\ECM_score\Figure_4B_Cell-level_niche_assignment.png
```

## 9. Export results

```python
nichemap.utils.export_niche_results(
    adata=adata,
    out_dir=out_dir,
    niche_column=f"{score_id}_niche_id",
    file_prefix="SSc_1_1_2",
    export_csv=True,
    export_h5ad=True,
)
```

```
[Export] Assigned: 25114/82224
```