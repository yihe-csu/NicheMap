# How to Use NicheMap

This guide is a practical usage manual. For the project overview, method summary,
figures, citation, and general background, see `README.md`.

## 1. Prepare a Python Environment

NicheMap is intended for Python 3.10 or newer.

Using conda:

```bash
conda create -n nichemap python=3.10
conda activate nichemap
```

Using venv:

```bash
python -m venv nichemap

# Windows
nichemap\Scripts\activate

# Linux / macOS
source nichemap/bin/activate
```

## 2. Install NicheMap Locally

From the project root:

```bash
cd C:\Users\heyi\Desktop\NicheMap
pip install -e .
```

Editable installation is recommended for local development because changes in
the `nichemap/` package are immediately available in Python sessions.

If you only want to install dependencies without installing the package:

```bash
pip install -r requirements.txt
```

For notebooks, prefer `pip install -e .` over manually adding the project path.
Use `sys.path.append(...)` only for temporary debugging.

## 3. Verify Installation

Run this from Python:

```python
import nichemap as nm
import nichemap.neighborhood as nh

print("NicheMap imported successfully.")
print(nh.normalize_hops([1, 2, 3]))
```

Expected output:

```text
NicheMap imported successfully.
[1, 2, 3]
```

## 4. Run the Main NicheMap Pipeline

Use this workflow when starting from raw Xenium data and a marker gene list.

```python
from pathlib import Path

import nichemap as nm

sample_prefix = "SSc_1_1_2"
base_dir = r"F:\spatial_data_lung\SSc_1_1_2_raw"
anno_file = r"F:\spatial_data_lung\ssc112_annotation_map.csv"
gene_list = r"F:\spatial_data_lung\marker_genes\ECM-gene.csv"

score_id = "ECM_score"
out_dir = Path(r"F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result") / score_id
out_dir.mkdir(parents=True, exist_ok=True)

adata = nm.preprocess.load_xenium_data(
    base_dir=base_dir,
    anno_file=anno_file,
)

model = nm.NicheMap(
    adata=adata,
    score_id=score_id,
    sample_prefix=sample_prefix,
    out_dir=out_dir,
)

final_adata = model.run(
    gene_list_csv=gene_list,
    bins=300,
    peak_intensity=1.5,
    exp_intensity=1.0,
)
```

Main outputs include spatial score maps, grid maps, detected niche seeds,
segmented niche regions, cell-level niche labels, CSV metadata, and processed
`.h5ad` files.

For a line-by-line version, see:

```text
Tutorials/NicheMap_Lung_data_Xenium_step_by_step.py
Tutorials/NicheMap_Lung_data_Xenium_step_by_step.ipynb
```

## 5. Run Cell-Type Neighborhood Analysis

Use this workflow when you already have an AnnData object with:

- spatial coordinates in `adata.obsm["spatial"]`
- a spatial connectivity graph in `adata.obsp["spatial_connectivities"]`, or
  enough spatial coordinates for Scanpy to build one
- a structure annotation column, such as `adata.obs["structure_label"]`
- a cell-type annotation column, such as `adata.obs["cell_type"]`

Example with the tutorial dataset:

```python
from pathlib import Path

import scanpy as sc
import nichemap.neighborhood as nh

adata = sc.read_h5ad("data/SSc_1_1_2_tutorial.h5ad")

results = nh.run_cell_type_neighborhood_analysis(
    adata=adata,
    target_regions=["Airway_wall"],
    hops=[1, 2, 3],
    structure_col="structure_label",
    cell_type_col="cell_type",
    output_dir=Path("Tutorials/outputs/neighborhood"),
    selected_cell_types=None,
    make_plots=True,
    display_plots=True,
    show_progress=True,
)

results["summary"]
results["proportions"][3]
```

Set `selected_cell_types` to a list to focus on selected cell types:

```python
selected_cell_types = [
    "Fibroblast",
    "Macrophage",
    "T cell",
    "B cell",
    "Plasma",
    "Myofibroblast",
    "Monocyte",
]
```

Neighborhood analysis outputs include:

- cell-type proportion CSV files for each hop
- stacked bar plots
- spatial topology plots for each target structure and hop
- a gradient trend plot when multiple hops are used

For the notebook version, see:

```text
Tutorials/NicheMap_neighborhood_tutorial.ipynb
```

## 6. Important Parameters

| Parameter | Used in | Meaning |
| --- | --- | --- |
| `base_dir` | main pipeline | Raw Xenium directory containing expression matrix and `cells.zarr` |
| `anno_file` | main pipeline | Optional cell annotation CSV |
| `gene_list_csv` | main pipeline | Marker gene CSV used for signature scoring |
| `score_id` | main pipeline | Name of the score column stored in `adata.obs` |
| `bins` | main pipeline | Spatial grid resolution |
| `peak_intensity` | main pipeline | Seed detection threshold strength |
| `exp_intensity` | main pipeline | Expansion mask threshold strength |
| `target_regions` | neighborhood | Structure labels used as neighborhood centers |
| `hops` | neighborhood | Graph-hop distances, such as `1` or `[1, 2, 3]` |
| `structure_col` | neighborhood | Column in `adata.obs` containing spatial structure labels |
| `cell_type_col` | neighborhood | Column in `adata.obs` containing cell-type labels |
| `selected_cell_types` | neighborhood | `None` for all cell types, or a list of target cell types |

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'nichemap'`

Install the package from the project root:

```bash
pip install -e .
```

Then restart the Python kernel or terminal.

### `ModuleNotFoundError: No module named 'scanpy'`

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

### `KeyError: 'spatial'`

The AnnData object does not contain spatial coordinates in `adata.obsm["spatial"]`.
Check available coordinate keys:

```python
adata.obsm.keys()
```

### `KeyError: 'structure_label'` or `KeyError: 'cell_type'`

The configured annotation column does not exist. Check available columns:

```python
adata.obs.columns
```

Then update `structure_col` or `cell_type_col`.

### Figures are saved but not shown in a notebook

Use:

```python
display_plots=True
```

in `run_cell_type_neighborhood_analysis`.

### Notebook uses old package code after editing `nichemap/`

Restart the kernel, or reload the module during development:

```python
import importlib
import nichemap.neighborhood as nh

importlib.reload(nh)
```
