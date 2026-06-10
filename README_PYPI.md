# NicheMap

<p align="center">
  <img src="https://raw.githubusercontent.com/yihe-csu/NicheMap/main/docs/imgs/NicheMap_logo.png" alt="NicheMap logo" width="360" />
</p>

**NicheMap** is a Python toolkit for spatial niche analysis in Xenium and other
coordinate-resolved spatial transcriptomics data.

NicheMap supports two complementary workflows:

- spatial niche identification from marker-gene signature scores
- cell-type neighborhood analysis around annotated spatial structures

## Installation

```bash
pip install nichemap
```

For local development from source:

```bash
git clone https://github.com/yihe-csu/NicheMap.git
cd NicheMap
pip install -e .
```

## Quick Start

### Spatial niche identification

```python
from pathlib import Path

import nichemap as nm

adata = nm.preprocess.load_xenium_data(
    base_dir=r"F:\spatial_data_lung\SSc_1_1_2_raw",
    anno_file=r"F:\spatial_data_lung\ssc112_annotation_map.csv",
)

model = nm.NicheMap(
    adata=adata,
    score_id="ECM_score",
    sample_prefix="SSc_1_1_2",
    out_dir=Path("outputs/ECM_score"),
)

final_adata = model.run(
    gene_list_csv=r"F:\spatial_data_lung\marker_genes\ECM-gene.csv",
    bins=300,
    peak_intensity=1.5,
    exp_intensity=1.0,
)
```

### Cell-type neighborhood analysis

```python
import scanpy as sc
import nichemap.neighborhood as nh

adata = sc.read_h5ad("data/SSc_1_1_2_tutorial.h5ad")

results = nh.run_cell_type_neighborhood_analysis(
    adata=adata,
    target_regions=["Airway_wall"],
    hops=[1, 2, 3],
    structure_col="structure_label",
    cell_type_col="cell_type",
    output_dir="outputs/neighborhood",
    selected_cell_types=None,
    make_plots=True,
)
```

## Documentation and Tutorials

- GitHub repository: https://github.com/yihe-csu/NicheMap
- Online tutorials: https://nichemap-tutorials.readthedocs.io/en/latest/
- Usage guide: https://github.com/yihe-csu/NicheMap/blob/main/How%20to%20use%20NicheMap.md
- Tutorials: https://github.com/yihe-csu/NicheMap/tree/main/Tutorials

## Citation

If you use NicheMap in your work, please cite:

```text
He, Y. et al. NicheMap: a spatial grid-based pipeline for niche identification
in spatial transcriptomics. (Manuscript in preparation)
```

## License

NicheMap is released under the MIT License.
