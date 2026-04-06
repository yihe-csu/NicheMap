import os
import anndata as ad
import numpy as np
import pandas as pd
import zarr
from scipy.io import mmread
import zipfile




def load_xenium_data(base_dir, anno_file=None, verbose=True):
    """Load Xenium data, compute cell centroids, and optionally merge annotations."""

    mex_dir = os.path.join(base_dir, "cell_feature_matrix")
    matrix_file = os.path.join(mex_dir, "matrix.mtx.gz")
    features_file = os.path.join(mex_dir, "features.tsv.gz")
    barcodes_file = os.path.join(mex_dir, "barcodes.tsv.gz")
    
    cells_zarr_path = os.path.join(base_dir, "cells.zarr")
    cells_zarr_zip_path = os.path.join(base_dir, "cells.zarr.zip")

    is_valid_zarr = os.path.exists(cells_zarr_path) and os.path.exists(os.path.join(cells_zarr_path, "polygon_sets"))

    if not is_valid_zarr and os.path.exists(os.path.join(base_dir, "polygon_sets")):
        cells_zarr_path = base_dir
        is_valid_zarr = True
        if verbose:
            print("Detected Zarr structure directly in base_dir. Using base_dir as Zarr store.")

    nested_zarr = os.path.join(cells_zarr_path, "cells.zarr")
    if not is_valid_zarr and os.path.exists(os.path.join(nested_zarr, "polygon_sets")):
        cells_zarr_path = nested_zarr
        is_valid_zarr = True

    if not is_valid_zarr:
        if os.path.exists(cells_zarr_zip_path):
            if verbose:
                print(f"Extracting {cells_zarr_zip_path} (This may take a moment)...")
            with zipfile.ZipFile(cells_zarr_zip_path, 'r') as zip_ref:

                top_level_items = {item.split('/')[0] for item in zip_ref.namelist()}
                
                if "cells.zarr" in top_level_items:
                    extract_path = base_dir
                else:
                    extract_path = cells_zarr_path
                    os.makedirs(extract_path, exist_ok=True)
                    
                zip_ref.extractall(extract_path)
        else:
            raise FileNotFoundError(f"Missing valid zarr store or zip file in {base_dir}")

    required_paths = [matrix_file, features_file, barcodes_file]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    if verbose:
        print(f"Loading expression matrix from: {mex_dir}")

    X = mmread(matrix_file).tocsr()
    features = pd.read_csv(features_file, sep="\t", header=None)
    barcodes = (
        pd.read_csv(barcodes_file, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    )

    if X.shape[0] == len(features) and X.shape[1] == len(barcodes):
        X = X.T
    elif X.shape[0] != len(barcodes) or X.shape[1] != len(features):
        raise ValueError("Matrix shape does not match features/barcodes.")

    gene_ids = features.iloc[:, 0].astype(str).values
    gene_names = (
        features.iloc[:, 1].astype(str).values
        if features.shape[1] >= 2
        else gene_ids
    )

    var = pd.DataFrame(
        {
            "gene_name": gene_names,
            "gene_id": gene_ids,
        },
        index=gene_names,
    )

    if var.index.duplicated().any():
        counts = {}
        unique_names = []

        for gene in var.index:
            if gene not in counts:
                counts[gene] = 0
                unique_names.append(gene)
            else:
                counts[gene] += 1
                unique_names.append(f"{gene}_{counts[gene]}")

        var.index = unique_names

    if verbose:
        print(f"Parsing spatial polygons from: {cells_zarr_path}")

    z = zarr.open(cells_zarr_path, mode="r")
    polygon_set = z["polygon_sets"]["1"]

    cell_index = polygon_set["cell_index"][:]
    num_vertices = polygon_set["num_vertices"][:]
    vertices = polygon_set["vertices"][:]
    n_cells = z["cell_id"].shape[0]

    if n_cells != len(barcodes):
        raise ValueError(
            f"Cell count mismatch: cells.zarr={n_cells}, barcodes={len(barcodes)}"
        )

    if vertices.shape[1] % 2 != 0:
        raise ValueError("Vertices cannot be reshaped into (x, y) pairs.")

    vertices_xy = vertices.reshape(vertices.shape[0], vertices.shape[1] // 2, 2)
    centroids = np.full((vertices_xy.shape[0], 2), np.nan, dtype=float)

    for i in range(vertices_xy.shape[0]):
        n_vertex = int(num_vertices[i])
        if n_vertex <= 0:
            continue

        points = vertices_xy[i, :n_vertex, :]
        centroids[i, 0] = points[:, 0].mean()
        centroids[i, 1] = points[:, 1].mean()

    df_poly = pd.DataFrame(
        {
            "cell_index": cell_index.astype(int),
            "x_centroid": centroids[:, 0],
            "y_centroid": centroids[:, 1],
        }
    )

    df_cell = df_poly.groupby("cell_index")[["x_centroid", "y_centroid"]].mean()
    df_cell = df_cell.reindex(range(n_cells))
    df_cell.index = pd.Index(barcodes, name="cell_id")

    obs = pd.DataFrame(index=df_cell.index).join(df_cell, how="left")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    if anno_file and os.path.exists(anno_file):
        if verbose:
            print(f"Merging annotation file: {anno_file}")

        anno = pd.read_csv(anno_file, sep=None, engine="python").iloc[:, :2].copy()
        anno.columns = ["cell_id", "annotation"]
        anno["cell_id"] = anno["cell_id"].astype(str)
        anno["annotation"] = anno["annotation"].astype(str)
        anno = anno.drop_duplicates(subset="cell_id")

        annotated_ids = adata.obs_names.intersection(anno["cell_id"])
        adata = adata[annotated_ids].copy()

        anno_map = anno.set_index("cell_id")["annotation"]
        adata.obs["annotation"] = adata.obs_names.map(anno_map)

        if verbose:
            print(f"Finished. Annotated cells: {adata.n_obs}")
    else:
        if verbose:
            print(f"Finished. Total cells: {adata.n_obs}")

    return adata


def calculate_gene_signature_score(
    adata,
    csv_path,
    score_id,
    gene_column="Gene Symbol",
    verbose=True,
):
    """Calculate mean expression score from a gene list CSV."""

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {csv_path}") from exc

    if gene_column not in df.columns:
        raise ValueError(
            f"Column '{gene_column}' not found. Available columns: {df.columns.tolist()}"
        )

    raw_genes = df[gene_column].dropna().astype(str).unique().tolist()
    adata_genes = set(adata.var_names)
    valid_genes = [gene for gene in raw_genes if gene in adata_genes]

    if not valid_genes:
        raise ValueError("No valid genes found in adata.var_names.")

    X = adata[:, valid_genes].X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    adata.obs[score_id] = X.mean(axis=1)

    if verbose:
        print(f"[Score] {score_id}")
        print(f"Genes in CSV: {len(raw_genes)}")
        print(f"Valid genes: {len(valid_genes)}")
        print(f"Stored in adata.obs['{score_id}']")

    return valid_genes