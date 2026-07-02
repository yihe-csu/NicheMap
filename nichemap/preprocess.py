import os
import anndata as ad
import numpy as np
import pandas as pd
import zarr
from scipy.io import mmread
from scipy import sparse
from tqdm.auto import tqdm
import zipfile




def load_xenium_data(base_dir, anno_file=None, verbose=True):
    """Load Xenium data, compute cell centroids, and optionally merge annotations.

    The default path reads a standard Xenium output directory containing
    ``cell_feature_matrix`` and ``cells.zarr``. If those files are not present
    and a ``*_transcripts.csv.gz`` table is found, the function falls back to
    transcript-level aggregation for GEO/GSM-style data folders.
    """

    base_dir = os.fspath(base_dir)

    mex_dir = os.path.join(base_dir, "cell_feature_matrix")
    matrix_file = os.path.join(mex_dir, "matrix.mtx.gz")
    features_file = os.path.join(mex_dir, "features.tsv.gz")
    barcodes_file = os.path.join(mex_dir, "barcodes.tsv.gz")

    cells_zarr_path = os.path.join(base_dir, "cells.zarr")
    cells_zarr_zip_path = os.path.join(base_dir, "cells.zarr.zip")

    is_valid_zarr = os.path.exists(cells_zarr_path) and os.path.exists(
        os.path.join(cells_zarr_path, "polygon_sets")
    )

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
            transcript_file = _find_xenium_transcript_file(base_dir)
            if transcript_file is not None:
                if verbose:
                    print(
                        "Standard Xenium matrix/zarr files were not found. "
                        "Detected transcript-table format instead."
                    )
                return load_xenium_transcript_data(
                    base_dir=base_dir,
                    transcript_file=transcript_file,
                    anno_file=anno_file,
                    verbose=verbose,
                )
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

    adata = _merge_annotation(adata, anno_file=anno_file, verbose=verbose)

    return adata


def load_xenium_transcript_data(
    base_dir,
    transcript_file=None,
    anno_file=None,
    chunk_size=500_000,
    min_qv=None,
    coordinate_method="mean",
    verbose=True,
    show_progress=True,
):
    """Load GEO/GSM-style Xenium data from a transcript CSV table.

    This loader aggregates transcript rows into a sparse cell-by-gene matrix and
    computes cell spatial coordinates as the mean transcript position per cell.
    """

    if coordinate_method != "mean":
        raise ValueError("Only coordinate_method='mean' is supported.")

    base_dir = os.fspath(base_dir)
    transcript_file = (
        os.fspath(transcript_file)
        if transcript_file is not None
        else _find_xenium_transcript_file(base_dir)
    )
    if transcript_file is None:
        raise FileNotFoundError(
            f"No transcript table found in {base_dir}. Expected "
            "'*transcripts.csv.gz' or '*transcripts.csv'."
        )
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Missing transcript table: {transcript_file}")

    if verbose:
        print(f"Loading transcript table from: {transcript_file}")

    preview = pd.read_csv(transcript_file, nrows=5)
    required_cols = ["cell_id", "feature_name", "x_location", "y_location"]
    missing = [col for col in required_cols if col not in preview.columns]
    if missing:
        raise ValueError(f"Transcript table is missing required columns: {missing}")

    usecols = required_cols.copy()
    optional_cols = [
        "qv",
        "overlaps_nucleus",
        "sample",
        "old_sample_name",
        "fov_name",
    ]
    for col in optional_cols:
        if col in preview.columns:
            usecols.append(col)

    count_parts = []
    coord_parts = []
    qc_parts = []
    n_rows = 0
    n_assigned_rows = 0

    reader = pd.read_csv(
        transcript_file,
        usecols=usecols,
        chunksize=chunk_size,
        dtype={"cell_id": "string", "feature_name": "string"},
    )
    chunk_reader = (
        tqdm(reader, desc="Reading transcripts", unit="chunk")
        if verbose and show_progress
        else reader
    )

    for chunk_i, chunk in enumerate(chunk_reader, start=1):
        n_rows += len(chunk)
        chunk = chunk.dropna(
            subset=["cell_id", "feature_name", "x_location", "y_location"]
        )
        chunk["cell_id"] = chunk["cell_id"].astype(str)
        chunk["feature_name"] = chunk["feature_name"].astype(str)

        assigned_mask = ~chunk["cell_id"].str.endswith("_UNASSIGNED", na=False)
        assigned_mask &= chunk["cell_id"] != "UNASSIGNED"
        chunk = chunk.loc[assigned_mask].copy()

        if min_qv is not None and "qv" in chunk.columns:
            chunk = chunk.loc[
                pd.to_numeric(chunk["qv"], errors="coerce") >= min_qv
            ].copy()

        n_assigned_rows += len(chunk)
        if chunk.empty:
            _update_transcript_progress(chunk_reader, n_rows, n_assigned_rows)
            continue

        chunk["x_location"] = pd.to_numeric(chunk["x_location"], errors="coerce")
        chunk["y_location"] = pd.to_numeric(chunk["y_location"], errors="coerce")
        chunk = chunk.dropna(subset=["x_location", "y_location"])
        if chunk.empty:
            _update_transcript_progress(chunk_reader, n_rows, n_assigned_rows)
            continue

        counts = (
            chunk.groupby(["cell_id", "feature_name"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        count_parts.append(counts)

        coords = (
            chunk.groupby("cell_id", observed=True)
            .agg(
                x_sum=("x_location", "sum"),
                y_sum=("y_location", "sum"),
                transcript_count=("feature_name", "size"),
                x_min=("x_location", "min"),
                x_max=("x_location", "max"),
                y_min=("y_location", "min"),
                y_max=("y_location", "max"),
            )
            .reset_index()
        )
        coord_parts.append(coords)

        qc_agg = {}
        if "overlaps_nucleus" in chunk.columns:
            chunk["overlaps_nucleus"] = pd.to_numeric(
                chunk["overlaps_nucleus"], errors="coerce"
            )
            qc_agg["n_nucleus_transcripts"] = ("overlaps_nucleus", "sum")
        for text_col in ["sample", "old_sample_name"]:
            if text_col in chunk.columns:
                qc_agg[text_col] = (
                    text_col,
                    lambda s: s.dropna().astype(str).iloc[0]
                    if len(s.dropna())
                    else "",
                )
        if "fov_name" in chunk.columns:
            qc_agg["fov_names"] = (
                "fov_name",
                lambda s: ";".join(sorted(s.dropna().astype(str).unique())),
            )

        if qc_agg:
            qc = chunk.groupby("cell_id", observed=True).agg(**qc_agg).reset_index()
        else:
            qc = pd.DataFrame({"cell_id": coords["cell_id"]})
        qc_parts.append(qc)

        _update_transcript_progress(
            chunk_reader,
            n_rows,
            n_assigned_rows,
            cells=coords["cell_id"].nunique(),
            cell_gene_pairs=len(counts),
        )

    if len(count_parts) == 0:
        raise RuntimeError("No assigned transcripts were found; cannot build AnnData.")

    counts_df = pd.concat(count_parts, ignore_index=True)
    counts_df = (
        counts_df.groupby(["cell_id", "feature_name"], observed=True)["count"]
        .sum()
        .reset_index()
    )

    coord_df = pd.concat(coord_parts, ignore_index=True)
    obs_coord = coord_df.groupby("cell_id", observed=True).agg(
        transcript_count=("transcript_count", "sum"),
        x_sum=("x_sum", "sum"),
        y_sum=("y_sum", "sum"),
        x_min=("x_min", "min"),
        x_max=("x_max", "max"),
        y_min=("y_min", "min"),
        y_max=("y_max", "max"),
    )
    obs_coord["x_centroid"] = obs_coord["x_sum"] / obs_coord["transcript_count"]
    obs_coord["y_centroid"] = obs_coord["y_sum"] / obs_coord["transcript_count"]
    obs_coord = obs_coord.drop(columns=["x_sum", "y_sum"])

    qc_df = pd.concat(qc_parts, ignore_index=True)
    agg_dict = {}
    if "n_nucleus_transcripts" in qc_df.columns:
        agg_dict["n_nucleus_transcripts"] = "sum"
    for text_col in ["sample", "old_sample_name"]:
        if text_col in qc_df.columns:
            agg_dict[text_col] = (
                lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else ""
            )
    if "fov_names" in qc_df.columns:
        agg_dict["fov_names"] = (
            lambda s: ";".join(
                sorted(set(";".join(s.dropna().astype(str)).split(";")) - {""})
            )
        )
    obs_qc = (
        qc_df.groupby("cell_id", observed=True).agg(agg_dict)
        if agg_dict
        else pd.DataFrame(index=obs_coord.index)
    )

    obs = obs_coord.join(obs_qc, how="left")
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs["slide_id"] = os.path.basename(os.path.normpath(base_dir))
    obs["source_format"] = "transcript_table_no_outs"
    obs["coordinate_method"] = coordinate_method

    cell_index = pd.Index(obs.index.astype(str), name="cell_id")
    gene_index = pd.Index(
        sorted(counts_df["feature_name"].astype(str).unique()), name="gene_name"
    )

    cell_codes = pd.Categorical(
        counts_df["cell_id"].astype(str), categories=cell_index
    ).codes
    gene_codes = pd.Categorical(
        counts_df["feature_name"].astype(str), categories=gene_index
    ).codes
    valid = (cell_codes >= 0) & (gene_codes >= 0)
    if not np.all(valid):
        raise RuntimeError("Some cell/gene codes were invalid while building the sparse matrix.")

    X = sparse.csr_matrix(
        (
            counts_df.loc[valid, "count"].to_numpy(dtype=np.float32),
            (cell_codes[valid], gene_codes[valid]),
        ),
        shape=(len(cell_index), len(gene_index)),
    )

    var = pd.DataFrame(index=gene_index)
    var["gene_name"] = var.index.astype(str)
    var["gene_id"] = var.index.astype(str)
    var["n_cells_by_counts"] = np.asarray((X > 0).sum(axis=0)).ravel().astype(int)
    var["total_counts"] = np.asarray(X.sum(axis=0)).ravel().astype(float)

    obs = obs.loc[cell_index].copy()
    obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel().astype(float)
    obs["n_genes_by_counts"] = np.asarray((X > 0).sum(axis=1)).ravel().astype(int)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(
        dtype=float
    )
    adata.uns["source_files"] = {
        "transcript_file": transcript_file,
        "node_meta_file": _find_first_file(base_dir, ["*node_meta.csv.gz", "*node_meta.csv"]),
        "embedding_file": _find_first_file(base_dir, ["*embeddings.npy.gz", "*embeddings.npy"]),
        "he_image_file": _find_first_file(
            base_dir, ["*registered_HE.tif", "*registered_HE.tif.gz"]
        ),
    }
    adata.uns["build_note"] = (
        "Built from transcript CSV because this slice does not have a standard Xenium outs folder. "
        "Expression counts are transcript counts aggregated by cell_id and feature_name."
    )

    if verbose:
        print("total transcript rows read:", f"{n_rows:,}")
        print("assigned transcript rows used:", f"{n_assigned_rows:,}")
        print(f"Built AnnData: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    return _merge_annotation(adata, anno_file=anno_file, verbose=verbose)


def _find_xenium_transcript_file(base_dir):
    candidates = []
    for pattern in ("*transcripts.csv.gz", "*transcripts.csv"):
        candidates.extend(_glob_files(base_dir, pattern))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    base_name = os.path.basename(os.path.normpath(base_dir))
    preferred = [path for path in candidates if os.path.basename(path).startswith(base_name)]
    if len(preferred) == 1:
        return preferred[0]

    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise ValueError(
        "Multiple transcript tables were found. Pass transcript_file explicitly:\n"
        f"{formatted}"
    )


def _find_first_file(base_dir, patterns):
    for pattern in patterns:
        matches = _glob_files(base_dir, pattern)
        if matches:
            return matches[0]
    return None


def _glob_files(base_dir, pattern):
    import glob

    return sorted(glob.glob(os.path.join(os.fspath(base_dir), pattern)))


def _update_transcript_progress(
    progress_reader,
    rows_read,
    assigned_rows,
    cells=None,
    cell_gene_pairs=None,
):
    if not hasattr(progress_reader, "set_postfix"):
        return

    postfix = {
        "rows": f"{rows_read:,}",
        "assigned": f"{assigned_rows:,}",
    }
    if cells is not None:
        postfix["cells"] = f"{cells:,}"
    if cell_gene_pairs is not None:
        postfix["cell_gene_pairs"] = f"{cell_gene_pairs:,}"

    progress_reader.set_postfix(postfix, refresh=False)


def _merge_annotation(adata, anno_file=None, verbose=True):
    if anno_file is None:
        if verbose:
            print(f"Finished. Total cells: {adata.n_obs}")
            print("No annotation file provided; skipping annotation merge.")
        return adata

    anno_file = os.fspath(anno_file)
    if not os.path.exists(anno_file):
        if verbose:
            print(f"Finished. Total cells: {adata.n_obs}")
            print(f"Annotation file not found: {anno_file}. Skipping annotation merge.")
        return adata

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
