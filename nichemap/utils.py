import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed


def generate_mean_grid_map(
    adata,
    score_id,
    x_col="x_centroid",
    y_col="y_centroid",
    bins=300,
    sigma_peak=2,
    verbose=True,
):
    """Generate grid-based mean map and smoothed map for peak detection."""

    x = adata.obs[x_col].values
    y = adata.obs[y_col].values
    z = adata.obs[score_id].values

    weighted_sum, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=z)
    counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])

    mean_map = weighted_sum / np.maximum(counts, 1)

    mean_map_filled = np.nan_to_num(mean_map, nan=0.0)
    smooth_map_peak = gaussian_filter(mean_map_filled, sigma=sigma_peak)

    if verbose:
        print(f"[Grid] {score_id}")
        print(f"Shape: {mean_map.shape}, Non-empty: {np.sum(counts > 0)}")

    return mean_map, smooth_map_peak, counts, xedges, yedges


def find_peaks(
    smooth_map_peak,
    mode="heuristic",
    intensity_sigma=3.0,
    use_otsu_base=False,
    target_niche_num=20,
    default_percentile=99.0,
    min_distance=20,
    verbose=True,
):
    """Detect local peaks using multiple threshold strategies."""

    peak_map = np.nan_to_num(smooth_map_peak, nan=0.0)
    valid_values = peak_map[peak_map > 0]

    if len(valid_values) == 0:
        return 0, 0, [], None

    min_distance = int(min_distance)

    if mode == "heuristic":
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values)
        threshold = mean_val + intensity_sigma * std_val

        if use_otsu_base:
            threshold = max(threshold, threshold_otsu(valid_values))

        coords = peak_local_max(
            peak_map, min_distance=min_distance, threshold_abs=threshold
        )

    elif mode == "target":
        percentiles = np.arange(90, 100, 0.1)
        results = []

        for p in percentiles:
            th = np.percentile(valid_values, p)
            coords_tmp = peak_local_max(
                peak_map, min_distance=min_distance, threshold_abs=th
            )
            results.append(
                {
                    "percentile": p,
                    "threshold": th,
                    "n_peaks": len(coords_tmp),
                    "diff": abs(len(coords_tmp) - target_niche_num),
                }
            )

        df = pd.DataFrame(results)
        best = df.sort_values(["diff", "percentile"], ascending=[True, False]).iloc[0]

        threshold = best["threshold"]
        coords = peak_local_max(
            peak_map, min_distance=min_distance, threshold_abs=threshold
        )

        return best["percentile"], threshold, coords, df

    elif mode == "percentile":
        threshold = np.percentile(valid_values, default_percentile)
        coords = peak_local_max(
            peak_map, min_distance=min_distance, threshold_abs=threshold
        )

    else:
        raise ValueError("Invalid mode")

    return None, threshold, coords, None


def create_expansion_mask(
    mean_map_filled,
    sigma=2,
    mode="heuristic",
    expansion_sigma=0.5,
    use_otsu_base=True,
    percentile=88,
    verbose=True,
):
    """Create mask for watershed expansion."""

    smooth_map = gaussian_filter(mean_map_filled, sigma=sigma)
    valid_values = smooth_map[smooth_map > 0]

    if len(valid_values) == 0:
        return smooth_map, np.zeros_like(smooth_map, dtype=bool)

    if mode == "heuristic":
        threshold = np.mean(valid_values) + expansion_sigma * np.std(valid_values)
        if use_otsu_base:
            threshold = max(threshold, threshold_otsu(valid_values))

    elif mode == "percentile":
        threshold = np.percentile(valid_values, percentile)

    else:
        raise ValueError("Invalid mode")

    mask = smooth_map > threshold
    return smooth_map, mask


def segment_niche_regions(smooth_map, markers, mask, verbose=True):
    """Run watershed segmentation."""

    elevation = sobel(np.nan_to_num(smooth_map, nan=0.0))

    labels = watershed(elevation, markers=markers, mask=mask)
    niche_ids = np.unique(labels)
    niche_ids = niche_ids[niche_ids != 0]

    if verbose:
        print(f"[Watershed] Niche count: {len(niche_ids)}")

    return labels, niche_ids


def map_niche_to_cells(
    adata,
    labels,
    xedges,
    yedges,
    x_col="x_centroid",
    y_col="y_centroid",
    output_col="ECM_niche_id",
    verbose=True,
):
    """Map grid labels to single-cell level."""

    x = adata.obs[x_col].values
    y = adata.obs[y_col].values

    x_bin = np.clip(np.digitize(x, xedges) - 1, 0, labels.shape[0] - 1)
    y_bin = np.clip(np.digitize(y, yedges) - 1, 0, labels.shape[1] - 1)

    adata.obs[output_col] = labels[x_bin, y_bin].astype(int)

    if verbose:
        print(f"[Assign] {output_col}")
        print(adata.obs[output_col].value_counts().head())

    return


def export_niche_results(
    adata,
    out_dir,
    niche_column="ECM_niche_id",
    file_prefix="niche",
    export_csv=True,
    export_h5ad=True,
    verbose=True,
):
    """Export results to CSV and/or H5AD."""

    os.makedirs(out_dir, exist_ok=True)

    if niche_column not in adata.obs:
        raise ValueError(f"{niche_column} not found")

    if export_csv:
        path = os.path.join(out_dir, f"{file_prefix}.csv")
        df = adata.obs.copy()
        df.index.name = "cell_id"
        df.to_csv(path)

    if export_h5ad:
        path = os.path.join(out_dir, f"{file_prefix}.h5ad")
        adata.write_h5ad(path)

    if verbose:
        assigned = (adata.obs[niche_column] > 0).sum()
        print(f"[Export] Assigned: {assigned}/{adata.n_obs}")

    return