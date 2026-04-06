import os

import scanpy as sc
from tqdm.auto import tqdm

from . import plot as plots
from . import preprocess
from . import utils


class NicheMap:
    """Pipeline manager for spatial niche discovery."""

    def __init__(
        self,
        adata,
        score_id,
        out_dir,
        sample_prefix="Sample",
        x_col="x_centroid",
        y_col="y_centroid",
        show_progress=True,
        verbose=True,
    ):
        self.adata = adata
        self.score_id = score_id
        self.out_dir = out_dir
        self.sample_prefix = sample_prefix
        self.x_col = x_col
        self.y_col = y_col
        self.niche_col = f"{score_id}_niche_id"
        self.show_progress = show_progress
        self.verbose = verbose

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        self.xedges = None
        self.yedges = None
        self.mean_map = None
        self.smooth_map_peak = None
        self.counts = None
        self.peak_coords = None
        self.peak_x = None
        self.peak_y = None
        self.markers = None
        self.niche_labels = None
        self.smooth_map_exp = None

    def _log(self, message):
        """Print a formatted message when verbose mode is enabled."""

        if self.verbose:
            print(f"[{self.score_id}] {message}")

    @staticmethod
    def _build_markers(shape, peak_coords):
        """Build watershed marker matrix from peak coordinates."""

        markers = utils.np.zeros(shape, dtype=int)
        for i, (row, col) in enumerate(peak_coords, start=1):
            markers[row, col] = i
        return markers

    def calculate_score(self, gene_list_csv, normalize=True, plot=True):
        """Normalize data and calculate the gene signature score."""

        self._log("Calculating gene signature score")

        if normalize:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)

        preprocess.calculate_gene_signature_score(
            adata=self.adata,
            csv_path=gene_list_csv,
            score_id=self.score_id,
            gene_column="Gene Symbol",
            verbose=self.verbose,
        )

        if plot:
            plots.plot_spatial_score(
                self.adata,
                score_name=self.score_id,
                out_dir=self.out_dir,
            )

        return self

    def build_grid(self, bins=300, sigma_peak=2, plot=True):
        """Build spatial grids and smoothed maps."""

        self._log(f"Building spatial grids (bins={bins}, sigma_peak={sigma_peak})")

        (
            self.mean_map,
            self.smooth_map_peak,
            self.counts,
            self.xedges,
            self.yedges,
        ) = utils.generate_mean_grid_map(
            self.adata,
            score_id=self.score_id,
            bins=bins,
            sigma_peak=sigma_peak,
            verbose=self.verbose,
        )

        if plot:
            plots.plot_grid_map(
                self.mean_map,
                self.xedges,
                self.yedges,
                cmap="inferno",
                title=f"Figure 1A: Raw {self.score_id} grid map",
                out_dir=self.out_dir,
            )
            plots.plot_grid_map(
                self.counts,
                self.xedges,
                self.yedges,
                cmap="viridis",
                title="Figure 1B: Grid cell density",
                out_dir=self.out_dir,
            )

        return self

    def find_seeds(self, intensity_sigma=1.5, plot=True):
        """Detect niche seed points from the smoothed peak map."""

        self._log(f"Finding niche seeds (intensity_sigma={intensity_sigma})")

        _, _, self.peak_coords, _ = utils.find_peaks(
            self.smooth_map_peak,
            mode="heuristic",
            intensity_sigma=intensity_sigma,
            use_otsu_base=True,
            verbose=self.verbose,
        )

        self.markers = self._build_markers(self.smooth_map_peak.shape, self.peak_coords)

        x_centers = (self.xedges[:-1] + self.xedges[1:]) / 2
        y_centers = (self.yedges[:-1] + self.yedges[1:]) / 2
        self.peak_x = [x_centers[row] for row, col in self.peak_coords]
        self.peak_y = [y_centers[col] for row, col in self.peak_coords]

        if plot:
            plots.visualize_and_export_peaks(
                self.smooth_map_peak,
                self.peak_coords,
                self.xedges,
                self.yedges,
                out_dir=self.out_dir,
            )
            plots.plot_peak_positions_on_scatter(
                x_col=self.adata.obs[self.x_col].values,
                y_col=self.adata.obs[self.y_col].values,
                peak_coords=self.peak_coords,
                xedges=self.xedges,
                yedges=self.yedges,
                out_dir=self.out_dir,
            )

        return self

    def segment_niches(self, expansion_sigma=1.0, plot=True):
        """Create expansion mask and run watershed segmentation."""

        self._log(f"Segmenting niches (expansion_sigma={expansion_sigma})")

        self.smooth_map_exp, niche_mask = utils.create_expansion_mask(
            self.mean_map,
            sigma=2,
            mode="heuristic",
            expansion_sigma=expansion_sigma,
            use_otsu_base=True,
            verbose=self.verbose,
        )

        if plot:
            plots.plot_expansion_mask(
                self.smooth_map_exp,
                niche_mask,
                self.peak_x,
                self.peak_y,
                self.xedges,
                self.yedges,
                out_dir=self.out_dir,
            )

        self.niche_labels, _ = utils.segment_niche_regions(
            self.smooth_map_exp,
            self.markers,
            niche_mask,
            verbose=self.verbose,
        )

        if plot:
            plots.plot_niche_map(
                self.smooth_map_exp,
                self.niche_labels,
                self.peak_x,
                self.peak_y,
                self.xedges,
                self.yedges,
                cmap_base="magma",
                cmap_labels="Set3",
                out_dir=self.out_dir,
            )

        return self

    def map_and_export(self, plot=True):
        """Map niche labels back to cells and export results."""

        self._log("Mapping niche labels to cells and exporting results")

        utils.map_niche_to_cells(
            self.adata,
            self.niche_labels,
            self.xedges,
            self.yedges,
            x_col=self.x_col,
            y_col=self.y_col,
            output_col=self.niche_col,
            verbose=self.verbose,
        )

        if plot:
            plots.plot_cell_level_niches(
                self.adata,
                self.niche_labels,
                self.xedges,
                self.yedges,
                niche_column=self.niche_col,
                coords_columns=(self.x_col, self.y_col),
                cmap="tab20",
                boundary_color="cyan",
                out_dir=self.out_dir,
                verbose=self.verbose,
            )

        utils.export_niche_results(
            self.adata,
            out_dir=self.out_dir,
            niche_column=self.niche_col,
            file_prefix=self.sample_prefix,
            export_csv=True,
            export_h5ad=True,
            verbose=self.verbose,
        )

        return self.adata

    def run(
        self,
        gene_list_csv,
        bins=300,
        peak_intensity=1.5,
        exp_intensity=1.0,
        normalize=True,
        plot=True,
    ):
        """Run the full NicheMap pipeline."""

        steps = [
            ("Calculate score", lambda: self.calculate_score(
                gene_list_csv=gene_list_csv,
                normalize=normalize,
                plot=plot,
            )),
            ("Build grid", lambda: self.build_grid(
                bins=bins,
                sigma_peak=2,
                plot=plot,
            )),
            ("Find seeds", lambda: self.find_seeds(
                intensity_sigma=peak_intensity,
                plot=plot,
            )),
            ("Segment niches", lambda: self.segment_niches(
                expansion_sigma=exp_intensity,
                plot=plot,
            )),
            ("Map and export", lambda: self.map_and_export(
                plot=plot,
            )),
        ]

        if self.verbose:
            print(f"========== Starting NicheMap Pipeline: {self.sample_prefix} ==========")

        iterator = steps
        if self.show_progress:
            iterator = tqdm(steps, desc="NicheMap pipeline", unit="step")

        result = None
        for step_name, step_func in iterator:
            if self.show_progress and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(step_name)
            elif self.verbose:
                self._log(step_name)

            result = step_func()

        if self.verbose:
            print(f"========== Finished NicheMap Pipeline: {self.sample_prefix} ==========")

        return result