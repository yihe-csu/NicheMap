import nichemap as nm
import os

base_dir = r"F:\spatial_data_lung\SSc_1_1_2_raw"
anno_file = r"F:\spatial_data_lung\ssc112_annotation_map.csv"
gene_list = r"F:\spatial_data_lung\marker_genes\ECM-gene.csv"
score_id = 'ECM_score'
peak_intensity=1.5
exp_intensity=1.0
out_dir = rf"F:\spatial_data_lung\Xenium_Result_data\SSc_1_1_2_result\{score_id}"
os.makedirs(out_dir, exist_ok=True)

adata = nm.preprocess.load_xenium_data(base_dir=base_dir, anno_file=anno_file)


model = nm.NicheMap(
    adata=adata,
    score_id=score_id,
    sample_prefix="SSc_1_1_2",
    out_dir=out_dir
)

final_adata = model.run(
    gene_list_csv=gene_list,
    bins=300,
    peak_intensity=peak_intensity,
    exp_intensity=exp_intensity
)