from setuptools import find_packages, setup


setup(
    name="nichemap",
    version="0.1.0",
    keywords=[
        "spatial transcriptomics",
        "xenium",
        "spatial niche",
        "watershed segmentation",
        "gene signature scoring",
    ],
    description=(
        "A spatial grid-based pipeline for niche identification in Xenium "
        "and spatial transcriptomics data"
    ),
    license="MIT License",
    url="https://github.com/yihe-csu/NicheMap",
    author="Yi He",
    author_email="yihe_csu@csu.edu.cn",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "scanpy>=1.9",
        "anndata>=0.9",
        "scipy>=1.10",
        "scikit-image>=0.20",
        "zarr>=2.14",
        "tqdm>=4.65",
    ],
    extras_require={
        "plot": ["matplotlib-scalebar"],
    },
)