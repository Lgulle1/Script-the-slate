import importlib

import config

DEPENDENCIES = [
    "nflreadpy",
    "polars",
    "duckdb",
    "numpy",
    "sklearn",
    "lightgbm",
    "shap",
    "streamlit",
    "requests",
    "pytest",
]


def test_dependencies_import_and_data_dirs_exist():
    for name in DEPENDENCIES:
        importlib.import_module(name)

    config.ensure_data_dirs()  # data/ is git-ignored, so a fresh clone lacks it
    for d in (config.RAW_DIR, config.SNAPSHOT_DIR, config.PROCESSED_DIR):
        assert d.is_dir(), d
