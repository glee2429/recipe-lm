from dagster import Definitions, load_assets_from_modules, load_asset_checks_from_modules

from data_pipeline import assets, checks
from data_pipeline.io_managers import hf_dataset_io_manager
from data_pipeline.resources import HuggingFaceConfig

all_assets = load_assets_from_modules([assets])
all_checks = load_asset_checks_from_modules([checks])

defs = Definitions(
    assets=all_assets,
    asset_checks=all_checks,
    resources={
        "hf_config": HuggingFaceConfig(),
        "io_manager": hf_dataset_io_manager,
    },
)
