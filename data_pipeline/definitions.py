from dagster import Definitions, load_assets_from_modules

from data_pipeline import assets
from data_pipeline.io_managers import hf_dataset_io_manager
from data_pipeline.resources import HuggingFaceConfig

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "hf_config": HuggingFaceConfig(),
        "io_manager": hf_dataset_io_manager,
    },
)
