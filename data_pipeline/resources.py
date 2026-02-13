from dagster import ConfigurableResource
from typing import Optional


class HuggingFaceConfig(ConfigurableResource):
    """Configuration for HuggingFace dataset and model."""

    dataset_name: str = "tatsu-lab/alpaca"
    dataset_subset: Optional[str] = None
    model_name: str = "google/gemma-2b"
    text_column: str = "text"
    max_seq_length: int = 512
    val_split_ratio: float = 0.1
    seed: int = 42
    output_dir: str = "./processed_data"
