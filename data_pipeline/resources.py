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

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
