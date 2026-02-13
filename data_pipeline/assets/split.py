import os
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from datasets import Dataset

from data_pipeline.resources import HuggingFaceConfig


@asset(group_name="data_processing")
def train_val_splits(
    context: AssetExecutionContext,
    hf_config: HuggingFaceConfig,
    tokenized_dataset: Dataset,
) -> MaterializeResult:
    """Split the tokenized dataset into train/validation sets and save to disk."""
    split = tokenized_dataset.train_test_split(
        test_size=hf_config.val_split_ratio,
        seed=hf_config.seed,
    )

    train_ds = split["train"]
    val_ds = split["test"]

    output_dir = hf_config.output_dir
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")

    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)

    context.log.info(
        f"Saved train ({len(train_ds)} examples) to {train_path}"
    )
    context.log.info(
        f"Saved val ({len(val_ds)} examples) to {val_path}"
    )

    return MaterializeResult(
        metadata={
            "train_examples": MetadataValue.int(len(train_ds)),
            "val_examples": MetadataValue.int(len(val_ds)),
            "output_dir": MetadataValue.path(output_dir),
        }
    )
