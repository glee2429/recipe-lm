from dagster import asset, AssetExecutionContext
from datasets import Dataset

from data_pipeline.resources import HuggingFaceConfig


@asset(group_name="data_processing")
def cleaned_dataset(
    context: AssetExecutionContext, hf_config: HuggingFaceConfig, raw_dataset: Dataset
) -> Dataset:
    """Clean and filter the raw dataset."""
    text_col = hf_config.text_column
    initial_count = len(raw_dataset)

    if text_col not in raw_dataset.column_names:
        available = raw_dataset.column_names
        context.log.warning(
            f"Column '{text_col}' not found. Available: {available}. "
            f"Skipping text-based cleaning."
        )
        return raw_dataset

    # Strip whitespace
    ds = raw_dataset.map(lambda x: {text_col: x[text_col].strip()})

    # Remove empty rows
    ds = ds.filter(lambda x: len(x[text_col]) > 0)

    # Deduplicate
    seen = set()

    def dedup(example):
        text = example[text_col]
        if text in seen:
            return False
        seen.add(text)
        return True

    ds = ds.filter(dedup)

    final_count = len(ds)
    removed = initial_count - final_count
    context.log.info(
        f"Cleaning complete: {initial_count} -> {final_count} "
        f"({removed} removed, {removed / initial_count * 100:.1f}%)"
    )
    return ds
