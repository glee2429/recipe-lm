from dagster import asset, AssetExecutionContext
from datasets import load_dataset, Dataset

from data_pipeline.resources import HuggingFaceConfig


@asset(group_name="data_processing")
def raw_dataset(
    context: AssetExecutionContext, hf_config: HuggingFaceConfig
) -> Dataset:
    """Download a dataset from HuggingFace Hub."""
    context.log.info(
        f"Downloading dataset: {hf_config.dataset_name} "
        f"(subset={hf_config.dataset_subset})"
    )

    ds = load_dataset(
        hf_config.dataset_name,
        name=hf_config.dataset_subset,
        split="train",
    )

    context.log.info(f"Downloaded {len(ds)} examples")
    context.log.info(f"Columns: {ds.column_names}")

    if hf_config.max_samples and len(ds) > hf_config.max_samples:
        ds = ds.shuffle(seed=hf_config.seed).select(range(hf_config.max_samples))
        context.log.info(f"Sampled down to {len(ds)} examples")

    return ds
