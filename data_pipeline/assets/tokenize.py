from dagster import asset, AssetExecutionContext
from datasets import Dataset
from transformers import AutoTokenizer

from data_pipeline.resources import HuggingFaceConfig


@asset(group_name="data_processing")
def tokenized_dataset(
    context: AssetExecutionContext,
    hf_config: HuggingFaceConfig,
    cleaned_dataset: Dataset,
) -> Dataset:
    """Tokenize the cleaned dataset using the configured model's tokenizer."""
    context.log.info(f"Loading tokenizer for: {hf_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_col = hf_config.text_column
    max_len = hf_config.max_seq_length

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    context.log.info(f"Tokenizing {len(cleaned_dataset)} examples (max_len={max_len})")
    ds = cleaned_dataset.map(tokenize_fn, batched=True, remove_columns=cleaned_dataset.column_names)

    context.log.info(f"Tokenized dataset columns: {ds.column_names}")
    return ds
