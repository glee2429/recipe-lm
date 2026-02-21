import os

from dagster import AssetCheckResult, AssetCheckSeverity, asset_check
from datasets import Dataset

from data_pipeline.resources import HuggingFaceConfig


# ── raw_dataset checks ──────────────────────────────────────────────


@asset_check(asset="raw_dataset", blocking=True)
def raw_dataset_is_nonempty(raw_dataset: Dataset) -> AssetCheckResult:
    """Fail if the downloaded dataset has zero rows."""
    n = len(raw_dataset)
    passed = n > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={"num_rows": n},
    )


@asset_check(asset="raw_dataset", blocking=True)
def raw_dataset_has_text_column(
    raw_dataset: Dataset, hf_config: HuggingFaceConfig
) -> AssetCheckResult:
    """Fail if the expected text column is missing from the schema."""
    col = hf_config.text_column
    has_col = col in raw_dataset.column_names
    return AssetCheckResult(
        passed=has_col,
        severity=AssetCheckSeverity.ERROR,
        metadata={"expected_column": col, "columns": raw_dataset.column_names},
    )


# ── cleaned_dataset checks ──────────────────────────────────────────


@asset_check(asset="cleaned_dataset", blocking=True)
def cleaned_dataset_is_nonempty(cleaned_dataset: Dataset) -> AssetCheckResult:
    """Fail if cleaning removed every row."""
    n = len(cleaned_dataset)
    passed = n > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={"num_rows": n},
    )


@asset_check(asset="cleaned_dataset", blocking=True)
def cleaned_dataset_no_empty_texts(
    cleaned_dataset: Dataset, hf_config: HuggingFaceConfig
) -> AssetCheckResult:
    """Fail if any row has an empty text value after cleaning."""
    col = hf_config.text_column
    if col not in cleaned_dataset.column_names:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": f"Column '{col}' not found"},
        )
    empty_count = sum(1 for t in cleaned_dataset[col] if not t.strip())
    passed = empty_count == 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={"empty_count": empty_count},
    )


# ── tokenized_dataset checks ────────────────────────────────────────


@asset_check(asset="tokenized_dataset", blocking=True)
def tokenized_dataset_has_expected_columns(
    tokenized_dataset: Dataset,
) -> AssetCheckResult:
    """Fail if input_ids or attention_mask columns are missing."""
    cols = tokenized_dataset.column_names
    missing = [c for c in ("input_ids", "attention_mask") if c not in cols]
    passed = len(missing) == 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={"missing_columns": missing, "columns": cols},
    )


@asset_check(asset="tokenized_dataset", blocking=True)
def tokenized_dataset_is_nonempty(tokenized_dataset: Dataset) -> AssetCheckResult:
    """Fail if the tokenized dataset has zero rows."""
    n = len(tokenized_dataset)
    passed = n > 0
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={"num_rows": n},
    )


@asset_check(asset="tokenized_dataset", blocking=True)
def tokenized_dataset_sequences_valid(
    tokenized_dataset: Dataset,
) -> AssetCheckResult:
    """Fail if every token sequence is all zeros (corrupted tokenization)."""
    if len(tokenized_dataset) == 0 or "input_ids" not in tokenized_dataset.column_names:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"reason": "Empty dataset or missing input_ids"},
        )
    all_zero_count = sum(
        1 for ids in tokenized_dataset["input_ids"] if not any(ids)
    )
    passed = all_zero_count < len(tokenized_dataset)
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "all_zero_sequences": all_zero_count,
            "total_sequences": len(tokenized_dataset),
        },
    )


# ── train_val_splits checks ─────────────────────────────────────────


@asset_check(asset="train_val_splits", blocking=True)
def train_val_splits_exist_on_disk(
    hf_config: HuggingFaceConfig,
) -> AssetCheckResult:
    """Fail if the train/ or val/ directories are missing or empty."""
    output_dir = hf_config.output_dir
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")

    train_exists = os.path.isdir(train_path) and len(os.listdir(train_path)) > 0
    val_exists = os.path.isdir(val_path) and len(os.listdir(val_path)) > 0

    passed = train_exists and val_exists
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={"train_exists": train_exists, "val_exists": val_exists},
    )
