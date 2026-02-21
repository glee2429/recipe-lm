import os

from dagster import build_asset_context
from datasets import Dataset

from data_pipeline.checks import (
    cleaned_dataset_is_nonempty,
    cleaned_dataset_no_empty_texts,
    raw_dataset_has_text_column,
    raw_dataset_is_nonempty,
    tokenized_dataset_has_expected_columns,
    tokenized_dataset_is_nonempty,
    tokenized_dataset_sequences_valid,
    train_val_splits_exist_on_disk,
)
from data_pipeline.resources import HuggingFaceConfig


def _make_config(**overrides) -> HuggingFaceConfig:
    defaults = dict(
        dataset_name="test",
        model_name="test",
        text_column="text",
        max_seq_length=32,
        val_split_ratio=0.2,
        seed=42,
        output_dir="/tmp/test_processed_data",
    )
    defaults.update(overrides)
    return HuggingFaceConfig(**defaults)


# ── raw_dataset checks ──────────────────────────────────────────────


def test_raw_dataset_is_nonempty_pass():
    ds = Dataset.from_dict({"text": ["a", "b"]})
    result = raw_dataset_is_nonempty(raw_dataset=ds)
    assert result.passed


def test_raw_dataset_is_nonempty_fail():
    ds = Dataset.from_dict({"text": []})
    result = raw_dataset_is_nonempty(raw_dataset=ds)
    assert not result.passed


def test_raw_dataset_has_text_column_pass():
    ds = Dataset.from_dict({"text": ["a"]})
    config = _make_config(text_column="text")
    result = raw_dataset_has_text_column(raw_dataset=ds, hf_config=config)
    assert result.passed


def test_raw_dataset_has_text_column_fail():
    ds = Dataset.from_dict({"other": ["a"]})
    config = _make_config(text_column="text")
    result = raw_dataset_has_text_column(raw_dataset=ds, hf_config=config)
    assert not result.passed


# ── cleaned_dataset checks ──────────────────────────────────────────


def test_cleaned_dataset_is_nonempty_pass():
    ds = Dataset.from_dict({"text": ["hello"]})
    result = cleaned_dataset_is_nonempty(cleaned_dataset=ds)
    assert result.passed


def test_cleaned_dataset_is_nonempty_fail():
    ds = Dataset.from_dict({"text": []})
    result = cleaned_dataset_is_nonempty(cleaned_dataset=ds)
    assert not result.passed


def test_cleaned_dataset_no_empty_texts_pass():
    ds = Dataset.from_dict({"text": ["hello", "world"]})
    config = _make_config(text_column="text")
    result = cleaned_dataset_no_empty_texts(cleaned_dataset=ds, hf_config=config)
    assert result.passed


def test_cleaned_dataset_no_empty_texts_fail():
    ds = Dataset.from_dict({"text": ["hello", "", "world"]})
    config = _make_config(text_column="text")
    result = cleaned_dataset_no_empty_texts(cleaned_dataset=ds, hf_config=config)
    assert not result.passed


def test_cleaned_dataset_no_empty_texts_missing_column():
    ds = Dataset.from_dict({"other": ["hello"]})
    config = _make_config(text_column="text")
    result = cleaned_dataset_no_empty_texts(cleaned_dataset=ds, hf_config=config)
    assert not result.passed


# ── tokenized_dataset checks ────────────────────────────────────────


def test_tokenized_dataset_has_expected_columns_pass():
    ds = Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})
    result = tokenized_dataset_has_expected_columns(tokenized_dataset=ds)
    assert result.passed


def test_tokenized_dataset_has_expected_columns_fail():
    ds = Dataset.from_dict({"input_ids": [[1, 2]]})
    result = tokenized_dataset_has_expected_columns(tokenized_dataset=ds)
    assert not result.passed


def test_tokenized_dataset_is_nonempty_pass():
    ds = Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})
    result = tokenized_dataset_is_nonempty(tokenized_dataset=ds)
    assert result.passed


def test_tokenized_dataset_is_nonempty_fail():
    ds = Dataset.from_dict({"input_ids": [], "attention_mask": []})
    result = tokenized_dataset_is_nonempty(tokenized_dataset=ds)
    assert not result.passed


def test_tokenized_dataset_sequences_valid_pass():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5, 6]]})
    result = tokenized_dataset_sequences_valid(tokenized_dataset=ds)
    assert result.passed


def test_tokenized_dataset_sequences_valid_fail_all_zeros():
    ds = Dataset.from_dict({"input_ids": [[0, 0, 0], [0, 0, 0]]})
    result = tokenized_dataset_sequences_valid(tokenized_dataset=ds)
    assert not result.passed


def test_tokenized_dataset_sequences_valid_partial_zeros_pass():
    ds = Dataset.from_dict({"input_ids": [[0, 0, 0], [1, 2, 3]]})
    result = tokenized_dataset_sequences_valid(tokenized_dataset=ds)
    assert result.passed


def test_tokenized_dataset_sequences_valid_empty_dataset():
    ds = Dataset.from_dict({"input_ids": []})
    result = tokenized_dataset_sequences_valid(tokenized_dataset=ds)
    assert not result.passed


# ── train_val_splits checks ─────────────────────────────────────────


def test_train_val_splits_exist_on_disk_pass(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    (train_dir / "data.arrow").write_text("fake")
    (val_dir / "data.arrow").write_text("fake")

    config = _make_config(output_dir=str(tmp_path))
    result = train_val_splits_exist_on_disk(hf_config=config)
    assert result.passed


def test_train_val_splits_exist_on_disk_fail_missing(tmp_path):
    config = _make_config(output_dir=str(tmp_path))
    result = train_val_splits_exist_on_disk(hf_config=config)
    assert not result.passed


def test_train_val_splits_exist_on_disk_fail_empty_dirs(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()

    config = _make_config(output_dir=str(tmp_path))
    result = train_val_splits_exist_on_disk(hf_config=config)
    assert not result.passed
