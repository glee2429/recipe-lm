from dagster import build_asset_context
from datasets import Dataset

from data_pipeline.assets.clean import cleaned_dataset
from data_pipeline.assets.split import train_val_splits
from data_pipeline.assets.train import trained_model
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


def test_cleaned_dataset_removes_empty_and_duplicates():
    raw = Dataset.from_dict(
        {"text": ["hello", "  ", "hello", "world", "", "world", "foo"]}
    )
    config = _make_config()
    context = build_asset_context()
    result = cleaned_dataset(context, config, raw)

    texts = result["text"]
    assert "" not in texts
    assert len(texts) == len(set(texts))
    assert set(texts) == {"hello", "world", "foo"}


def test_cleaned_dataset_removes_trailing_commentary():
    raw = Dataset.from_dict(
        {
            "text": [
                "Recipe for cookies:\n- 1 cup flour\nDirections:\n- Mix.\nSubmitted by: Jane",
                "Recipe for soup:\n- 1 onion\nDirections:\n- Boil.\nSource: allrecipes.com",
                "Recipe for cake:\n- 2 eggs\nDirections:\n- Bake.\nNotes: Use room temp eggs.",
                "Recipe for bread:\n- 3 cups flour\nDirections:\n- Knead.\n- I love this recipe!",
                "Recipe for pie:\n- 2 cups flour\nDirections:\n- Roll out.\n- The recipe was given to me by my friend.",
                "Recipe for stew:\n- 1 lb beef\nDirections:\n- Simmer.\n- Thanks, Nancy!\n- Another favorite recipe.",
                "Recipe for salad:\n- 1 head lettuce\nDirections:\n- Toss.\n- For more delicious recipes, check out our Pinterest boards.",
                "Recipe for tacos:\n- 1 lb ground beef\nDirections:\n- Brown meat.\n- If you love this recipe, please give it five stars!",
                "Recipe for pasta:\n- 1 lb spaghetti\nDirections:\n- Boil.\n- Photo by Sara's Kitchen",
                "Recipe for rice:\n- 2 cups rice\nDirections:\n- Cook.\nThis post contains affiliate links.",
            ]
        }
    )
    config = _make_config()
    context = build_asset_context()
    result = cleaned_dataset(context, config, raw)

    for text in result["text"]:
        assert "Submitted by" not in text
        assert "Source:" not in text
        assert "Notes:" not in text
        assert "I love this" not in text
        assert "The recipe was given" not in text
        assert "Thanks" not in text
        assert "Another favorite" not in text
        assert "For more" not in text
        assert "Pinterest" not in text
        assert "five stars" not in text
        assert "Photo by" not in text
        assert "affiliate" not in text
        # The actual recipe content should be preserved
        assert "Directions:" in text


def test_cleaned_dataset_skips_when_column_missing():
    raw = Dataset.from_dict({"other_col": ["a", "b"]})
    config = _make_config(text_column="text")
    context = build_asset_context()
    result = cleaned_dataset(context, config, raw)
    assert len(result) == 2


def test_train_val_splits_creates_files(tmp_path):
    ds = Dataset.from_dict(
        {
            "input_ids": list(range(100)),
            "attention_mask": [1] * 100,
        }
    )
    config = _make_config(output_dir=str(tmp_path), val_split_ratio=0.2)
    context = build_asset_context()
    result = train_val_splits(context, config, ds)

    assert (tmp_path / "train").exists()
    assert (tmp_path / "val").exists()
    assert result.metadata["train_examples"].value == 80
    assert result.metadata["val_examples"].value == 20


def test_trained_model_asset_has_correct_deps():
    asset_def = trained_model
    dep_keys = {str(k) for k in asset_def.asset_deps.values() for k in k}
    assert any("train_val_splits" in k for k in dep_keys)

    for spec in asset_def.specs:
        assert spec.group_name == "training"
