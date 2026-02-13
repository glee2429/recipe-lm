# Data Pipeline: HuggingFace LLM Fine-Tuning

A [Dagster](https://dagster.io/) pipeline that downloads datasets from HuggingFace, processes them, and fine-tunes a language model using LoRA/QLoRA.

## Pipeline

```
raw_dataset → cleaned_dataset → tokenized_dataset → train_val_splits → trained_model
```

| Asset | Description |
|---|---|
| **raw_dataset** | Downloads a dataset from HuggingFace Hub |
| **cleaned_dataset** | Strips whitespace, removes empty rows, deduplicates |
| **tokenized_dataset** | Tokenizes text using the model's `AutoTokenizer` with padding/truncation |
| **train_val_splits** | Splits into train/validation sets, saves as Arrow files |
| **trained_model** | Fine-tunes the model with QLoRA (4-bit) using HF `Trainer`, evaluates on val set, saves the LoRA adapter |

## Setup

```bash
pip install -e ".[dev]"
huggingface-cli login  # required for gated models (Gemma, Llama, etc.)
```

## Usage

Launch the Dagster UI and materialize all assets:

```bash
dagster dev
```

Or run programmatically:

```python
from dagster import materialize
from data_pipeline.assets import *
from data_pipeline.resources import HuggingFaceConfig
from data_pipeline.io_managers import hf_dataset_io_manager

materialize(
    [raw_dataset, cleaned_dataset, tokenized_dataset, train_val_splits, trained_model],
    resources={
        "hf_config": HuggingFaceConfig(
            dataset_name="tatsu-lab/alpaca",
            model_name="google/gemma-2b",
        ),
        "io_manager": hf_dataset_io_manager,
    },
)
```

## Configuration

All parameters are configurable via the `hf_config` resource (see `configs/default.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `dataset_name` | `tatsu-lab/alpaca` | HuggingFace dataset identifier |
| `dataset_subset` | `None` | Dataset subset/config name |
| `model_name` | `google/gemma-2b` | HuggingFace model identifier |
| `text_column` | `text` | Column to tokenize |
| `max_seq_length` | `512` | Max token length |
| `val_split_ratio` | `0.1` | Fraction of data for validation |
| `num_train_epochs` | `3` | Training epochs |
| `per_device_train_batch_size` | `4` | Batch size per device |
| `learning_rate` | `2e-4` | Learning rate |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |

## Outputs

After a full run:

- `./processed_data/train/` — tokenized training split (Arrow format)
- `./processed_data/val/` — tokenized validation split (Arrow format)
- `./processed_data/lora_adapter/` — trained LoRA adapter weights

Load the adapter for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
model = PeftModel.from_pretrained(base_model, "./processed_data/lora_adapter")
```
