# recipe-lm

A [Dagster](https://dagster.io/) pipeline that fine-tunes [Gemma-2B](https://huggingface.co/google/gemma-2b) on [recipe data](https://huggingface.co/datasets/corbt/all-recipes) to generate recipes from prompts, using LoRA for parameter-efficient fine-tuning.

The trained adapter is published at [ClaireLee2429/gemma-2b-recipes-lora](https://huggingface.co/ClaireLee2429/gemma-2b-recipes-lora).

## Pipeline

```
raw_dataset → cleaned_dataset → tokenized_dataset → train_val_splits → trained_model
```

| Asset | Description |
|---|---|
| **raw_dataset** | Downloads a dataset from HuggingFace Hub, optionally samples down to `max_samples` |
| **cleaned_dataset** | Strips whitespace, removes empty rows, deduplicates |
| **tokenized_dataset** | Tokenizes text using the model's `AutoTokenizer` with padding/truncation |
| **train_val_splits** | Splits into train/validation sets, saves as Arrow files |
| **trained_model** | Fine-tunes the model with LoRA using HF `Trainer`, evaluates on val set, saves the LoRA adapter |

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
            dataset_name="corbt/all-recipes",
            model_name="google/gemma-2b",
            text_column="input",
            max_samples=5000,
        ),
        "io_manager": hf_dataset_io_manager,
    },
)
```

## Configuration

All parameters are configurable via the `hf_config` resource (see `configs/default.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `dataset_name` | `corbt/all-recipes` | HuggingFace dataset identifier |
| `dataset_subset` | `None` | Dataset subset/config name |
| `model_name` | `google/gemma-2b` | HuggingFace model identifier |
| `text_column` | `input` | Column to tokenize |
| `max_samples` | `5000` | Max examples to sample (set to `None` for full dataset) |
| `max_seq_length` | `512` | Max token length |
| `val_split_ratio` | `0.1` | Fraction of data for validation |
| `num_train_epochs` | `3` | Training epochs |
| `per_device_train_batch_size` | `4` | Batch size per device |
| `learning_rate` | `2e-4` | Learning rate |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |

## Device Support

The training step automatically detects the available hardware:

| Device | Behavior |
|---|---|
| **CUDA** | QLoRA with 4-bit quantization (bitsandbytes), bf16 training |
| **MPS (Apple Silicon)** | LoRA without quantization, fp32 training |
| **CPU** | LoRA without quantization, fp32 training |

## Outputs

After a full run:

- `./processed_data/train/` — tokenized training split (Arrow format)
- `./processed_data/val/` — tokenized validation split (Arrow format)
- `./processed_data/lora_adapter/` — trained LoRA adapter weights

## Inference

A standalone inference script with built-in post-processing is provided. It removes common generation artifacts (trailing comments, empty bullets, malformed lines, truncated text).

### Quick start

```bash
python inference.py --prompt "Recipe for chocolate chip cookies:"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `"Recipe for chocolate chip cookies:"` | Prompt for recipe generation |
| `--adapter` | `./processed_data/lora_adapter` | Path to LoRA adapter (local or HuggingFace Hub ID) |
| `--model` | `google/gemma-2b` | Base model name |
| `--max-tokens` | `256` | Maximum new tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--raw` | off | Show raw output without post-processing |
| `--save` | none | Save output to file |

### Examples

```bash
# Generate with post-processing (default)
python inference.py --prompt "Recipe for pasta carbonara:"

# Compare raw vs cleaned output
python inference.py --prompt "Recipe for tomato soup:" --raw

# Save to file
python inference.py --prompt "Recipe for banana bread:" --save output.txt

# Use adapter from HuggingFace Hub
python inference.py --adapter ClaireLee2429/gemma-2b-recipes-lora --prompt "Recipe for chicken stir fry:"
```

### Using the model directly in Python

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# From local adapter
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
model = PeftModel.from_pretrained(base_model, "./processed_data/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("./processed_data/lora_adapter")

# Or from HuggingFace Hub
# model = PeftModel.from_pretrained(base_model, "ClaireLee2429/gemma-2b-recipes-lora")
# tokenizer = AutoTokenizer.from_pretrained("ClaireLee2429/gemma-2b-recipes-lora")
```

### Sample output

> **Prompt:** `Recipe for chocolate chip cookies:`

```
Recipe for chocolate chip cookies:
Ingredients:
- 1/2 cup butter
- 1/4 cup sugar
- 1/4 cup packed brown sugar
- 3/4 cup flour
- 1/2 teaspoon baking soda
- 1/2 teaspoon salt
- 1 egg
- 1 teaspoon vanilla
- 1/2 cup chocolate chips

Directions:
- In a medium bowl, cream together the butter and sugars.
- Add in the egg and vanilla, mixing until combined.
- In another bowl, whisk together the flour, baking soda and salt.
- Add to the creamed mixture alternately with the chocolate chips, ending with the dry ingredients.
- Stir in gently using a rubber spatula.
- Drop by rounded teaspoonfuls onto ungreased cookie sheets.
- Bake at 350 degrees F for 9 minutes.
- Cool on wire racks before serving.
```

> **Prompt:** `Recipe for pasta carbonara:`

```
Recipe for pasta carbonara:
Ingredients:
- 100 g spaghetti
- 100 g smoked bacon or pancetta
- 250 g mushrooms
- 2 eggs
- 1 tablespoon olive oil
- 1 tablespoon white wine
- 1 teaspoon freshly grated parmesan cheese

Directions:
- Cook the pasta according to package instructions.
- Meanwhile, brown the bacon in a frying pan with some olive oil.
- Add the mushrooms and cook them until they are tender (about 10 minutes).
- Add the cooked pasta to the mushroom mixture along with the eggs and stir well.
- Stir in the wine and then sprinkle over the grated parmesan cheese.
- Season with salt and pepper and serve immediately.
```

> **Prompt:** `Recipe for tomato soup:`

```
Recipe for tomato soup:
Ingredients:
- 1 (28 ounce) can crushed tomatoes
- 1 (4 ounce) can tomato paste
- 1 tablespoon dried basil
- 1/2 tablespoon salt
- 1/2 tablespoon sugar
- 1/2 teaspoon garlic powder
- 1/4 teaspoon red pepper flakes

Directions:
- Combine all ingredients in a large pot and bring to a boil.
- Reduce heat, cover and simmer until thickened, about 1 hour.
- Serve over grilled cheese sandwiches or with tortilla chips.
```

### Comparing base vs. fine-tuned model

To see the effect of fine-tuning, compare outputs from the base Gemma-2B model (without the adapter) against the fine-tuned version:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "Recipe for banana bread:\n"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
inputs = tokenizer(prompt, return_tensors="pt")

# Base model (no fine-tuning)
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
base_model.eval()
with torch.no_grad():
    base_out = base_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print("=== BASE MODEL ===")
print(tokenizer.decode(base_out[0], skip_special_tokens=True))

# Fine-tuned model
ft_model = PeftModel.from_pretrained(base_model, "./processed_data/lora_adapter")
ft_model.eval()
with torch.no_grad():
    ft_out = ft_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print("\n=== FINE-TUNED MODEL ===")
print(tokenizer.decode(ft_out[0], skip_special_tokens=True))
```

## Training Results

Trained on 4,500 recipes for 1 epoch on Apple Silicon (MPS):

| Metric | Value |
|---|---|
| Train loss | 1.3928 |
| Val loss | 1.4030 |
| Val perplexity | 4.07 |
| Training time | ~1 hr 21 min |

## Tests

```bash
pytest tests/
```
