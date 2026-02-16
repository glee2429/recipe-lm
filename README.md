# recipe-lm

An end-to-end recipe generation system: a [Dagster](https://dagster.io/) training pipeline that fine-tunes [Gemma-2B](https://huggingface.co/google/gemma-2b) on [recipe data](https://huggingface.co/datasets/corbt/all-recipes) with LoRA, a FastAPI streaming server, and a React web UI ([kitchen-genie](https://github.com/glee2429/kitchen-genie)).

The trained adapter is published at [ClaireLee2429/gemma-2b-recipes-lora](https://huggingface.co/ClaireLee2429/gemma-2b-recipes-lora).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
│                                                              │
│  HuggingFace Hub ──► Dagster Pipeline ──► LoRA Adapter       │
│  (corbt/all-recipes)  (clean, tokenize,   (pushed to HF Hub) │
│                        split, train)                         │
│                                                              │
│  Alternative: Google Colab + Unsloth (2x faster on T4 GPU)  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Inference & Serving                     │
│                                                              │
│  inference.py ── CLI script for batch generation             │
│  server.py ───── FastAPI + SSE streaming (POST /generate)    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                        Web Frontend                          │
│                                                              │
│  kitchen-genie ── React + Vite + Tailwind + shadcn/ui        │
│                   Streams tokens from server.py in real time  │
│                   github.com/glee2429/kitchen-genie          │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
recipe-lm/
├── data_pipeline/          # Dagster pipeline
│   ├── assets/
│   │   ├── download.py     # Fetch dataset from HuggingFace Hub
│   │   ├── clean.py        # Strip whitespace, deduplicate
│   │   ├── tokenize.py     # Tokenize with AutoTokenizer
│   │   ├── split.py        # Train/val split
│   │   └── train.py        # LoRA fine-tuning with HF Trainer
│   ├── resources.py        # HuggingFaceConfig resource
│   ├── io_managers.py      # Arrow dataset I/O manager
│   └── definitions.py      # Dagster definitions
├── notebooks/
│   └── train_unsloth.ipynb # Colab notebook (Unsloth + T4 GPU)
├── inference.py            # CLI inference with post-processing
├── server.py               # FastAPI streaming API server
├── configs/
│   └── default.yaml        # Default pipeline configuration
├── tests/
│   └── test_assets.py      # Pipeline asset tests
└── pyproject.toml          # Dependencies (dev, colab, serve)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[serve]"
huggingface-cli login

# 2. Start the API server (uses pre-trained adapter from HuggingFace Hub)
python server.py --adapter ClaireLee2429/gemma-2b-recipes-lora

# 3. Start the frontend (in a separate terminal)
cd ../kitchen-genie && npm install && npm run dev

# 4. Open http://localhost:8080 and generate recipes!
```

## Training Pipeline

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
| **MPS (Apple Silicon)** | LoRA without quantization, fp16 inference / fp32 training |
| **CPU** | LoRA without quantization, fp32 training / inference |

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

### Why fine-tune?

Gemma-2B is a general-purpose model — it knows about recipes but doesn't reliably produce clean, structured output. Fine-tuning on 5,000 recipes teaches it a consistent format and keeps it focused. Here's a side-by-side comparison:

#### Base Gemma-2B (`--no-adapter`)

> **Prompt:** `Recipe for beef tacos:`

```
Recipe for beef tacos:
1 lb ground beef
2 large onion, chopped
1 large tomato, chopped
3-4 fresh jalapeños, sliced
Salt and pepper to taste
1 tbsp olive oil
1 can refried beans
1 can tomato sauce
2-3 cups shredded cheese (I used mozzarella)
1 cup sour cream
1 bag of your favorite taco shells

Heat the olive oil in a large skillet over medium heat. Brown the ground beef,
then drain off excess grease. Add onions, jalapeño, and tomatoes, along with
salt and pepper. Cook until vegetables are softened. Stir in the can of refried
beans and tomato sauce, then remove from heat.

To assemble tacos, place a layer of beef mixture on the bottom of each taco
shell. Top with some shredded cheese, followed by sour cream. Serve with your
favorite taco toppings, such as lettuce, tomatoes, and salsa!

Enjoy!

<h2>What is the most popular Mexican dish?</h2>

The top five Mexican dishes include tacos, burritos, enchiladas, tamales, and nachos.
```

#### Fine-tuned model

> **Prompt:** `Recipe for beef tacos:`

```
Recipe for beef tacos:
Ingredients:
- 2 lb. ground beef
- 1 can (8 oz.) diced tomatoes
- 1/4 c. chopped onion
- 1/2 tsp. salt
- 1/2 tsp. pepper
- 1/2 tsp. garlic powder
- 1/2 c. water
- 1 can (6 oz.) refried beans
- 1 c. shredded cheese
- 2 flour tortillas

Directions:
- Brown ground beef in a skillet; drain.
- Combine the other ingredients and mix well.
- Pour into a 9 x 5 x 3-inch baking dish.
- Cover with the refried beans and cheese.
- Bake at 350° for 30 minutes.
- Serve with sour cream and lettuce.
```

#### Key differences

| | Base Gemma-2B | Fine-tuned |
|---|---|---|
| **Format** | No section headers, mixed styles | Consistent Ingredients/Directions structure |
| **Bullet style** | None or `•` bullets | Clean `- ` bullets throughout |
| **Focus** | Rambles into commentary, HTML tags, FAQ sections | Stays on-task, recipe only |
| **Measurements** | Inconsistent (some missing) | Precise and uniform |

You can compare them yourself with the `--no-adapter` flag:

```bash
# Base model
python inference.py --no-adapter --prompt "Recipe for banana bread:"

# Fine-tuned
python inference.py --adapter ClaireLee2429/gemma-2b-recipes-lora --prompt "Recipe for banana bread:"
```

## API Server

A FastAPI server with streaming token output via Server-Sent Events (SSE).

### Start the server

```bash
pip install -e ".[serve]"
python server.py --adapter ClaireLee2429/gemma-2b-recipes-lora
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status, model name, and device |
| `POST` | `/generate` | Stream recipe tokens via SSE |

### Generate request

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Recipe for pasta carbonara:"}'
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | (required) | Recipe prompt |
| `max_tokens` | int | 256 | Max tokens to generate (1-1024) |
| `temperature` | float | 0.7 | Sampling temperature (0.1-2.0) |

**Response:** SSE stream of JSON events:

```
data: {"token": "Ingredients"}
data: {"token": ":\n"}
data: {"token": "- 1"}
...
data: {"done": true, "full_text": "Recipe for pasta carbonara:\nIngredients:\n..."}
```

### Server options

| Flag | Default | Description |
|---|---|---|
| `--adapter` | `./processed_data/lora_adapter` | LoRA adapter path or HuggingFace Hub ID |
| `--model` | `google/gemma-2b` | Base model name |
| `--no-adapter` | off | Run base model without adapter |
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8000` | Port to bind to |

## Training on Google Colab (with Unsloth)

For faster training on a free GPU, use the provided Colab notebook which uses [Unsloth](https://unsloth.ai/) for ~2x faster training with ~70% less VRAM:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glee2429/recipe-lm/blob/main/notebooks/train_unsloth.ipynb)

1. Open the notebook in Google Colab
2. Set runtime to **T4 GPU** (Runtime > Change runtime type)
3. Run all cells — training takes ~15-30 min on a T4
4. Download the trained adapter or push it to HuggingFace Hub

The notebook is self-contained and uses the same hyperparameters as the local Dagster pipeline. The resulting LoRA adapter is compatible with the `inference.py` script.

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
