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
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                       Voice Interface                        │
│                                                              │
│  recipe-voice-agent ── Call a phone number, get a recipe     │
│                        Claude Haiku routes, Gemma-2B cooks   │
│                        Sends recipe via SMS after the call   │
│                        github.com/glee2429/recipe-voice-agent│
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install and start the API server
pip install -e ".[serve]"
huggingface-cli login
python server.py --adapter ClaireLee2429/gemma-2b-recipes-lora

# Start the frontend (in a separate terminal)
cd ../kitchen-genie && npm install && npm run dev

# Open http://localhost:8080
```

## Project Structure

```
recipe-lm/
├── data_pipeline/          # Dagster training pipeline
│   ├── assets/             #   download, clean, tokenize, split, train
│   ├── resources.py        #   HuggingFaceConfig resource
│   ├── io_managers.py      #   Arrow dataset I/O manager
│   └── definitions.py      #   Dagster definitions
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

## Training

Run the Dagster pipeline locally or use the Colab notebook for free GPU training:

```bash
# Local (Dagster)
pip install -e ".[dev]"
dagster dev

# Colab (Unsloth, ~2x faster)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glee2429/recipe-lm/blob/main/notebooks/train_unsloth.ipynb)

Pipeline: `raw_dataset → cleaned_dataset → tokenized_dataset → train_val_splits → trained_model`

See `configs/default.yaml` for all training parameters (LoRA rank, learning rate, batch size, etc.).

## Inference

```bash
# CLI
python inference.py --prompt "Recipe for chocolate chip cookies:"
python inference.py --adapter ClaireLee2429/gemma-2b-recipes-lora --prompt "Recipe for soup:"
python inference.py --no-adapter --prompt "Recipe for tacos:"  # base model comparison

# API server
python server.py --adapter ClaireLee2429/gemma-2b-recipes-lora
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Recipe for pasta carbonara:"}'
```

Run `python inference.py --help` or `python server.py --help` for all options.

## Tests

```bash
pytest tests/
```

## Related Projects

- [kitchen-genie](https://github.com/glee2429/kitchen-genie) - React web frontend for recipe generation
- [recipe-voice-agent](https://github.com/glee2429/recipe-voice-agent) - Containerized voice interface (phone call + SMS) powered by OpenClaw and ClawdTalk
