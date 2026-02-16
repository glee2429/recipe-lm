---
title: Recipe-LM API
emoji: 🍳
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
---

# Recipe-LM API

FastAPI streaming server for recipe generation using a fine-tuned Gemma-2B model with LoRA.

- **Model**: [google/gemma-2b](https://huggingface.co/google/gemma-2b) + [ClaireLee2429/gemma-2b-recipes-lora](https://huggingface.co/ClaireLee2429/gemma-2b-recipes-lora)
- **Frontend**: [kitchen-genie](https://github.com/glee2429/kitchen-genie)
- **Source**: [recipe-lm](https://github.com/glee2429/recipe-lm)

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status, model name, and device |
| `POST` | `/generate` | Stream recipe tokens via SSE |

## Usage

```bash
curl -N -X POST https://ClaireLee2429-recipe-lm-api.hf.space/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Recipe for pasta carbonara:", "max_tokens": 256}'
```
