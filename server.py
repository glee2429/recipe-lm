"""
FastAPI server for recipe generation with streaming output (GGUF).

Usage:
    pip install llama-cpp-python fastapi "uvicorn[standard]" sse-starlette httpx gradio
    python server.py
    python server.py --port 8080
"""

import argparse
import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager

import gradio as gr
import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from inference import (
    GGUF_FILE,
    GGUF_REPO,
    clean_recipe,
    generate_recipe,
    load_model,
    parse_ingredients,
    stream_recipe,
)

_llm = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


class ParseRequest(BaseModel):
    text: str


class ProductSearchRequest(BaseModel):
    query: str
    location_id: str = Field(default="")


# Kroger OAuth2 token cache
_kroger_token: str | None = None
_kroger_token_expiry: float = 0


def _parse_args():
    parser = argparse.ArgumentParser(description="Recipe generation API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a local GGUF file (skips HF Hub download)",
    )
    return parser.parse_args()


args = _parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    _llm = load_model(n_threads=n_threads, n_ctx=2048, model_path=args.model_path)
    print(f"Server ready — GGUF model loaded, {n_threads} threads")
    yield


app = FastAPI(title="Recipe-LM API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": f"{GGUF_REPO}/{GGUF_FILE}",
        "device": "cpu (GGUF Q4_K_M)",
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt if req.prompt.endswith("\n") else req.prompt + "\n"

    async def event_stream():
        full_text = prompt
        for token in stream_recipe(
            _llm,
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        ):
            full_text += token
            yield {"data": json.dumps({"token": token})}
            await asyncio.sleep(0)  # yield control to event loop
        cleaned = clean_recipe(full_text)
        yield {"data": json.dumps({"done": True, "full_text": cleaned})}

    return EventSourceResponse(event_stream())


@app.post("/parse-ingredients")
async def parse_ingredients_endpoint(req: ParseRequest):
    ingredients = parse_ingredients(req.text)
    return {"ingredients": ingredients}


async def _get_kroger_token() -> str | None:
    """Get a valid Kroger OAuth2 token, refreshing if expired."""
    global _kroger_token, _kroger_token_expiry

    client_id = os.environ.get("KROGER_CLIENT_ID")
    client_secret = os.environ.get("KROGER_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    if _kroger_token and time.time() < _kroger_token_expiry:
        return _kroger_token

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.kroger.com/v1/connect/oauth2/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials", "scope": "product.compact"},
        )
        resp.raise_for_status()
        data = resp.json()
        _kroger_token = data["access_token"]
        _kroger_token_expiry = time.time() + data.get("expires_in", 1800) - 60
        return _kroger_token


@app.post("/search-products")
async def search_products(req: ProductSearchRequest):
    token = await _get_kroger_token()
    if not token:
        return {"error": "Kroger API credentials not configured", "products": []}

    params = {"filter.term": req.query, "filter.limit": 3}
    if req.location_id:
        params["filter.locationId"] = req.location_id

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.kroger.com/v1/products",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            params=params,
        )
        if resp.status_code != 200:
            return {"error": f"Kroger API error: {resp.status_code}", "products": []}

        data = resp.json()

    products = []
    for item in data.get("data", []):
        # Extract price
        price = None
        if "items" in item and item["items"]:
            price_info = item["items"][0].get("price", {})
            regular = price_info.get("regular")
            if regular:
                price = f"${regular:.2f}"

        # Extract image
        image_url = ""
        for img in item.get("images", []):
            if img.get("perspective") == "front":
                for size in img.get("sizes", []):
                    if size.get("size") == "medium":
                        image_url = size.get("url", "")
                        break
                break

        products.append({
            "name": item.get("description", ""),
            "brand": item.get("brand", ""),
            "price": price,
            "image_url": image_url,
            "product_id": item.get("productId", ""),
        })

    return {"products": products}


# ---------------------------------------------------------------------------
# Gradio demo UI (mounted at / on HuggingFace Spaces)
# ---------------------------------------------------------------------------

EXAMPLES = [
    "Recipe for chocolate chip cookies:",
    "Recipe for pasta carbonara:",
    "Recipe for chicken stir fry:",
    "Recipe for banana bread:",
    "Recipe for tomato soup:",
]


def _gradio_generate(dish: str, max_tokens: int, temperature: float):
    """Streaming generator for the Gradio interface."""
    prompt = f"Recipe for {dish}:\n" if not dish.lower().startswith("recipe") else dish
    if not prompt.endswith("\n"):
        prompt += "\n"

    full_text = prompt
    for token in stream_recipe(_llm, prompt, max_tokens=max_tokens, temperature=temperature):
        full_text += token
        yield clean_recipe(full_text)
    yield clean_recipe(full_text)


with gr.Blocks(title="Kitchen Genie") as demo:
    gr.Markdown(
        "# Kitchen Genie\n"
        "Generate recipes with a fine-tuned Gemma-2B model (GGUF Q4_K_M). "
        "Type a dish name or pick an example below."
    )

    with gr.Row():
        with gr.Column(scale=1):
            dish_input = gr.Textbox(
                label="What do you want to cook?",
                placeholder="e.g. chocolate chip cookies",
                lines=1,
            )
            with gr.Row():
                max_tok = gr.Slider(64, 512, value=256, step=32, label="Max tokens")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            generate_btn = gr.Button("Generate Recipe", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=dish_input)

        with gr.Column(scale=2):
            output = gr.Textbox(label="Generated Recipe", lines=20)

    generate_btn.click(
        fn=_gradio_generate,
        inputs=[dish_input, max_tok, temp],
        outputs=output,
    )
    dish_input.submit(
        fn=_gradio_generate,
        inputs=[dish_input, max_tok, temp],
        outputs=output,
    )

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
