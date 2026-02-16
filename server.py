"""
FastAPI server for recipe generation with streaming output.

Usage:
    pip install -e ".[serve]"
    python server.py
    python server.py --adapter ClaireLee2429/gemma-2b-recipes-lora
    python server.py --port 8080
"""

import argparse
import base64
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import httpx
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import TextIteratorStreamer

from inference import clean_recipe, load_model, parse_ingredients

# Global state for the loaded model
_model = None
_tokenizer = None
_device = None
_model_name = None


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
    parser.add_argument(
        "--adapter",
        type=str,
        default="./processed_data/lora_adapter",
        help="Path to LoRA adapter (local or HuggingFace Hub ID)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b",
        help="Base model name",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run the base model without a LoRA adapter",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    return parser.parse_args()


args = _parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device, _model_name
    _model_name = args.model
    _model, _tokenizer, _device = load_model(args.model, args.adapter, args.no_adapter)
    print(f"Server ready — model: {args.model}, device: {_device}")
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
        "model": _model_name,
        "device": _device,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt if req.prompt.endswith("\n") else req.prompt + "\n"
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        streamer=streamer,
    )

    thread = threading.Thread(target=_model.generate, kwargs=generate_kwargs)
    thread.start()

    async def event_stream():
        full_text = prompt
        for token in streamer:
            if not token:
                continue
            full_text += token
            yield {"data": json.dumps({"token": token})}
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


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
