"""
FastAPI server for recipe generation with streaming output.

Usage:
    pip install -e ".[serve]"
    python server.py
    python server.py --adapter ClaireLee2429/gemma-2b-recipes-lora
    python server.py --port 8080
"""

import argparse
import json
import threading
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import TextIteratorStreamer

from inference import clean_recipe, load_model

# Global state for the loaded model
_model = None
_tokenizer = None
_device = None
_model_name = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


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
            full_text += token
            yield {"data": json.dumps({"token": token})}
        cleaned = clean_recipe(full_text)
        yield {"data": json.dumps({"done": True, "full_text": cleaned})}

    return EventSourceResponse(event_stream())


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
