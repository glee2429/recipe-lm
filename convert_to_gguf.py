"""
One-time script: merge LoRA adapter into base Gemma-2B and save the
merged model so it can be converted to GGUF with llama.cpp tooling.

Usage:
    pip install torch transformers peft
    python convert_to_gguf.py

Then, with llama.cpp built locally:
    python llama.cpp/convert_hf_to_gguf.py ./merged_model --outtype f16 --outfile model.f16.gguf
    ./llama.cpp/build/bin/llama-quantize model.f16.gguf model.q4_k_m.gguf Q4_K_M

Finally, upload:
    huggingface-cli upload ClaireLee2429/gemma-2b-recipes-gguf model.q4_k_m.gguf
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "google/gemma-2b"
ADAPTER = "ClaireLee2429/gemma-2b-recipes-lora"
OUTPUT_DIR = "./merged_model"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

print(f"Loading LoRA adapter from {ADAPTER}...")
model = PeftModel.from_pretrained(base, ADAPTER)

print("Merging adapter weights into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done. Merged model saved to {OUTPUT_DIR}/")
print("Next steps: see docstring for llama.cpp conversion commands.")
