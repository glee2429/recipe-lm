"""
Standalone inference script for the fine-tuned recipe generation model.

Usage:
    python inference.py --prompt "Recipe for chocolate chip cookies:"
    python inference.py --prompt "Recipe for pasta carbonara:" --save output.txt
    python inference.py --prompt "Recipe for banana bread:" --raw
    python inference.py --adapter ClaireLee2429/gemma-2b-recipes-lora --prompt "Recipe for soup:"
    python inference.py --no-adapter --prompt "Recipe for Thai green curry:"
"""

import argparse
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_recipe(text: str) -> str:
    """Post-process generated recipe text to remove artifacts."""
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Remove empty or malformed bullet lines (e.g., "- ", "- .", "- ,", "- AZ")
        if re.match(r"^-\s*[.,;:]*\s*$", stripped):
            continue

        # Remove short junk bullets (single word/number fragments like "- AZ", "- 12-07-02.")
        if re.match(r"^-\s+\S{1,10}$", stripped) and not re.match(r"^-\s+\d+", stripped):
            # Allow numeric items like "- 1 cup" but skip junk like "- AZ"
            words_after_dash = stripped[2:].strip()
            if len(words_after_dash.split()) <= 1 and not any(
                c.islower() for c in words_after_dash
            ):
                continue

        # Stop at trailing commentary sections
        if re.match(r"^-?\s*Notes?:", stripped, re.IGNORECASE):
            break
        if re.match(r"^-?\s*Recipe (from|by|submitted)", stripped, re.IGNORECASE):
            break
        if re.match(r"^-?\s*Source:", stripped, re.IGNORECASE):
            break
        if re.match(
            r"^-\s+(I |My |This is |You can |That |He |She |We |It |Visit )",
            stripped,
        ):
            break
        if re.match(r"^-\s+Bon App", stripped):
            break

        cleaned.append(line)

    # Remove duplicate consecutive lines
    deduped = []
    for line in cleaned:
        if not deduped or line.strip() != deduped[-1].strip():
            deduped.append(line)

    # Trim trailing incomplete line (doesn't end with punctuation)
    while deduped:
        last = deduped[-1].strip()
        if not last:
            deduped.pop()
            continue
        if last and last[-1] not in ".!?)\":;":
            deduped.pop()
        else:
            break

    # Remove trailing blank lines
    while deduped and not deduped[-1].strip():
        deduped.pop()

    return "\n".join(deduped)


def parse_ingredients(text: str) -> list[dict]:
    """Extract structured ingredients from generated recipe text."""
    lines = text.split("\n")

    # Find the Ingredients section
    in_ingredients = False
    ingredient_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^Ingredients:?\s*$", stripped, re.IGNORECASE):
            in_ingredients = True
            continue
        if re.match(r"^Directions:?\s*$", stripped, re.IGNORECASE):
            break
        if in_ingredients and stripped.startswith("- "):
            ingredient_lines.append(stripped[2:].strip())

    # Units to recognize (single-letter units like g/l require trailing space or end)
    units = (
        r"cups?|tbsp|tsp|tablespoons?|teaspoons?|lb\.?|lbs\.?|pounds?|oz\.?|ounces?|"
        r"kg|ml|liters?|cloves?|bunch(?:es)?|cans?|sticks?|pieces?|pcs?|"
        r"pinch(?:es)?|dash(?:es)?|slices?|heads?|stalks?|sprigs?|"
        r"large|medium|small|c\.|pt\.|qt\."
    )

    # Category keywords
    produce = {"onion", "garlic", "tomato", "potato", "carrot", "celery", "pepper",
               "lettuce", "spinach", "broccoli", "mushroom", "lemon", "lime", "ginger",
               "cilantro", "parsley", "basil", "avocado", "corn", "bean sprouts",
               "scallion", "green onion", "jalape", "zucchini", "squash", "cabbage",
               "cucumber", "bell pepper", "chili", "banana", "apple", "berry", "mango"}
    protein = {"chicken", "beef", "pork", "shrimp", "salmon", "fish", "turkey", "lamb",
               "bacon", "sausage", "tofu", "ground", "steak", "thigh", "breast", "meat"}
    dairy = {"butter", "milk", "cream", "cheese", "yogurt", "egg", "sour cream",
             "mozzarella", "parmesan", "cheddar", "ricotta", "whipping cream"}
    spices = {"salt", "pepper", "cumin", "paprika", "cinnamon", "oregano", "thyme",
              "chili powder", "garlic powder", "onion powder", "cayenne", "turmeric",
              "nutmeg", "bay leaf", "red pepper flakes", "curry", "coriander"}

    pattern = re.compile(
        rf"^([\d/\.\-\s]+(?:\([^)]+\))?)?\s*({units})?\s*\.?\s*(.+)$",
        re.IGNORECASE,
    )

    ingredients = []
    for line in ingredient_lines:
        m = pattern.match(line)
        if m:
            amount = (m.group(1) or "").strip()
            unit = (m.group(2) or "").strip()
            name = (m.group(3) or line).strip().rstrip(",.")
        else:
            amount, unit, name = "", "", line.strip().rstrip(",.")

        # Classify category
        name_lower = name.lower()
        if any(k in name_lower for k in spices):
            category = "spices"
        elif any(k in name_lower for k in dairy):
            category = "dairy"
        elif any(k in name_lower for k in protein):
            category = "protein"
        elif any(k in name_lower for k in produce):
            category = "produce"
        else:
            category = "pantry"

        ingredients.append({
            "name": name,
            "amount": amount,
            "unit": unit,
            "category": category,
        })

    return ingredients


def load_model(model_name: str, adapter_path: str, no_adapter: bool = False):
    """Load the base model, optionally with a LoRA adapter."""
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    dtype = torch.bfloat16 if use_cuda else (torch.float16 if use_mps else torch.float32)
    if use_cuda:
        device_map = "auto"
    elif use_mps:
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}
        # Use all available CPU cores for inference
        num_threads = int(os.environ.get("OMP_NUM_THREADS", 8))
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

    device_name = "CUDA" if use_cuda else ("MPS" if use_mps else "CPU")
    print(f"Loading base model ({device_name})...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map
    )

    if no_adapter:
        print("Running base model without adapter")
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model.eval()

    device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    return model, tokenizer, device


def generate_recipe(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    raw: bool = False,
) -> str:
    """Generate a recipe from a prompt and optionally post-process."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if raw:
        return text
    return clean_recipe(text)


def main():
    parser = argparse.ArgumentParser(description="Generate recipes with the fine-tuned model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Recipe for chocolate chip cookies:\n",
        help="Prompt for recipe generation",
    )
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
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run the base model without a LoRA adapter",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw output without post-processing",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save output to file",
    )

    args = parser.parse_args()

    # Ensure prompt ends with newline
    prompt = args.prompt if args.prompt.endswith("\n") else args.prompt + "\n"

    model, tokenizer, device = load_model(args.model, args.adapter, args.no_adapter)

    print(f"\nPrompt: {prompt.strip()}")
    print("-" * 40)

    result = generate_recipe(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        raw=args.raw,
    )

    print(result)

    if args.save:
        with open(args.save, "w") as f:
            f.write(result + "\n")
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
