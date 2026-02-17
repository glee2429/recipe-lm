"""
Standalone inference script for the fine-tuned recipe generation model (GGUF).

Usage:
    python inference.py --prompt "Recipe for chocolate chip cookies:"
    python inference.py --prompt "Recipe for pasta carbonara:" --save output.txt
    python inference.py --prompt "Recipe for banana bread:" --raw
"""

import argparse
import os
import re

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

GGUF_REPO = os.environ.get("GGUF_REPO", "ClaireLee2429/gemma-2b-recipes-gguf")
GGUF_FILE = os.environ.get("GGUF_FILE", "model.q4_k_m.gguf")


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


def load_model(
    n_threads: int = 8,
    n_ctx: int = 2048,
    model_path: str | None = None,
) -> Llama:
    """Download GGUF model from HuggingFace Hub and load with llama-cpp-python."""
    if model_path is None:
        print(f"Downloading {GGUF_REPO}/{GGUF_FILE}...")
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    print(f"Loading GGUF model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_threads=n_threads,
        n_ctx=n_ctx,
        verbose=False,
    )
    print(f"Model loaded ({n_threads} threads, {n_ctx} ctx).")
    return llm


def stream_recipe(
    llm: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
):
    """Yield token strings as they are generated."""
    for chunk in llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.2,
        stream=True,
    ):
        token_text = chunk["choices"][0]["text"]
        if token_text:
            yield token_text


def generate_recipe(
    llm: Llama,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    raw: bool = False,
) -> str:
    """Generate a complete recipe (non-streaming, for CLI use)."""
    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.2,
    )
    text = prompt + output["choices"][0]["text"]
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
        "--model-path",
        type=str,
        default=None,
        help="Path to a local GGUF file (skips HF Hub download)",
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

    llm = load_model(model_path=args.model_path)

    print(f"\nPrompt: {prompt.strip()}")
    print("-" * 40)

    result = generate_recipe(
        llm,
        prompt,
        max_tokens=args.max_tokens,
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
