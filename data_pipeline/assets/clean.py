import re

from dagster import asset, AssetExecutionContext
from datasets import Dataset

from data_pipeline.resources import HuggingFaceConfig

# Patterns that mark the start of trailing commentary (everything from
# the matching line onward is removed).  These appear after the actual
# recipe directions in the source dataset.
_TRAILING_PATTERNS = [
    # Attribution / source
    re.compile(r"^-?\s*Submitted\s+by\b", re.IGNORECASE),
    re.compile(r"^-?\s*Recipe\s+(from|by|submitted|courtesy)", re.IGNORECASE),
    re.compile(r"^-?\s*Source:", re.IGNORECASE),
    re.compile(r"^-?\s*Photo\s+(by|from|credit)", re.IGNORECASE),
    re.compile(r"^-?\s*Adapted\s+from\b", re.IGNORECASE),
    re.compile(r"^-?\s*Originally\s+published", re.IGNORECASE),
    # Notes / tips
    re.compile(r"^-?\s*Notes?:", re.IGNORECASE),
    re.compile(r"^-?\s*Tips?:", re.IGNORECASE),
    re.compile(r"^-?\s*Bon\s+App", re.IGNORECASE),
    # Personal commentary (with or without leading dash)
    re.compile(r"^-?\s*(I\s|My\s|This is\s|You can\s|That\s|He\s|She\s|We\s|It\s|Visit\s)", re.IGNORECASE),
    re.compile(r"^-?\s*Thanks", re.IGNORECASE),
    re.compile(r"^-?\s*Thank\s+you", re.IGNORECASE),
    re.compile(r"^-?\s*Another\s+favorite", re.IGNORECASE),
    re.compile(r"^-?\s*The\s+recipe\s+was\b", re.IGNORECASE),
    # Social media / engagement
    re.compile(r"^-?\s*(For\s+more|Check\s+out|Follow\s+us|And\s+follow)", re.IGNORECASE),
    re.compile(r"^-?\s*If\s+you\s+(love|like|enjoy|try)", re.IGNORECASE),
    re.compile(r"^-?\s*Please\s+(give|rate|share|leave)", re.IGNORECASE),
    re.compile(r"^-?\s*(Pinterest|Twitter|Instagram|Facebook|YouTube)\b", re.IGNORECASE),
    re.compile(r"^-?\s*(five|5|4)\s+stars?", re.IGNORECASE),
    # Promo / misc
    re.compile(r"^-?\s*This\s+(post|article|page)\b", re.IGNORECASE),
    re.compile(r"^-?\s*(Disclosure|Affiliate|Sponsored)\b", re.IGNORECASE),
]


def _clean_recipe_text(text: str) -> str:
    """Remove trailing commentary sections from a recipe."""
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        # Stop at the first trailing commentary pattern
        if any(p.match(stripped) for p in _TRAILING_PATTERNS):
            break
        cleaned.append(line)

    # Remove trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned)


@asset(group_name="data_processing")
def cleaned_dataset(
    context: AssetExecutionContext, hf_config: HuggingFaceConfig, raw_dataset: Dataset
) -> Dataset:
    """Clean and filter the raw dataset."""
    text_col = hf_config.text_column
    initial_count = len(raw_dataset)

    if text_col not in raw_dataset.column_names:
        available = raw_dataset.column_names
        context.log.warning(
            f"Column '{text_col}' not found. Available: {available}. "
            f"Skipping text-based cleaning."
        )
        return raw_dataset

    # Strip whitespace
    ds = raw_dataset.map(lambda x: {text_col: x[text_col].strip()})

    # Remove trailing commentary (Submitted by, Notes, Source, etc.)
    ds = ds.map(lambda x: {text_col: _clean_recipe_text(x[text_col])})

    # Remove empty rows
    ds = ds.filter(lambda x: len(x[text_col]) > 0)

    # Deduplicate
    seen = set()

    def dedup(example):
        text = example[text_col]
        if text in seen:
            return False
        seen.add(text)
        return True

    ds = ds.filter(dedup)

    final_count = len(ds)
    removed = initial_count - final_count
    context.log.info(
        f"Cleaning complete: {initial_count} -> {final_count} "
        f"({removed} removed, {removed / initial_count * 100:.1f}%)"
    )
    return ds
