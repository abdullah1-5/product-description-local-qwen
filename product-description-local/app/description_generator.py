"""
Stage 2 of the pipeline: turn a verified ProductAnalysis into e-commerce
listing copy using the same (already loaded) Qwen2.5-VL model, text-only.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.model import get_model
from app.prompts import build_description_prompt
from app.schemas import ProductAnalysis, ProductListing


def generate_listing(product_analysis: ProductAnalysis) -> ProductListing:
    """Run Stage 2: ProductAnalysis -> validated ProductListing."""
    model = get_model()
    analysis_json = product_analysis.model_dump_json(indent=2)
    prompt = build_description_prompt(analysis_json)

    # No image passed here - description generation is text-only, reusing
    # the same model instance already sitting in memory.
    raw_output = model.generate(image=None, text_prompt=prompt)

    data = _parse_json_object(raw_output)
    try:
        return ProductListing.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Model's generated listing was missing fields or had the wrong shape:\n{exc}"
        ) from exc


def _parse_json_object(text: str) -> dict:
    """Pull a JSON object out of the model's raw text output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return valid JSON:\n{text}") from None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model returned malformed JSON:\n{text}") from exc
