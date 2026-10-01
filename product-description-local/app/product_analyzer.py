"""
Stage 1 of the pipeline: load a product image and turn it into a
structured, validated ProductAnalysis using Qwen2.5-VL.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from app.model import get_model
from app.prompts import build_product_analysis_prompt
from app.schemas import ProductAnalysis

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageLoadError(Exception):
    """Raised when the given path isn't a usable image."""


def load_and_validate_image(image_path: str) -> Image.Image:
    """Open the image, confirm it's a real, supported image, and return it."""
    path = Path(image_path)
    if not path.exists():
        raise ImageLoadError(f"Image not found: {image_path}")

    try:
        with Image.open(path) as check:
            check.verify()  # raises if the file is corrupt/not an image
    except Exception as exc:  # noqa: BLE001 - any failure here means "not a valid image"
        raise ImageLoadError(f"File is not a valid image: {image_path}") from exc

    # verify() leaves the file unusable for further operations, so reopen it.
    image = Image.open(path)
    if image.format not in SUPPORTED_FORMATS:
        raise ImageLoadError(
            f"Unsupported image format '{image.format}'. Use JPG, PNG, or WebP."
        )

    print(f"Loaded image: {path.name} ({image.format}, {image.width}x{image.height})")
    return image.convert("RGB")


def analyze_product(image_path: str) -> ProductAnalysis:
    """Run Stage 1: image path -> validated ProductAnalysis."""
    image = load_and_validate_image(image_path)

    model = get_model()
    prompt = build_product_analysis_prompt()
    raw_output = model.generate(image=image, text_prompt=prompt)

    data = _parse_json_object(raw_output)
    try:
        return ProductAnalysis.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Model's product analysis was missing fields or had the wrong shape:\n{exc}"
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
