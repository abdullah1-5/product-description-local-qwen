"""
Pydantic models used across the pipeline.

ProductAnalysis  -> output of Stage 1 (image understanding)
ProductListing   -> output of Stage 2 (e-commerce copywriting)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ProductAnalysis(BaseModel):
    """What Qwen2.5-VL observes directly from the product image."""

    product_name: str
    category: str
    subcategory: Optional[str] = None
    material: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_colors: List[str] = Field(default_factory=list)
    shape: Optional[str] = None
    style: Optional[str] = None
    visible_features: List[str] = Field(default_factory=list)
    craftsmanship_details: List[str] = Field(default_factory=list)
    intended_use: Optional[str] = None
    target_customer: List[str] = Field(default_factory=list)
    search_keywords: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("product_name", "category")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()


class ProductListing(BaseModel):
    """The final e-commerce-ready listing copy, generated from ProductAnalysis."""

    title: str
    short_description: str
    detailed_description: str
    key_features: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    @field_validator("title", "short_description", "detailed_description")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()
