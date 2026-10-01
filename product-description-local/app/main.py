"""
Local entry point for the Product Description Generation module.

Usage:
    python -m app.main
    python -m app.main path/to/other.jpg

Pipeline:

    Input Image
        ↓
    Background Removal
        ↓
    Super Resolution
        ↓
    Product Analysis
        ↓
    Qwen Description Generation
        ↓
    JSON Output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.image_pipeline import ImagePipeline

from app.description_generator import generate_listing
from app.model import ModelLoadError, get_model
from app.product_analyzer import ImageLoadError, analyze_product


# --------------------------------------------------
# Paths
# --------------------------------------------------

DEFAULT_IMAGE_PATH = "images/product.jpg"

SR_MODEL_PATH = "models/EDSR_x4.pb"

OUTPUT_PATH = Path(
    "outputs/product_listing.json"
)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    # --------------------------------------------------
    # 1. Get input image
    # --------------------------------------------------

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_IMAGE_PATH
    )

    print("\n========================================")
    print("        PRODUCT AI PIPELINE")
    print("========================================")

    print(f"\nInput image: {image_path}")


    # --------------------------------------------------
    # 2. PHASE 1
    # Background Removal + Super Resolution
    # --------------------------------------------------

    print("\n========================================")
    print("           PHASE 1")
    print("      IMAGE ENHANCEMENT")
    print("========================================")

    try:

        pipeline = ImagePipeline(
            sr_model_path=SR_MODEL_PATH
        )

        result = pipeline.process(
            image_path
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError
    ) as exc:

        print(
            f"\nPhase 1 Error: {exc}"
        )

        sys.exit(1)


    # --------------------------------------------------
    # 3. Get final processed image
    # --------------------------------------------------

    processed_image = result["enhanced"]

    print(
        f"\nProcessed image:"
        f"\n{processed_image}"
    )


    # --------------------------------------------------
    # 4. Load Qwen model
    # --------------------------------------------------

    print("\n========================================")
    print("          LOADING QWEN")
    print("========================================")

    try:

        get_model()

    except ModelLoadError as exc:

        print(
            f"Error loading Qwen model: {exc}"
        )

        sys.exit(1)


    # --------------------------------------------------
    # 5. Product Analysis
    # --------------------------------------------------

    print("\n========================================")
    print("        PRODUCT ANALYSIS")
    print("========================================")

    print(
        f"Analyzing processed image: "
        f"{processed_image}"
    )

    try:

        analysis = analyze_product(
            processed_image
        )

    except (
        ImageLoadError,
        ValueError
    ) as exc:

        print(
            f"Error analyzing product: {exc}"
        )

        sys.exit(1)


    print("\n--- Product Analysis ---")

    print(
        analysis.model_dump_json(
            indent=2
        )
    )


    # --------------------------------------------------
    # 6. Generate Qwen listing
    # --------------------------------------------------

    print("\n========================================")
    print("      GENERATING PRODUCT LISTING")
    print("========================================")

    try:

        listing = generate_listing(
            analysis
        )

    except ValueError as exc:

        print(
            f"Error generating listing: {exc}"
        )

        sys.exit(1)


    print("\n--- Product Listing ---")

    print(
        listing.model_dump_json(
            indent=2
        )
    )


    # --------------------------------------------------
    # 7. Save final result
    # --------------------------------------------------

    result_data = {

        "input_image": str(
            image_path
        ),

        "processed_image": str(
            processed_image
        ),

        "no_background_image": str(
            result["no_background"]
        ),

        "product_analysis":
            analysis.model_dump(),

        "listing":
            listing.model_dump(),
    }


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result_data,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        f"\nSaved result to "
        f"{OUTPUT_PATH}"
    )


    print("\n========================================")
    print("          PIPELINE COMPLETE")
    print("========================================")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()