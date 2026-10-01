from pathlib import Path

from app.background_remover import BackgroundRemover
from app.super_resolution import SuperResolution


class ImagePipeline:

    def __init__(
        self,
        sr_model_path: str
    ):

        print("\n==============================")
        print("INITIALIZING IMAGE PIPELINE")
        print("==============================")

        self.background_remover = (
            BackgroundRemover()
        )

        self.super_resolution = (
            SuperResolution(
                model_path=sr_model_path,
                scale=4
            )
        )

        print(
            "Image pipeline ready."
        )

    def process(
        self,
        input_path: str
    ) -> dict:

        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {input_path}"
            )

        # ----------------------------
        # STEP 1
        # Background removal
        # ----------------------------

        no_background_path = (
            Path("outputs/no_background")
            / f"{input_path.stem}_no_bg.png"
        )

        print("\n[PHASE 1 - STEP 1]")
        print("Removing background...")

        self.background_remover.remove(
            input_path=str(input_path),
            output_path=str(no_background_path)
        )

        # ----------------------------
        # STEP 2
        # Super resolution
        # ----------------------------

        enhanced_path = (
            Path("outputs/enhanced")
            / f"{input_path.stem}_enhanced.png"
        )

        print("\n[PHASE 1 - STEP 2]")
        print("Increasing resolution...")

        self.super_resolution.enhance(
            input_path=str(no_background_path),
            output_path=str(enhanced_path)
        )

        # ----------------------------
        # RESULT
        # ----------------------------

        print("\n==============================")
        print("PHASE 1 COMPLETE")
        print("==============================")

        print(
            f"Final image: {enhanced_path}"
        )

        return {
            "input": str(input_path),
            "no_background": str(
                no_background_path
            ),
            "enhanced": str(
                enhanced_path
            )
        }
