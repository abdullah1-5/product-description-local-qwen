from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class SuperResolution:

    def __init__(
        self,
        model_path: str,
        scale: int = 4
    ):

        self.model_path = Path(model_path)
        self.scale = scale

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Super-resolution model not found: "
                f"{self.model_path}"
            )

        print("[Super Resolution] Loading EDSR...")

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

        self.sr.readModel(
            str(self.model_path)
        )

        self.sr.setModel(
            "edsr",
            self.scale
        )

        print(
            "[Super Resolution] EDSR loaded."
        )

    def enhance(
        self,
        input_path: str,
        output_path: str
    ) -> str:

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Image not found: {input_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "[Super Resolution] Processing..."
        )

        # Load image including alpha channel
        image = Image.open(input_path).convert("RGBA")

        rgba = np.array(image)

        # Separate RGB and alpha
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        # OpenCV uses BGR
        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        # Super resolution on RGB
        enhanced_bgr = self.sr.upsample(bgr)

        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(
            enhanced_bgr,
            cv2.COLOR_BGR2RGB
        )

        # Resize alpha to same 4x size
        new_width = enhanced_rgb.shape[1]
        new_height = enhanced_rgb.shape[0]

        enhanced_alpha = cv2.resize(
            alpha,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Combine RGB + alpha
        enhanced_rgba = np.dstack(
            (
                enhanced_rgb,
                enhanced_alpha
            )
        )

        result = Image.fromarray(
            enhanced_rgba,
            "RGBA"
        )

        result.save(
            output_path,
            "PNG"
        )

        print(
            f"[Super Resolution] "
            f"Saved: {output_path}"
        )

        return str(output_path)