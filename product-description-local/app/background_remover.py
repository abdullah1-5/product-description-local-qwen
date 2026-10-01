from pathlib import Path

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover:

    def __init__(self):
        print("[Background] Loading model...")

        # Load once and reuse it
        self.session = new_session()

        print("[Background] Model loaded.")

    def remove(
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

        print("[Background] Removing background...")

        image = Image.open(input_path)

        # Convert to RGBA
        image = image.convert("RGBA")

        result = remove(
            image,
            session=self.session
        )

        result.save(output_path)

        print(
            f"[Background] Saved: {output_path}"
        )

        return str(output_path)