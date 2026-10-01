from __future__ import annotations

import gc
import os
from typing import Any, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 256

# Your GTX 1650 has 4 GB VRAM
MIN_GPU_VRAM_GIB = 3


_model_instance: Optional["QwenVLModel"] = None


class ModelLoadError(RuntimeError):
    """Raised when the model cannot be loaded on the selected device."""


def _select_device() -> str:
    """Choose a safe inference device, optionally honoring an explicit override."""

    requested = os.environ.get("PRODUCT_DESCRIPTION_DEVICE", "auto").lower()

    if requested not in {"auto", "cpu", "cuda"}:
        raise ModelLoadError(
            "PRODUCT_DESCRIPTION_DEVICE must be one of: auto, cpu, cuda."
        )

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError(
                "CUDA was requested, but PyTorch cannot access a CUDA GPU."
            )
        return "cuda"

    if not torch.cuda.is_available():
        return "cpu"

    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if vram_gib < MIN_GPU_VRAM_GIB:
        print(
            f"GPU has {vram_gib:.1f} GiB VRAM; using CPU."
        )
        return "cpu"

    return "cuda"


class QwenVLModel:
    """Thin wrapper around the Qwen2.5-VL model + processor."""

    def __init__(self) -> None:
        self.device = _select_device()
        self.model: Any = None
        self.processor: Any = None

        self._load_on_device(self.device)

    def _load_on_device(self, device: str) -> None:
        """Load Qwen2.5-VL, using 4-bit quantization on CUDA."""

        self.device = device

        print(f"Device: {device.upper()}")

        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(
            f"Loading {MODEL_NAME} "
            "(first run downloads the model - this can take a while)..."
        )

        try:

            # ============================================================
            # CORRECTION 1:
            # Use proper 4-bit quantization instead of torch.int8/FP16
            # ============================================================

            if device == "cuda":

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map={"": 0},
                    low_cpu_mem_usage=True,
                )

            else:

                # CPU fallback
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )

                model.to(device)

            model.eval()

            processor = AutoProcessor.from_pretrained(
                MODEL_NAME
            )

        except RuntimeError as exc:

            is_oom = "out of memory" in str(exc).lower()

            automatic = (
                os.environ.get(
                    "PRODUCT_DESCRIPTION_DEVICE",
                    "auto"
                ).lower()
                == "auto"
            )

            if device == "cuda" and is_oom and automatic:

                print(
                    "GPU memory was exhausted; retrying on CPU."
                )

                if "model" in locals():
                    del model

                gc.collect()
                torch.cuda.empty_cache()

                self._load_on_device("cpu")
                return

            raise ModelLoadError(
                f"Could not load {MODEL_NAME} on {device}: {exc}"
            ) from exc

        except Exception as exc:

            raise ModelLoadError(
                f"Could not load {MODEL_NAME} on {device}: {exc}"
            ) from exc

        self.model = model
        self.processor = processor

        print("Model loaded.\n")

    def generate(
        self,
        image: Optional[Image.Image],
        text_prompt: str,
    ) -> str:

        content: list[dict[str, Any]] = []

        if image is not None:
            content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        content.append(
            {
                "type": "text",
                "text": text_prompt,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]


def get_model() -> QwenVLModel:
    """Return the shared QwenVLModel instance."""

    global _model_instance

    if _model_instance is None:
        _model_instance = QwenVLModel()

    return _model_instance