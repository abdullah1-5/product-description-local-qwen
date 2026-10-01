# Product Description Generation (Local, Qwen2.5-VL)

A local Python module that takes a product photo and produces a structured
e-commerce listing, entirely offline, using the open-source
`Qwen/Qwen2.5-VL-3B-Instruct` vision-language model - no cloud API, no key,
no cost.

## What it does

```
Product Image
    ↓
Qwen2.5-VL 3B  (image understanding)
    ↓
Product Analysis          (Pydantic-validated)
    ↓
Qwen2.5-VL 3B  (text-only, same loaded model)
    ↓
E-commerce Listing         (Pydantic-validated)
    ↓
JSON Output (outputs/product_listing.json)
```

**Stage 1 - Product Analysis.** The image is sent to Qwen2.5-VL, which is
prompted to identify the actual product (not the table, background, props,
etc.) and extract only what's visibly determinable - material, color,
shape, style, visible features, and so on. Anything it can't reasonably
tell from the image (dimensions, brand, price, origin...) comes back as
`null` rather than a guess.

**Stage 2 - Description Generation.** The validated `ProductAnalysis` is
handed back to the *same* loaded model, this time with a text-only prompt,
to write the title, descriptions, key features, and search keywords -
using only facts from Stage 1, nothing invented.

## Architecture / folder structure

```
product-description/
├── app/
│   ├── __init__.py
│   ├── main.py                  # entry point - run this
│   ├── model.py                 # loads Qwen2.5-VL once, exposes generate()
│   ├── product_analyzer.py      # Stage 1: image -> ProductAnalysis
│   ├── description_generator.py # Stage 2: ProductAnalysis -> ProductListing
│   ├── prompts.py                # prompt text for both stages
│   └── schemas.py                # ProductAnalysis / ProductListing (Pydantic)
├── images/
│   └── product.jpg               # put your product photo here
├── outputs/
│   └── product_listing.json      # written after each run
├── requirements.txt
├── .gitignore
└── README.md
```

`app/model.py` loads the model exactly once, the first time `get_model()`
is called (from `main.py`), and both stages reuse that same instance.

## Requirements

- Python 3.10-3.12 (the supported Windows range for PyTorch)
- Windows. GPU inference is selected automatically only on NVIDIA GPUs with
  at least 8 GB VRAM. Smaller GPUs, including this computer's 4 GB GTX 1650,
  use CPU to avoid an out-of-memory failure.

## 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install Python 3.12 first if `py -3.12` is unavailable. Do not reuse the
existing Python 3.14 environment: PyTorch for Windows supports Python 3.9-3.12.

## 2. Install PyTorch first

Install PyTorch *before* the rest of `requirements.txt`, using the command
for your CUDA version from the official selector: https://pytorch.org/get-started/locally/

For this 4 GB GTX 1650, use CPU-only PyTorch:

```powershell
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with at least 8 GB VRAM, choose the command for Windows,
Pip, and your CUDA version in the official PyTorch selector instead.

## 3. Install the rest of the dependencies

```powershell
pip install -r requirements.txt
```

## 4. Verify CUDA is detected

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

On this computer, `True` only means CUDA is available; the app intentionally
chooses CPU because the GTX 1650 has only 4 GB VRAM.

## 5. Add a product photo

Drop a JPG, PNG, or WebP image at:

```
images/product.jpg
```

(or pass any other path as a command-line argument - see below).

## 6. Run it

```powershell
python -m app.main
```

or with a specific image:

```powershell
python -m app.main path\to\your\photo.jpg
```

**First run** will download the ~3B-parameter model from Hugging Face
(several GB) and cache it locally - this only happens once. Every
subsequent run loads it from the local cache.

The program prints:
- which device it's using (`Device: CUDA` / `GPU: NVIDIA GeForce RTX 3050`,
  or `Device: CPU`)
- the loaded image's format and dimensions
- the Stage 1 product analysis, as JSON
- the Stage 2 listing, as JSON

...and writes the combined result to `outputs/product_listing.json`, e.g.:

```json
{
  "product_analysis": {
    "product_name": "Handcrafted Wooden Bowl",
    "category": "Home Decor",
    "subcategory": "Decorative Bowl",
    "material": "Wood",
    "primary_color": "Brown",
    "secondary_colors": [],
    "shape": "Round",
    "style": "Traditional",
    "visible_features": ["Natural wood texture", "Rounded shape"],
    "craftsmanship_details": ["Visible handcrafted appearance"],
    "intended_use": "Home decoration",
    "target_customer": ["Home decor enthusiasts"],
    "search_keywords": ["handmade wooden bowl", "wooden decor", "artisan home decor"],
    "confidence": 0.91
  },
  "listing": {
    "title": "Handcrafted Wooden Decorative Bowl",
    "short_description": "...",
    "detailed_description": "...",
    "key_features": ["Natural wood texture", "Rounded decorative design", "Handcrafted appearance"],
    "keywords": ["wooden bowl", "handmade decor", "artisan home decor"]
  }
}
```

Exact content depends on the photo you provide.

## Notes / assumptions

- **GPU memory:** Qwen2.5-VL-3B in float16 needs roughly 7-8 GB of VRAM.
  The app therefore uses CPU automatically below 8 GB. Set
  `PRODUCT_DESCRIPTION_DEVICE=cuda` only to deliberately try GPU inference,
  or set it to `cpu` to force CPU. CPU inference is much slower (often well
  over a minute per image).
- Both pipeline stages currently run with a single fixed prompt and no
  retry logic - if the model's JSON output is malformed, the run stops
  with a clear error message rather than silently guessing at values.
- Per your instructions, this module intentionally has no API layer, no
  frontend, no database, and no automated test suite - it's a single
  local script (`python -m app.main`) with the two stages you asked for,
  nothing more.
