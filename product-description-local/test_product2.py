from PIL import Image
from app.model import get_model


model = get_model()

image = Image.open("images/product2.jpg").convert("RGB")

# IMPORTANT:
# Reduce resolution because GTX 1650 has only 4 GB VRAM
image.thumbnail((512, 512))

result = model.generate(
    image=image,
    text_prompt="""
    Analyze this product image and generate an e-commerce product listing.

    Include:
    - Product name
    - Product type
    - Key features
    - Visible characteristics
    - Suggested use
    - Attractive product description

    Do not invent information that cannot be determined from the image.
    """
)

print("\nGenerated Product Description:\n")
print(result)