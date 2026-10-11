import torch

from transformers import (
    CLIPProcessor,
    CLIPModel
)

# -----------------------------
# device
# -----------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# -----------------------------
# model
# -----------------------------
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

# -----------------------------
# processor
# -----------------------------
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

print("🔥 CLIP model loaded")