"""Transformer embeddings: frozen CLIP image encoder (ViT-B/32 by default).

Needs: pip install torch transformers pillow. Tested 2026-10-21 with torch 2.11 + transformers 5.20.
Preprocessing (CLIPProcessor default): resize the short side to 224, then CENTER-CROP 224x224, so on
a 5:7 card the top and bottom ~13% (name bar, set number) are not seen. ResNet sees the whole card.
"""
import cv2

from .embedding import EmbeddingRetriever


class CLIPRetriever(EmbeddingRetriever):
    name = "clip"

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed(self, imgs):
        from PIL import Image
        pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.get_image_features(**inputs)
        # transformers < 5 returns the 512-d tensor; >= 5 returns an object holding it in pooler_output
        feats = out if self.torch.is_tensor(out) else out.pooler_output
        return feats.cpu().numpy()
