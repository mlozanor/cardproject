"""CNN embeddings: frozen ImageNet ResNet-50, global-average-pooled 2048-d vector.

Needs: pip install torch torchvision. Tested 2026-10-21 with torch 2.11 / torchvision 0.26 (GPU).
"""
import cv2
import numpy as np

from .embedding import EmbeddingRetriever


class ResNetRetriever(EmbeddingRetriever):
    name = "resnet"

    def __init__(self, device=None, size=(320, 224)):          # (height, width), ~5:7 card ratio
        import torch
        import torchvision
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def embed(self, imgs):
        arr = []
        for im in imgs:
            rgb = cv2.cvtColor(cv2.resize(im, (self.size[1], self.size[0])), cv2.COLOR_BGR2RGB)
            arr.append(((rgb.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1))
        x = self.torch.from_numpy(np.stack(arr)).to(self.device)
        with self.torch.no_grad():
            return self.model(x).cpu().numpy()
