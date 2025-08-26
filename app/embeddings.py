import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from .config import EMBED_BACKBONE, EMBED_SIZE

class Embedder(nn.Module):
    """Extractor de embeddings con GPU y AMP (FP16)."""
    def __init__(self, device: str = None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = (device == "cuda")
        if device == "cuda":
            torch.backends.cudnn.benchmark = True

        if EMBED_BACKBONE == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(self.device).eval()
            self.out_dim = EMBED_SIZE
        else:
            raise ValueError("Backbone no soportado.")

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, batch_chw: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch_chw).to(self.device, non_blocking=True)
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = self.feature_extractor(x)
        else:
            feats = self.feature_extractor(x)
        feats = feats.squeeze(-1).squeeze(-1)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.float().cpu().numpy()

    @torch.no_grad()
    def encode_single(self, x_chw: np.ndarray) -> np.ndarray:
        return self.encode_batch(x_chw[None, ...])[0]
