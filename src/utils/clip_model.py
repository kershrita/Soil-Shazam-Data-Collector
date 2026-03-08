"""Singleton CLIP model loader for shared use across filtering and labeling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import open_clip
import torch
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_instance: CLIPModel | None = None


class CLIPModel:
    """Manages an OpenCLIP model instance with batched encode/similarity methods."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        batch_size: int = 64,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU (will be slower)")
            device = "cpu"

        self.device = torch.device(device)
        self.batch_size = batch_size

        logger.info(f"Loading CLIP model {model_name} ({pretrained}) on {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        logger.info("CLIP model loaded successfully")

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL images into normalized CLIP embeddings.

        Returns tensor of shape (N, embed_dim) on CPU.
        """
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings into normalized CLIP embeddings.

        Returns tensor of shape (N, embed_dim) on CPU.
        """
        all_features = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    def similarity(
        self, images: list[Image.Image], texts: list[str]
    ) -> torch.Tensor:
        """Compute cosine similarity between images and texts.

        Returns tensor of shape (num_images, num_texts).
        """
        img_features = self.encode_images(images)
        txt_features = self.encode_texts(texts)
        return img_features @ txt_features.T


def get_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda",
    batch_size: int = 64,
) -> CLIPModel:
    """Get or create the singleton CLIP model instance."""
    global _instance
    if _instance is None:
        _instance = CLIPModel(model_name, pretrained, device, batch_size)
    return _instance
