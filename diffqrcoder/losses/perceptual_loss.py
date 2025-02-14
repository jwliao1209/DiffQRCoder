from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16
from torchvision.transforms import Normalize


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        requires_grad: bool = False,
        pretrained_weights: str = "DEFAULT",
    ) -> None:

        super().__init__()
        self.norm = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.slice_indices = [(0, 4), (4, 9), (9, 16), (16, 23)]
        self.slices = nn.ModuleList([nn.Sequential() for _ in range(len(self.slice_indices))])
        self._initialize_slices(pretrained_weights)
        self.features = namedtuple("Outputs", [f"layer{i}" for i in range(len(self.slice_indices))])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def lp_norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2.0, dim=1)

    def _initialize_slices(self, pretrained_weights: str = "DEFAULT") -> None:
        features = vgg16(weights=pretrained_weights).features
        for slice_idx, (start, end) in enumerate(self.slice_indices):
            for i in range(start, end):
                self.slices[slice_idx].add_module(str(i), features[i])

    def forward(self, x: torch.Tensor) -> namedtuple:
        outputs = []
        x = self.norm(x)
        for slice_model in self.slices:
            x = self.lp_norm(slice_model(x))
            outputs.append(x)
        return self.features(*outputs)


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        requires_grad: bool = False,
        pretrained_weights: str = "DEFAULT",
    ):
        super(PerceptualLoss, self).__init__()
        self.extractor = VGGFeatureExtractor(
            pretrained_weights=pretrained_weights,
            requires_grad=requires_grad,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.tensor(
                [
                    torch.nn.functional.mse_loss(fx, fy)
                    for fx, fy in zip(self.extractor(x), self.extractor(y))
                ]
            )
        )
