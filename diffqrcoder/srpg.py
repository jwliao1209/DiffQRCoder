from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import BaseOutput
from torch import nn

from diffqrcoder.losses import PerceptualLoss, ScanningRobustLoss



GRADIENT_SCALE = 100


class ScanningRobustPerceptualGuidance(nn.Module):
    def __init__(
        self,
        module_size: int = 20,
        scanning_robust_guidance_scale: int = 500,
        perceptual_guidance_scale: int = 2,
    ):
        super().__init__()
        self.module_size = module_size
        self.scanning_robust_guidance_scale = scanning_robust_guidance_scale
        self.perceptual_guidance_scale = perceptual_guidance_scale
        self.scanning_robust_loss_fn = ScanningRobustLoss(module_size=module_size)
        self.perceptual_loss_fn = PerceptualLoss()

    def compute_loss(self, image: torch.Tensor, qrcode: torch.Tensor, ref_image: torch.Tensor) -> torch.Tensor:
        return (
            self.scanning_robust_guidance_scale * self.scanning_robust_loss_fn(image, qrcode) + \
            self.perceptual_guidance_scale * self.perceptual_loss_fn(image, ref_image)
        ) * GRADIENT_SCALE

    def compute_score(self, latents: torch.Tensor, image: torch.Tensor, qrcode: torch.Tensor, ref_image: torch.Tensor) -> torch.Tensor:
        loss = self.compute_loss(image, qrcode, ref_image)
        return torch.autograd.grad(loss, latents)[0] / GRADIENT_SCALE
