import numpy as np
import pytest
import torch
from diffusers.schedulers import DDIMScheduler
from unittest.mock import Mock

from diffqrcoder.losses import PerceptualLoss, ScanningRobustLoss
from diffqrcoder.srpg import ScanningRobustPerceptualGuidance


@pytest.fixture
def setup_guidance_scheduler():
    guidance_scheduler = ScanningRobustPerceptualGuidance(
        module_size=20,
        scanning_robust_guidance_scale=500,
        perceptual_guidance_scale=10,
    )
    return guidance_scheduler


@pytest.fixture
def test_inputs():
    image = torch.rand(1, 3, 592, 592, requires_grad=True)
    qrcode = torch.rand(1, 1, 592, 592)
    ref_image = torch.rand(1, 3, 592, 592, requires_grad=True)
    return image, qrcode, ref_image


def test_compute_loss(setup_guidance_scheduler, test_inputs):
    guidance_scheduler = setup_guidance_scheduler
    image, qrcode, ref_image = test_inputs
    loss = guidance_scheduler.compute_loss(image, qrcode, ref_image)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.shape == torch.Size([]), "Loss should be a scalar."


def test_compute_score(setup_guidance_scheduler, test_inputs):
    guidance_scheduler = setup_guidance_scheduler
    image, qrcode, ref_image = test_inputs
    score = guidance_scheduler.compute_score(image, image, qrcode, ref_image)

    assert isinstance(score, torch.Tensor), "Score should be a torch.Tensor."
    assert score.shape == image.shape, "Score should have the same shape as the input image."
