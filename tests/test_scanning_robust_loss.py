import numpy as np
import pytest
import torch
from torch import nn

from diffqrcoder.losses.scanning_robust_loss import (
    GaussianFilter,
    RegionMeanFilter,
    CenterPixelExtractor,
    QRCodeErrorExtractor,
    ScanningRobustLoss
)


@pytest.mark.parametrize("module_size", [3, 5])
def test_gaussian_filter(module_size):
    input_size = 15
    input_tensor = torch.rand(1, 1, input_size, input_size)

    gaussian_filter = GaussianFilter(module_size)
    output_tensor = gaussian_filter(input_tensor)
    output_size = (input_size - module_size) // module_size + 1

    assert output_tensor.shape == (1, 1, output_size, output_size)


@pytest.mark.parametrize("module_size", [3, 5])
def test_region_mean_filter(module_size):
    input_size = 15
    input_tensor = torch.rand(1, 1, input_size, input_size)

    region_mean_filter = RegionMeanFilter(module_size)
    output_tensor = region_mean_filter(input_tensor)
    output_size = (input_size - module_size) // module_size + 1

    assert output_tensor.shape == (1, 1, output_size, output_size)


@pytest.mark.parametrize("module_size", [3, 5])
def test_center_pixel_extractor(module_size):
    input_size = 15
    input_tensor = torch.rand(1, 1, input_size, input_size)

    center_pixel_extractor = CenterPixelExtractor(module_size)
    output_tensor = center_pixel_extractor(input_tensor)
    output_size = (input_size - module_size) // module_size + 1

    assert output_tensor.shape == (1, 1, output_size, output_size)


@pytest.mark.parametrize("module_size", [3, 5])
def test_qrcode_error_extractor(module_size):
    input_size = 15
    x = torch.rand(1, 1, input_size, input_size)
    y = torch.randint(0, 2, (1, 1, input_size, input_size)).float()

    extractor = QRCodeErrorExtractor(module_size)
    output_tensor = extractor(x, y)
    output_size = (input_size - module_size) // module_size + 1

    assert output_tensor.shape == (1, 1, output_size, output_size)
    assert output_tensor.max() <= 1.0 and output_tensor.min() >= 0.0


def test_scanning_robust_loss():
    module_size = 3
    loss = ScanningRobustLoss(module_size)
    image = torch.rand(1, 3, 15, 15)
    qrcode = torch.randint(0, 2, (1, 1, 15, 15)).float()

    output_loss = loss(image, qrcode)
    assert isinstance(output_loss, torch.Tensor)
    assert output_loss.ndim == 0
