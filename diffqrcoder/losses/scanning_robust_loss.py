import math

import cv2
import numpy as np
import torch
from torch import nn

from ..image_processor import convert_to_gray, image_binarize, min_max_normalize


class GaussianFilter(nn.Module):
    def __init__(self, module_size: int) -> None:
        super().__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=False,
            groups=1,
        )
        self._setup_filter_weights()

    def _setup_filter_weights(self) -> None:
        filter_1d = cv2.getGaussianKernel(
            ksize=self.module_size,
            sigma=int((self.module_size -1) / 5),
            ktype=cv2.CV_32F
        )
        filter_2d = filter_1d * filter_1d.T
        gaussian_filter_init = torch.tensor(filter_2d, dtype=torch.float32)
        self.conv.weight = nn.Parameter(
            gaussian_filter_init.reshape(1, 1, *gaussian_filter_init.shape),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RegionMeanFilter(nn.Module):
    def __init__(self, module_size: int) -> None:
        super().__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
            groups=1,
        )
        self._setup_kernel_weights()

    def _setup_kernel_weights(self) -> None:
        module_center = int(self.module_size / 2)
        radius = math.ceil(self.module_size / 6)
        center_filter = torch.zeros((1, 1, self.module_size, self.module_size))
        center_filter[
            :, :,
            module_center-radius : module_center+radius,
            module_center-radius : module_center+radius,
        ] = 1.0

        self.conv.weight = nn.Parameter(
            center_filter / center_filter.sum(),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CenterPixelExtractor(nn.Module):
    def __init__(self, module_size: int) -> None:
        super().__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
            groups=1,
        )
        self._setup_kernel_weights()

    def _setup_kernel_weights(self) -> None:
        module_center = int(self.module_size / 2) + 1
        center_filter = torch.zeros((1, 1, self.module_size, self.module_size))
        center_filter[:, :, module_center, module_center] = 1.0
        self.conv.weight = nn.Parameter(center_filter, requires_grad=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class QRCodeErrorExtractor(nn.Module):
    def __init__(self, module_size: int) -> None:
        super().__init__()
        self.module_size = module_size
        self.region_mean_filter = RegionMeanFilter(module_size)
        self.center_pixel_extractor = CenterPixelExtractor(module_size=module_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_center_mean = self.region_mean_filter(x)
        y_center_pixel = self.center_pixel_extractor(y)
        error_mask = (y_center_pixel == 0) & (x_center_mean > 0.45) | \
                     (y_center_pixel == 1) & (x_center_mean < 0.65)
        return error_mask.float()


class ScanningRobustLoss(nn.Module):
    def __init__(self, module_size: int) -> None:
        super().__init__()
        self.gaussian_filter = GaussianFilter(module_size=module_size)
        self.center_filter = RegionMeanFilter(module_size=module_size)
        self.module_error_extractor = QRCodeErrorExtractor(module_size=module_size)

    def _compute_error(self, image: torch.Tensor, qrcode: torch.Tensor) -> torch.Tensor:
        gray_image = convert_to_gray(image)
        error0 = 2 * torch.relu(gray_image - 0.45) * (1 - qrcode)
        error1 = 2 * torch.relu(0.65 - gray_image) * qrcode
        return error0 + error1

    def _compute_ealy_stopping_mask(self, image: torch.Tensor, qrcode: torch.Tensor) -> torch.Tensor:
        return self.module_error_extractor(
            convert_to_gray(image.clone().detach()),
            image_binarize(qrcode),
        )

    def forward(self, image: torch.Tensor, qrcode: torch.Tensor) -> torch.Tensor:
        error = self._compute_error(image, qrcode)
        sample_error = self.gaussian_filter(error)
        mask = self._compute_ealy_stopping_mask(image, qrcode)
        return torch.mean(sample_error * mask)
