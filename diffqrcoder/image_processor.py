from typing import Optional

import torch


IMAGE_MAX_VAL = 255


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())


def convert_to_gray(
    images: torch.Tensor,
    cr: float = 0.2999,
    cg: float = 0.587,
    cb: float = 0.1114,
) -> torch.Tensor:

    assert images.shape[1] == 3, \
        f"The channel of color images must be 3 but get {images.shape[1]}. They are not color images."

    gray_image = cr * images[:, 0] + cg * images[:, 1] + cb * images[:, 2]
    return gray_image.unsqueeze(1)


def image_binarize(
    image: torch.Tensor,
    binary_threshold: Optional[float] = None,
) -> torch.Tensor:

    if image.shape[1] == 3:
        image = convert_to_gray(image)

    if binary_threshold is None:
        if image.max() <= 1:
            binary_threshold = 0.5
        else:
            binary_threshold = 0.5 * IMAGE_MAX_VAL
    return (image > binary_threshold).to(image.dtype)


def crop_padding(x: torch.Tensor, padding: int):
    return x[:, :, padding:-padding, padding:-padding]
