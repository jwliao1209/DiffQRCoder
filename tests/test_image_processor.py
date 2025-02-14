import pytest
import torch

from diffqrcoder.image_processor import (
    convert_to_gray,
    crop_padding,
    image_binarize,
    min_max_normalize,
    IMAGE_MAX_VAL,
)


def test_min_max_normalize():
    x = torch.tensor([0, 5, 10, 15], dtype=torch.float32)
    normalized_x = min_max_normalize(x)

    assert normalized_x.min() == 0, "Normalization min value should be 0."
    assert normalized_x.max() == 1, "Normalization max value should be 1."

    expected = torch.tensor([0.0, 0.3333, 0.6667, 1.0], dtype=torch.float32)
    assert torch.allclose(normalized_x, expected, atol=1e-4), "Normalized values are incorrect."


def test_convert_to_gray():
    images = torch.rand(2, 3, 4, 4)
    gray_images = convert_to_gray(images)

    assert gray_images.shape == (2, 1, 4, 4), "Gray image should have 1 channel."

    images_invalid = torch.rand(2, 1, 4, 4)
    with pytest.raises(AssertionError):
        convert_to_gray(images_invalid)


def test_image_binarize_with_color_image():
    images = torch.rand(2, 3, 4, 4)
    binarized_images = image_binarize(images, binary_threshold=0.5)

    assert binarized_images.shape == (2, 1, 4, 4), "Binarized image should have the same shape as input."
    assert binarized_images.min() >= 0, "Binarized image should have minimum value 0."
    assert binarized_images.max() <= 1, "Binarized image should have maximum value 1."


def test_image_binarize_with_gray_image():
    images = torch.rand(2, 3, 4, 4)
    binarized_images = image_binarize(images, binary_threshold=0.5)

    assert binarized_images.shape == (2, 1, 4, 4), "Binarized image should have 1 channel."
    assert binarized_images.min() >= 0, "Binarized image should have minimum value 0."
    assert binarized_images.max() <= 1, "Binarized image should have maximum value 1."


def test_image_binarize_auto_threshold():
    images = torch.rand(2, 1, 4, 4) * IMAGE_MAX_VAL
    binarized_images = image_binarize(images)

    assert binarized_images.shape == (2, 1, 4, 4), "Binarized image should have the same shape as input."
    assert binarized_images.min() >= 0, "Binarized image should have minimum value 0."
    assert binarized_images.max() <= 1, "Binarized image should have maximum value 1."


def test_crop_padding():
    images = torch.rand(2, 1, 4, 4)
    padding = 1
    cropped_images = crop_padding(images, padding)

    assert cropped_images.shape[2] == images.shape[2] - padding * 2, "Cropped height is incorrect."
    assert cropped_images.shape[3] == images.shape[3] - padding * 2, "Cropped width is incorrect."
