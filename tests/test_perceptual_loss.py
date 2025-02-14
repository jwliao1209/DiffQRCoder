import pytest
import torch
from torchvision.models import VGG16_Weights

from diffqrcoder.losses.perceptual_loss import VGGFeatureExtractor, PerceptualLoss


@pytest.fixture
def sample_tensors():
    x = torch.rand(1, 3, 224, 224)
    y = torch.rand(1, 3, 224, 224)
    return x, y


def test_vgg_feature_extractor_initialization():
    extractor = VGGFeatureExtractor(pretrained_weights=VGG16_Weights.IMAGENET1K_V1)
    assert len(extractor.slices) == 4, "Feature extractor slices should have 4 layers."
    for slice_model in extractor.slices:
        assert isinstance(slice_model, torch.nn.Sequential), "Each slice should be a Sequential model."


def test_vgg_feature_extractor_forward(sample_tensors):
    x, _ = sample_tensors
    extractor = VGGFeatureExtractor(pretrained_weights=VGG16_Weights.IMAGENET1K_V1)
    outputs = extractor(x)

    assert len(outputs) == 4, "Output should contain 4 layers."
    assert outputs.layer0.shape[1] == 64, "First layer output should have 64 channels."
    assert outputs.layer1.shape[1] == 128, "Second layer output should have 128 channels."
    assert outputs.layer2.shape[1] == 256, "Third layer output should have 256 channels."
    assert outputs.layer3.shape[1] == 512, "Fourth layer output should have 512 channels."


def test_perceptual_loss_initialization():
    loss = PerceptualLoss(pretrained_weights=VGG16_Weights.IMAGENET1K_V1)
    assert isinstance(loss.extractor, VGGFeatureExtractor), "Loss should have a VGGFeatureExtractor."


def test_perceptual_loss_forward(sample_tensors):
    x, y = sample_tensors
    perceptual_loss = PerceptualLoss(pretrained_weights=VGG16_Weights.IMAGENET1K_V1)

    loss_value = perceptual_loss(x, y)
    assert isinstance(loss_value, torch.Tensor), "Loss output should be a tensor."
    assert loss_value.shape == torch.Size([]), "Loss output should be a scalar tensor."
