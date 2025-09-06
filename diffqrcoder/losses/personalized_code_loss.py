import torch
from torch import nn

from .scanning_robust_loss import ScanningRobustLoss


class PersonalizedCodeLoss(nn.Module):
    def __init__(
        self,
        qrcode_image: torch.Tensor,
        content_image: torch.Tensor,
        module_size: int = 16,
        b_thres: float = 50,
        w_thres: float = 200,
        b_soft_value: float = 40 / 255,
        w_soft_value: float = 220 / 255,
        code_weight: float = 1e12,
        content_weight: float = 1e8,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(PersonalizedCodeLoss, self).__init__()
        self.code_loss = ScanningRobustLoss(
            module_size=module_size,
        ).to(device)

        self.content_image = content_image
        self.code_weight = code_weight
        self.content_weight = content_weight
        self.qrcode_image = qrcode_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code_loss = self.code_loss(x, self.qrcode_image)
        perceptual_loss = nn.MSELoss()(x, self.content_image)
        total_loss = (
            self.code_weight * code_loss + \
            self.content_weight * perceptual_loss
        )
        return {
            "code": code_loss,
            "perceptual": perceptual_loss,
            "total": total_loss
        }
