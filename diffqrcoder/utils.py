import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def convert_pil_to_normalized_tensor(image: Image) -> torch.Tensor:
    return pil_to_tensor(image).unsqueeze(0).float()


def convert_normalized_tensor_to_np_image(image: torch.Tensor) -> np.array:
    return image.clip(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def add_position_pattern(
    x: torch.Tensor,
    y: torch.Tensor,
    module_num: int,
    module_size: int
) -> torch.Tensor:

    x[: 8 * module_size - 1, : 8 * module_size - 1, :] = \
        y[: 8 * module_size - 1, : 8 * module_size - 1, :]

    x[
        (module_num - 8) * module_size + 1 : module_num * module_size,
        : 8 * module_size - 1,
        :
    ] = y[
        (module_num - 8) * module_size + 1 : module_num * module_size,
        : 8 * module_size - 1,
        :
    ]

    x[
        : 8 * module_size - 1,
        (module_num - 8) * module_size + 1 : module_num * module_size,
        :
    ] = y[
        : 8 * module_size - 1,
        (module_num - 8) * module_size + 1 : module_num * module_size,
        :
    ]

    x[
        (module_num - 9) * module_size : (module_num - 4) * module_size - 1,
        (module_num - 9) * module_size : (module_num - 4) * module_size - 1,
        :
    ] = y[
        (module_num - 9) * module_size : (module_num - 4) * module_size - 1,
        (module_num - 9) * module_size : (module_num - 4) * module_size - 1,
        :
    ]
    return x
