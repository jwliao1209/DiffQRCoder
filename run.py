import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from diffusers import ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

from diffqrcoder import DiffQRCoderPipeline


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--controlnet_ckpt",
        type=str,
        default="checkpoints/control_v1p_sd15_qrcode_monster"
    )
    parser.add_argument(
        "--pipe_ckpt",
        type=str,
        default="https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/blob/main/cetusMix_Whalefall2_fp16.safetensors"
    )
    parser.add_argument(
        "--qrcode_path",
        type=str,
        default="qrcode/thanks_reviewer.png"
    )
    parser.add_argument(
        "--qrcode_module_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--qrcode_padding",
        type=int,
        default=78,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Winter wonderland, fresh snowfall, evergreen trees, cozy log cabin, smoke rising from chimney, aurora borealis in night sky.",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="easynegative"
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.35,
    )
    parser.add_argument(
        "-srg",
        "--scanning_robust_guidance_scale",
        type=float,
        default=500,
    )
    parser.add_argument(
        "-pg",
        "--perceptual_guidance_scale",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--srmpgd_num_iteration",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--srmpgd_lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    qrcode = load_image(args.qrcode_path)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_ckpt,
        torch_dtype=torch.float16,
    )
    pipe = DiffQRCoderPipeline.from_single_file(
        args.pipe_ckpt,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_auth_token=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    result = pipe(
        prompt=args.prompt,
        qrcode=qrcode,
        qrcode_module_size=args.qrcode_module_size,
        qrcode_padding=args.qrcode_padding,
        negative_prompt=args.neg_prompt,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=args.device).manual_seed(1),
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        scanning_robust_guidance_scale=args.scanning_robust_guidance_scale,
        perceptual_guidance_scale=args.perceptual_guidance_scale,
        srmpgd_num_iteration=args.srmpgd_num_iteration,
        srmpgd_lr=args.srmpgd_lr,
    )
    result.images[0].save(Path(args.output_folder, "aesthetic_qrcode.png"))
