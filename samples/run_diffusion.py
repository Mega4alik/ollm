"""Minimal diffusion pipeline demo for low-VRAM GPUs."""

import argparse
from pathlib import Path

from PIL import Image

from ollm import Inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a diffusion pipeline with CPU/disk offload")
    parser.add_argument("model_id", help="Registered model identifier, e.g. sdxl-base-1.0 or qwen-image-edit")
    parser.add_argument("prompt", help="Text prompt to render")
    parser.add_argument("--output", default="output.png", help="Path to the generated image")
    parser.add_argument("--models-dir", default="./models", help="Directory to cache/download model weights")
    parser.add_argument("--device", default="cuda:0", help="Torch device (cuda:N or cpu)")
    parser.add_argument("--num-steps", type=int, default=30, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=5.5, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=None, help="Output height (defaults to pipeline default)")
    parser.add_argument("--width", type=int, default=None, help="Output width (defaults to pipeline default)")
    parser.add_argument("--negative", default=None, help="Negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic outputs")
    parser.add_argument("--download-url", default=None, help="Override download URL (e.g. CivitAI direct link)")
    parser.add_argument("--image", default=None, help="Optional init image for img2img/edit pipelines")
    parser.add_argument("--mask", default=None, help="Optional mask image for inpainting/editing pipelines")
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Denoising strength when using an init image (defaults to pipeline preference)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.download_url:
        overrides["download_url"] = args.download_url

    inference = Inference(
        args.model_id,
        device=args.device,
        **overrides,
    )
    inference.ini_model(models_dir=args.models_dir)

    image = None
    if args.image:
        image = Image.open(args.image)
        if image.mode != "RGB":
            image = image.convert("RGB")

    mask_image = None
    if args.mask:
        mask_image = Image.open(args.mask)
        if mask_image.mode not in ("L", "1"):
            mask_image = mask_image.convert("L")

    strength = args.strength
    if strength is None and image is not None:
        strength = 0.55

    result = inference.generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=args.seed,
        output_type="pil",
        image=image,
        mask_image=mask_image,
        strength=strength,
    )

    output_image = result.images[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
