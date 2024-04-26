import os
import random

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

CWD = os.getcwd()


def _get_generator(seed: int = 0) -> torch.Generator:
    return torch.Generator("cuda").manual_seed(seed)


def _load_pipeline(checkpoint: str) -> DiffusionPipeline:
    args = {"torch_dtype": torch.float16, "use_safetensors": True}
    model_path = os.path.join(CWD, "models", checkpoint)
    if os.path.isdir(model_path):
        pipeline = DiffusionPipeline.from_pretrained(model_path, **args)
    else:
        pipeline = DiffusionPipeline.from_pretrained(checkpoint, **args)
        pipeline.save_pretrained(model_path)

    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")

    return pipeline


def test_pipeline(prompt: str):
    # checkpoint = "runwayml/stable-diffusion-v1-5"
    checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
    n = 4

    seeds = [random.randint(0, 150) for _ in range(n)]
    inputs = {
        "prompt": n * [prompt],
        "generator": [_get_generator(seed) for seed in seeds],
        "num_inference_steps": 20,
        #'guidance_scale':20.5,
    }

    # instantiate pipeline
    pipeline = _load_pipeline(checkpoint)

    if n > 1:  # To prevent OOM errors
        # See https://huggingface.co/docs/diffusers/en/stable_diffusion#memory
        # pipeline.enable_attention_slicing()
        # See https://huggingface.co/blog/simple_sdxl_optimizations
        pipeline.enable_vae_slicing()
        pipeline.enable_sequential_cpu_offload()

    # inference
    images = pipeline(**inputs).images

    # Save and showresults
    make_image_grid(images, 2, 2).show()
    for i in range(n):
        images[i].save(
            os.path.join(CWD, "data", f"diffusers_output_seed-{seeds[i]}.png")
        )

    return f'Genetrated {n} images for prompt "{prompt}"'
