import os

import torch
from diffusers import DiffusionPipeline


def _get_generator(seed: int = 0) -> torch.Generator:
    return torch.Generator("cuda").manual_seed(seed)


def test_pipeline(prompt: str):
    seed = 123
    # checkpoint = "runwayml/stable-diffusion-v1-5"
    checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
    precision = torch.float16

    pipeline = DiffusionPipeline.from_pretrained(
        checkpoint,
        torch_dtype=precision,
        use_safetensors=True,
    )

    pipeline = pipeline.to("cuda")

    image = pipeline(
        prompt,
        generator=_get_generator(seed=seed),
        #  num_inference_steps=200,
        #  strength=0.3,
        #  guidance_scale=10.5,
    ).images[0]

    image.save(
        os.path.join(
            os.getcwd(), "data", f"diffusers_output_{str(precision)}_seed-{seed}.png"
        )
    )
