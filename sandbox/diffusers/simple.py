import os

import torch
from diffusers import DiffusionPipeline


def _get_generator(seed: int = 0) -> torch.Generator:
    return torch.Generator("cuda").manual_seed(seed)


def test_pipeline(prompt: str):
    checkpoint = "runwayml/stable-diffusion-v1-5"
    pipeline = DiffusionPipeline.from_pretrained(
        f"./models/{checkpoint}", use_safetensors=True
    )
    # pipeline.save_pretrained(f"./models/{checkpoint}")

    pipeline = pipeline.to("cuda")

    image = pipeline(prompt, generator=_get_generator()).images[0]

    image.save(os.path.join(os.getcwd(), "data", "diffusers_output.png"))
