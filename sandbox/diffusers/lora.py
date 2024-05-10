import os
import random
from enum import Enum

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

CWD = os.getcwd()


class AvailableLoras(Enum):
    TOY = ("toy", "CiroN2022/toy-face")

    @property
    def name(self) -> str:
        return self.value[0]

    @property
    def checkpoint(self) -> str:
        return self.value[1]


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

    # Load LoRA weights
    lora_path = os.path.join(CWD, "models", AvailableLoras.TOY.checkpoint)
    if os.path.isdir(lora_path):
        pipeline.load_lora_weights(
            AvailableLoras.TOY.checkpoint,
            weight_name="toy_face_sdxl.safetensors",
            adapter_name=AvailableLoras.TOY.name,
        )
    else:
        pipeline.load_lora_weights(
            AvailableLoras.TOY.checkpoint,
            weight_name="toy_face_sdxl.safetensors",
            adapter_name=AvailableLoras.TOY.name,
        )
        # pipeline.save_lora_weights(lora_path)

    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")

    return pipeline


def test_pipeline(prompt: str, directory: str, lora: AvailableLoras):
    # checkpoint = "runwayml/stable-diffusion-v1-5"
    checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
    n = 4

    seeds = [random.randint(0, 150) for _ in range(n)]
    inputs = {
        "prompt": n * [prompt],
        "generator": [_get_generator(seed) for seed in seeds],
        "num_inference_steps": 30,
        "cross_attention_kwargs": {"scale": 0.9},
        #'guidance_scale':20.5,
    }

    # instantiate pipeline
    pipeline = _load_pipeline(checkpoint)
    pipeline.set_adapters(lora)

    if n > 1:  # To prevent OOM errors
        # See https://huggingface.co/docs/diffusers/en/stable_diffusion#memory
        # pipeline.enable_attention_slicing()
        # See https://huggingface.co/blog/simple_sdxl_optimizations
        pipeline.enable_vae_slicing()
        pipeline.enable_sequential_cpu_offload()

    # inference
    images = pipeline(**inputs).images

    # Save and show results
    make_image_grid(images, 2, 2).show()

    output_dir = os.path.join(CWD, "data", "diffusers", directory)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n):
        images[i].save(os.path.join(output_dir, f"output_seed-{seeds[i]}.png"))

    return (
        f'Genetrated {n} images for prompt "{prompt}" into the directory {output_dir}'
    )
