import os

from diffusers import DiffusionPipeline


def test_pipeline(prompt: str):
    # From diffusers tutorial
    pipeline = DiffusionPipeline.from_pretrained(
        "./models/stable-diffusion-v1-5", use_safetensors=True
    )
    # pipeline.save_pretrained("./models")
    # Newest, smallest, based off SD v2.1
    # pipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
    # pipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
    # pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

    image = pipeline(prompt).images[0]
    image.save(os.path.join(os.getcwd(), "data", "diffusers_output.png"))
