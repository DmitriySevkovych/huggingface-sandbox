import argparse
import logging

from .app_logging.setup import setup_logging
from .diffusers.lora import test_pipeline as diffuser_lora_pipeline
from .diffusers.simple import test_pipeline as diffuser_pipeline
from .transformers.inference.simple import test_pipeline as transformer_pipeline

#
# Logging
#
logger = logging.getLogger(__name__)


#
# CLI interface
#
def _get_args(entrypoint: str):
    parser = argparse.ArgumentParser(description=f"Doodling around with {entrypoint}")
    parser.add_argument("prompt", type=str, help="Provide your prompt")
    if entrypoint == "diffusers":
        parser.add_argument(
            "-d",
            "--dir",
            type=str,
            default="",
            help="Name of the output directory for the generated result files",
        )
        parser.add_argument(
            "-l",
            "--lora",
            type=str,
            default=None,
            help="Name of the LoRA to apply",
        )
    return parser.parse_args()


#
# Some prompt ideas
#
# "Astronaut riding a horse"
# "An image of a squirrel in Picasso style"
# "New Mercedes-Benz headquarters building in downtown Stuttgart. The building has the Mercedes-Benz star on its roof and an advertisement banner on its facade."


#
# Entry points
#
def diffuser():
    """Launched with `poetry run diffuser` at project root level"""
    setup_logging()
    args = _get_args("diffusers")
    if args.lora:
        result = diffuser_lora_pipeline(args.prompt, args.dir, args.lora)
    else:
        result = diffuser_pipeline(args.prompt, args.dir)
    logger.debug(result)


def transformer():
    """Launched with `poetry run transformer` at project root level"""
    setup_logging()
    args = _get_args("transformers")
    result = transformer_pipeline(**args)
    logger.debug(result)


if __name__ == "__main__":
    """Python module entry point. Is run via `python -m api.main` e.g. on Docker container start (cf. Dockerfile)"""
    diffuser()
