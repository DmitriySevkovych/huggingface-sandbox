import logging

from .app_logging.setup import setup_logging
from .diffusers.simple import test_pipeline as diffuser_pipeline
from .transformers.inference.simple import test_pipeline as transformer_pipeline

#
# Logging
#
logger = logging.getLogger(__name__)


#
# Entry points
#
def diffuser():
    """Launched with `poetry run diffuser` at project root level"""
    setup_logging()
    result = diffuser_pipeline("An image of a squirrel in Picasso style")
    logger.debug(result)


def transformer():
    """Launched with `poetry run transformer` at project root level"""
    setup_logging()
    result = transformer_pipeline("An image of a squirrel in Picasso style")
    logger.debug(result)


if __name__ == "__main__":
    """Python module entry point. Is run via `python -m api.main` e.g. on Docker container start (cf. Dockerfile)"""
    diffuser()
