import logging

from .app_logging.setup import setup_logging

# from .transformers.inference import simple
from .diffusers import simple

#
# Logging
#
logger = logging.getLogger(__name__)


#
# Entry points
#
def start():
    """Launched with `poetry run start` at project root level"""
    setup_logging()
    result = simple.test_pipeline("An image of a squirrel in Picasso style")
    logger.debug(result)


if __name__ == "__main__":
    """Python module entry point. Is run via `python -m api.main` e.g. on Docker container start (cf. Dockerfile)"""
    start()
