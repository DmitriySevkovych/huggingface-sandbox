import logging

from .app_logging.setup import setup_logging

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
    logger.debug("Hi there!")
    logger.warning("Hi there!")


if __name__ == "__main__":
    """Python module entry point. Is run via `python -m api.main` e.g. on Docker container start (cf. Dockerfile)"""
    start()
