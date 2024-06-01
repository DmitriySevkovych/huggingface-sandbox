import atexit
import json
import logging.config
import logging.handlers
import pathlib
from os.path import dirname, join, realpath


def setup_logging():
    config_file = pathlib.Path(join(dirname(realpath(__file__)), "config.json"))
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
    