import logging
from typing import List

from typing_extensions import override


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


class IgnoreChangeDetectedFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord):
        return "%d change%s detected: %s" != record.msg


class IgnoreLoggersFilter(logging.Filter):
    def __init__(self, loggers: List[str] = None):
        self.excluded_loggers = loggers

    @override
    def filter(self, record: logging.LogRecord):
        return record.name not in self.excluded_loggers
