#!/usr/bin/env python
import logging
from copy import copy

try:
    from uvicorn.logging import DefaultFormatter

except ImportError:

    class DefaultFormatter(logging.Formatter):  # type: ignore

        def formatMessage(self, record: logging.LogRecord) -> str:
            recordcopy = copy(record)
            levelname = recordcopy.levelname
            seperator = ' ' * (8 - len(recordcopy.levelname))
            recordcopy.__dict__['levelprefix'] = levelname + ':' + seperator
            return super().formatMessage(recordcopy)


def get_logger(name: str) -> logging.Logger:
    """Return a new logger."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = DefaultFormatter('%(levelprefix)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
