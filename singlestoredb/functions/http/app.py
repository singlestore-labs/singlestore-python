#!/usr/bin/env python
import asyncio
import functools
import logging
import sys
from copy import copy
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

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


logger = get_logger('singlestoredb.functions.http.app')


def post(path: str, **kwargs: Any) -> Callable[..., Any]:
    """Configure a POST endpoint."""
    attrs = dict(method='post', args=[path], kwargs=kwargs)

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            return func(*args, **kwargs)  # type: ignore
        wrapper._singlestoredb_http_attrs = attrs  # type: ignore
        return functools.wraps(func)(wrapper)

    return decorate


def get(path: str, **kwargs: Any) -> Callable[..., Any]:
    """Configure a GET endpoint."""
    attrs = dict(method='get', args=[path], kwargs=kwargs)

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            return func(*args, **kwargs)  # type: ignore
        wrapper._singlestoredb_http_attrs = attrs  # type: ignore
        return functools.wraps(func)(wrapper)

    return decorate


def put(path: str, **kwargs: Any) -> Callable[..., Any]:
    """Configure a PUT endpoint."""
    attrs = dict(method='put', args=[path], kwargs=kwargs)

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            return func(*args, **kwargs)  # type: ignore
        wrapper._singlestoredb_http_attrs = attrs  # type: ignore
        return functools.wraps(func)(wrapper)

    return decorate


def delete(path: str, **kwargs: Any) -> Callable[..., Any]:
    """Configure a DELETE endpoint."""
    attrs = dict(method='delete', args=[path], kwargs=kwargs)

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            return func(*args, **kwargs)  # type: ignore
        wrapper._singlestoredb_http_attrs = attrs  # type: ignore
        return functools.wraps(func)(wrapper)

    return decorate


def patch(path: str, **kwargs: Any) -> Callable[..., Any]:
    """Configure a PATCH endpoint."""
    attrs = dict(method='patch', args=[path], kwargs=kwargs)

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            return func(*args, **kwargs)  # type: ignore
        wrapper._singlestoredb_http_attrs = attrs  # type: ignore
        return functools.wraps(func)(wrapper)

    return decorate


__all__ = ['get', 'put', 'post', 'patch', 'delete']


def main(argv: Optional[List[str]] = None) -> None:
    """Run the HTTP server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError('the uvicorn package is required to run this command')

    try:
        import fastapi
    except ImportError:
        raise ImportError('the fastapi package is required to run this command')

    # Namespace for notebook functions
    import __main__

    app = fastapi.FastAPI()

    for v in list(vars(__main__).values()):

        # See if this is an exported endpoint
        attrs = getattr(v, '_singlestoredb_http_attrs', None)
        if attrs is None:
            continue

        # Apply the function to the FastAPI app
        getattr(app, attrs['method'])(*attrs['args'], **attrs['kwargs'])(v)

    asyncio.create_task(
        _run_uvicorn(
            uvicorn,
            app,
            dict(host='0.0.0.0', port=9010),
        ),
    )


async def _run_uvicorn(
    uvicorn: Any,
    app: Any,
    app_args: Dict[str, Any],
    db: Optional[str] = None,
) -> None:
    """Run uvicorn server and clean up functions after shutdown."""
    await uvicorn.Server(uvicorn.Config(app, **app_args)).serve()
    if db:
        logger.info('dropping functions from database')
        app.drop_functions(db)


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        pass
