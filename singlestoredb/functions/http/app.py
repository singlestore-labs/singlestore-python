#!/usr/bin/env python
import asyncio
import functools
import inspect
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from makefun import with_signature

try:
    import fastapi
except ImportError:
    raise ImportError('the fastapi package is required to use this module')

from ..utils import get_logger


logger = get_logger('singlestoredb.functions.http.app')


def export_fastapi(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a FastAPI method into a standalone decorator."""

    name = func.__name__

    # Wrapper that acts like original function
    def meth(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        attrs = dict(method=name, args=tuple(args), kwargs=kwargs)

        # Add original function parameters to a decorated function
        def decorate(func: Callable[..., Any]) -> Callable[..., Any]:

            # Return the final callable
            def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
                return func(*args, **kwargs)  # type: ignore

            wrapper._singlestoredb_http_attrs = attrs  # type: ignore
            return functools.wraps(func)(wrapper)

        return decorate

    # Remove `self` parameter from signature
    sig = inspect.signature(func)
    sig = sig.replace(parameters=tuple(sig.parameters.values())[1:])

    return with_signature(sig, func_name=name)(meth)


get = export_fastapi(fastapi.FastAPI.get)
put = export_fastapi(fastapi.FastAPI.put)
post = export_fastapi(fastapi.FastAPI.post)
delete = export_fastapi(fastapi.FastAPI.delete)
patch = export_fastapi(fastapi.FastAPI.patch)
options = export_fastapi(fastapi.FastAPI.options)
head = export_fastapi(fastapi.FastAPI.head)


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create a FastAPI application with exported endpoints."""
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

    return app


__all__ = ['get', 'put', 'post', 'patch', 'delete', 'options', 'head', 'create_app']


def main(argv: Optional[List[str]] = None) -> None:
    """Run the HTTP server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError('the uvicorn package is required to run this command')

    asyncio.create_task(
        _run_uvicorn(
            uvicorn,
            create_app(),
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
