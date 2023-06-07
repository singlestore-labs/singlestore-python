#!/usr/bin/env python3
'''
Web application for SingleStoreDB external functions.

This module supplies a function that can create web apps intended for use
with the external function feature of SingleStoreDB. The application
function is a standard ASGI <https://asgi.readthedocs.io/en/latest/index.html>
request handler for use with servers such as Uvicorn <https://www.uvicorn.org>.

An external function web application can be created using the `create_app`
function. By default, the exported Python functions are specified by
environment variables starting with SINGLESTOREDB_EXT_FUNCTIONS. See the
documentation in `create_app` for the full syntax. If the application is
created in Python code rather than from the command-line, exported
functions can be specified in the parameters.

An example of starting a server is shown below.

Example
-------
$ SINGLESTOREDB_EXT_FUNCTIONS='myfuncs.[percentage_90,percentage_95]' \
    uvicorn --factory singlestoredb.ext_func:create_app

'''
import importlib
import itertools
import os
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from . import rowdat_1
from ..udf.signature import get_signature
from ..udf.signature import signature_to_sql

# If a number of processes is specified, create a pool of workers
num_processes = max(0, int(os.environ.get('SINGLESTOREDB_EXT_NUM_PROCESSES', 0)))
if num_processes > 1:
    try:
        from ray.util.multiprocessing import Pool
    except ImportError:
        from multiprocessing import Pool
    func_map = Pool(num_processes).starmap
else:
    func_map = itertools.starmap


def get_func_names(funcs: str) -> List[Tuple[str, str]]:
    '''
    Parse all function names from string.

    Parameters
    ----------
    func_names : str
        String containing one or more function names. The syntax is
        as follows: [func-name-1@func-alias-1,func-name-2@func-alias-2,...].
        The optional '@name' portion is an alias if you want the function
        to be renamed.

    Returns
    -------
    List[Tuple[str]] : a list of tuples containing the names and aliases
        of each function.

    '''
    if funcs.startswith('['):
        func_names = funcs.replace('[', '').replace(']', '').split(',')
        func_names = [x.strip() for x in func_names]
    else:
        func_names = [funcs]

    out = []
    for name in func_names:
        alias = name
        if '@' in name:
            name, alias = name.split('@', 1)
        out.append((name, alias))

    return out


def make_func(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    '''
    Make a function endpoint.

    Parameters
    ----------
    name : str
        Name of the function to create
    func : Callable
        The function to call as the endpoint

    Returns
    -------
    Callable

    '''
    async def do_func(row_ids: Sequence[int], rows: Sequence[Any]) -> List[Any]:
        '''Call function on given rows of data.'''
        return list(zip(row_ids, func_map(func, rows)))

    do_func.__name__ = name
    do_func.__doc__ = func.__doc__
    sig = get_signature(name, func)
    do_func._ext_func_signature = sig  # type: ignore
    do_func._ext_func_colspec = [(x['name'], x['type'])  # type: ignore
                                 for x in sig['args']]
    do_func._ext_func_returns = [x['type'] for x in sig['returns']]  # type: ignore

    return do_func


def create_app(
    functions: Optional[
        Union[
            str,
            Iterable[str],
            Callable[..., Any],
            Iterable[Callable[..., Any]],
        ]
    ] = None,


) -> Callable[..., Any]:
    '''
    Create an external function application.

    If `functions` is None, the environment is searched for function
    specifications in variables starting with `SINGLESTOREDB_EXT_FUNCTIONS`.
    Any number of environment variables can be specified as long as they
    have this prefix. The format of the environment variable value is the
    same as for the `functions` parameter.

    Parameters
    ----------
    functions : str or Iterable[str], optional
        Python functions are specified using a string format as follows:
            * Single function : <pkg1>.<func1>
            * Multiple functions : <pkg1>.[<func1-name,func2-name,...]
            * Function aliases : <pkg1>.[<func1@alias1,func2@alias2,...]
            * Multiple packages : <pkg1>.<func1>:<pkg2>.<func2>

    Returns
    -------
    Callable : the application request handler

    '''

    # List of functions specs
    specs: List[Union[str, Callable[..., Any]]] = []

    # Look up Python function specifications
    if functions is None:
        for k, v in os.environ.items():
            if k.startswith('SINGLESTOREDB_EXT_FUNCTIONS'):
                specs.append(v)
    elif isinstance(functions, str):
        specs = [functions]
    elif callable(functions):
        specs = [functions]
    else:
        specs = list(functions)

    # Add functions to application
    endpoints = dict()
    for funcs in itertools.chain(specs):
        if isinstance(funcs, str):
            pkg_path, func_names = funcs.rsplit('.', 1)
            pkg = importlib.import_module(pkg_path)

            # Add endpoint for each exported function
            for name, alias in get_func_names(func_names):
                item = getattr(pkg, name)
                func = make_func(alias, item)
                endpoints[f'/functions/{alias}'] = func
        else:
            alias = funcs.__name__
            func = make_func(alias, item)
            endpoints[f'/functions/{alias}'] = func

    # Plain text response start
    text_response_dict: Dict[str, Any] = dict(
        type='http.response.start',
        status=200,
        headers=[(b'content-type', b'text/plain')],
    )

    # JSON response start
    # json_response_dict: Dict[str, Any] = dict(
    #     type='http.response.start',
    #     status=200,
    #     headers=[(b'content-type', b'application/json')],
    # )

    # ROWDAT_1 response start
    rowdat_1_response_dict: Dict[str, Any] = dict(
        type='http.response.start',
        status=200,
        headers=[(b'content-type', b'x-application/rowdat_1')],
    )

    # Path not found response start
    path_not_found_response_dict: Dict[str, Any] = dict(
        type='http.response.start',
        status=404,
    )

    # Response body template
    body_response_dict: Dict[str, Any] = dict(
        type='http.response.body',
    )

    async def app(
        scope: Dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[Any]],
    ) -> None:
        '''
        Application request handler.

        Parameters
        ----------
        scope : dict
            ASGI request scope
        receive : Callable
            Function to receieve request information
        send : Callable
            Function to send response information

        '''
        assert scope['type'] == 'http'

        method = scope['method']
        path = scope['path']

        func = endpoints.get(path)

        # Call the endpoint
        if method == 'POST' and func is not None:
            data = []
            more_body = True
            while more_body:
                request = await receive()
                data.append(request['body'])
                more_body = request.get('more_body', False)

            out = await func(
                *rowdat_1.load(
                    func._ext_func_colspec, b''.join(data),  # type: ignore
                ),
            )
            body = rowdat_1.dump(func._ext_func_returns, out)  # type: ignore

            await send(rowdat_1_response_dict)

        # Handle api reflection
        elif method == 'GET' and path in ['/functions', '/functions/']:
            host = 'localhost:80'
            for k, v in scope['headers']:
                if k == b'host':
                    host = v.decode('utf-8')
                    break

            url = f'{scope["scheme"]}://{host}{path}'

            syntax = []
            for endpoint in endpoints.values():
                syntax.append(
                    signature_to_sql(
                        endpoint._ext_func_signature,  # type: ignore
                        base_url=url,
                    ),
                )
            body = '\n'.join(syntax).encode('utf-8')

            await send(text_response_dict)

        # Path not found
        else:
            body = b''
            await send(path_not_found_response_dict)

        # Send body
        out = body_response_dict.copy()
        out['body'] = body
        await send(out)

    return app
