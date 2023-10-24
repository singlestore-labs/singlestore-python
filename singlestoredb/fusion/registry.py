#!/usr/bin/env python3
import os
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

from . import result
from .. import connection
from .handler import SQLHandler

_enabled = ('1', 'yes', 'on', 'enabled', 'true')
_handlers: Dict[str, Type[SQLHandler]] = {}
_handlers_re: Optional[Any] = None


def register_handler(handler: Type[SQLHandler], overwrite: bool = False) -> None:
    """
    Register a new SQL handler.

    Parameters
    ----------
    handler : SQLHandler subclass
        The handler class to register
    overwrite : bool, optional
        Should an existing handler be overwritten if it uses the same command key?

    """
    global _handlers
    global _handlers_re

    # Build key for handler
    key = ' '.join(x.upper() for x in handler.command_key)

    # Check for existing handler with same key
    if not overwrite and key in _handlers:
        raise ValueError(f'command already exists, use overwrite=True to override: {key}')

    # Add handler to registry
    _handlers[key] = handler

    # Build regex to detect fusion query
    keys = sorted(_handlers.keys(), key=lambda x: -len(x[0]))
    keys_str = '|'.join(x.replace(' ', '\\s+') for x in keys)
    _handlers_re = re.compile(f'^\\s*({keys_str})(?:\\s+|;|$)', flags=re.I)


def get_handler(sql: Union[str, bytes]) -> Optional[Type[SQLHandler]]:
    """
    Return a fusion handler for the given query.

    Parameters
    ----------
    sql : str or bytes
        The SQL query

    Returns
    -------
    SQLHandler - if a matching one exists
    None - if no matching handler could be found

    """
    if not os.environ.get('SINGLESTOREDB_ENABLE_FUSION', '').lower() in _enabled:
        return None

    if isinstance(sql, (bytes, bytearray)):
        sql = sql.decode('utf-8')

    if _handlers_re is None:
        return None

    m = _handlers_re.match(sql)
    if m:
        return _handlers[re.sub(r'\s+', r' ', m.group(1).strip().upper())]

    return None


def execute(
    connection: connection.Connection,
    sql: str,
    handler: Optional[Type[SQLHandler]] = None,
) -> result.FusionSQLResult:
    """
    Execute a SQL query in the management interface.

    Parameters
    ----------
    connection : Connection
        The SingleStoreDB connection object
    sql : str
        The SQL query
    handler : SQLHandler, optional
        The handler to use for the commands. If not supplied, one will be
        looked up in the registry.

    Returns
    -------
    FusionSQLResult

    """
    if not os.environ.get('SINGLESTOREDB_ENABLE_FUSION', '').lower() in _enabled:
        raise RuntimeError('management API queries have not been enabled')

    if handler is None:
        handler = get_handler(sql)
        if handler is None:
            raise RuntimeError(f'could not find handler for query: {sql}')

    return handler(connection).execute(sql)
