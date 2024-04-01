#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import io
import sys
from typing import Any

try:
    import js
    import pyodide.ffi
except ImportError:
    raise ImportError('module only works in the Pyodide environment')


class WasmSocket(object):
    """Socket facade for Pyodide."""

    def __init__(self, **kwargs: Any) -> None:
        driver = kwargs.get('driver', 'wss')
        host = kwargs.get('host', 'localhost')
        port = int(kwargs.get('port', 443))

        self._uri = f'{driver}://{host}:{port}/proxy'

        self._debug = True
        self._jssocket = None
        self._incoming: asyncio.Queue[Any] = asyncio.Queue()
        self._errors: asyncio.Queue[Any] = asyncio.Queue()
        self._isopen: asyncio.Event = asyncio.Event()
        self._buffer = io.BytesIO()

    def connect(self) -> None:
        """Connect to the configured URI."""
        socket = js.WebSocket.new(self._uri)
        socket.binaryType = 'arraybuffer'
        socket.addEventListener(
            'open', pyodide.ffi.create_proxy(self._open_handler),
        )
        socket.addEventListener(
            'message', pyodide.ffi.create_proxy(self._message_handler),
        )
        socket.addEventListener(
            'error', pyodide.ffi.create_proxy(self._error_handler),
        )
        self._jssocket = socket
        asyncio.get_running_loop().run_until_complete(self._isopen.wait())

    def send(self, message: bytes) -> None:
        """Send bytes to the socket."""
        if self._jssocket is None:
            raise ValueError('socket not connected')
        if self._debug:
            js.console.log(f'[s2db send] {message!r}')
        data = pyodide.ffi.to_js(message)
        self._jssocket.send(data)

    def recv(self, n_bytes: int = -1) -> bytes:
        """
        Receive `n_bytes` bytes from the socket.

        Parameters
        ----------
        n_bytes : int, optional
            The number of bytes to receive. If the is negative, all
            of the available data will be returned.

        Returns
        -------
        bytes

        """
        if n_bytes is None or n_bytes < 0:
            n_bytes = sys.maxsize
        self._buffer.seek(0, 2)
        try:
            while self._buffer.tell() < n_bytes:
                self._buffer.write(self._incoming.get_nowait())
        except asyncio.QueueEmpty:
            pass
        self._buffer.seek(0, 0)
        out = self._buffer.read(n_bytes)
        self._buffer = io.BytesIO(self._buffer.read())
        return out

    def makefile(self, *args: Any) -> WasmSocket:
        """Return a file-like reader for the socket."""
        return self

    def settimeout(self, *args: Any) -> None:
        """Set a read timeout."""
        pass

    def read(self, n_bytes: int = -1) -> bytes:
        """Read `n_bytes` bytes from the socket."""
        return self.recv(n_bytes)

    def readall(self) -> bytes:
        """Read all bytes from the socket."""
        return self.recv(-1)

    def close(self) -> None:
        """Close the socket."""
        self._isopen.clear()
        self._buffer = io.BytesIO()
        self._incoming = asyncio.Queue()
        if self._jssocket is not None:
            self._jssocket.close()
            self._jssocket = None

    async def _open_handler(self, event: Any) -> None:
        """Event handler for socket-open events."""
        if self._debug:
            js.console.log(f'[s2db open] {event.to_py()}')
        self._isopen.set()

    async def _message_handler(self, event: Any) -> None:
        """Event handler for messages."""
        if self._debug:
            js.console.log(f'[s2db msg] {event.data.to_py()}')
        await self._incoming.put(event.data.to_bytes())

    async def _error_handler(self, event: Any) -> None:
        """Event handler for errors."""
        if self._debug:
            js.console.log(f'[s2db error] {vars(event.to_py())}')
        await self._errors.put(event.to_py())
