"""Local web UI: serves the orb page and streams assistant state over a WebSocket.

Bound to 127.0.0.1 only. Every launch gets a random token that the page must present, and the WebSocket
only accepts the page's own origin, so other websites open in the browser can't drive the assistant.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

log = logging.getLogger("orionkit.ui")
STATIC = Path(__file__).with_name("static")


class UIServer:
    def __init__(self, on_command: Callable[[dict[str, Any]], Awaitable[None]], port: int = 8770) -> None:
        self.port = port
        self.token = secrets.token_urlsafe(24)
        self._on_command = on_command
        self._clients: set[ServerConnection] = set()
        self._snapshot: dict[str, Any] = {}  # latest status, replayed to new tabs
        self._server: Any = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    async def start(self) -> None:
        origins = [f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"]
        self._server = await serve(self._handle, "127.0.0.1", self.port, origins=origins,
                                   process_request=self._http, max_size=64 * 1024)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def _http(self, connection: ServerConnection, request: Request) -> Response | None:
        path = request.path.split("?")[0]
        if path == "/ws":
            query = request.path.partition("?")[2]
            if f"t={self.token}" not in query.split("&"):
                return connection.respond(HTTPStatus.FORBIDDEN, "bad token\n")
            return None  # proceed with the WebSocket handshake
        if path in ("/", "/index.html"):
            resp = connection.respond(HTTPStatus.OK, (STATIC / "index.html").read_text())
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            resp.headers["Cache-Control"] = "no-store"
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                f"connect-src ws://127.0.0.1:{self.port} ws://localhost:{self.port}; img-src 'self' data:; frame-ancestors 'none'")
            return resp
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def _handle(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        try:
            for msg in self._snapshot.values():
                await ws.send(json.dumps(msg))
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(cmd, dict) and isinstance(cmd.get("t"), str):
                    try:
                        await self._on_command(cmd)
                    except Exception:
                        log.exception("UI command failed: %s", cmd.get("t"))
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    def send(self, msg: dict[str, Any], keep: str | None = None) -> None:
        """Broadcast to every open tab. ``keep`` stores it as the latest of its kind for tabs that open later."""
        if keep:
            self._snapshot[keep] = msg
        if not self._clients:
            return
        data = json.dumps(msg, default=str)
        for ws in list(self._clients):
            asyncio.ensure_future(self._safe_send(ws, data))

    async def _safe_send(self, ws: ServerConnection, data: str) -> None:
        try:
            await ws.send(data)
        except ConnectionClosed:
            self._clients.discard(ws)
