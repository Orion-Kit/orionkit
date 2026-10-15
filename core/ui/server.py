"""Web UI: serves the orb page and streams assistant state (and, in companion mode, audio) over a WebSocket.

Local mode: bound to 127.0.0.1 with a fresh random token per launch.
Companion mode (--remote): still bound to 127.0.0.1, published to your devices through `tailscale serve`
(HTTPS, private to your tailnet), with a long-lived access code. In both modes the WebSocket only accepts
the page's own origin, so other websites open in a browser can't drive the assistant.
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
from urllib.parse import urlparse

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

log = logging.getLogger("orionkit.ui")
STATIC = Path(__file__).with_name("static")
FILES = {"/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"), "/sw.js": ("sw.js", "text/javascript"),
         "/icon.svg": ("icon.svg", "image/svg+xml"), "/icon-192.png": ("icon-192.png", "image/png"),
         "/icon-512.png": ("icon-512.png", "image/png"), "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png")}
if (STATIC / "_test_ask.wav").exists():  # end-to-end tests only; never shipped
    FILES["/_test_ask.wav"] = ("_test_ask.wav", "audio/wav")


class UIServer:
    def __init__(self, on_command: Callable[[dict[str, Any]], Awaitable[None]], port: int = 8770,
                 on_first_connect: Callable[[], None] | None = None, token: str | None = None,
                 on_audio: Callable[[bytes], None] | None = None) -> None:
        self.port = port
        self.token = token or secrets.token_urlsafe(24)
        self._on_audio = on_audio
        self._audio_clients: set[ServerConnection] = set()  # devices whose mic/speaker Orion is using
        self._on_command = on_command
        self._clients: set[ServerConnection] = set()
        self._snapshot: dict[str, Any] = {}  # latest status, replayed to new tabs
        self._server: Any = None
        self._on_first_connect = on_first_connect

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    async def start(self) -> None:
        self._server = await serve(self._handle, "127.0.0.1", self.port, process_request=self._http, max_size=256 * 1024)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def _http(self, connection: ServerConnection, request: Request) -> Response | None:
        path = request.path.split("?")[0]
        if path == "/ws":
            query = request.path.partition("?")[2]
            if not secrets.compare_digest(dict(p.partition("=")[::2] for p in query.split("&")).get("t", ""), self.token):
                return connection.respond(HTTPStatus.FORBIDDEN, "bad token\n")
            origin = request.headers.get("Origin", "")
            hosts = {request.headers.get("Host", ""), request.headers.get("X-Forwarded-Host", "")} - {""}
            if urlparse(origin).netloc not in hosts:  # same-origin only: the page we served, nobody else
                return connection.respond(HTTPStatus.FORBIDDEN, "cross-origin\n")
            return None  # proceed with the WebSocket handshake
        if path in FILES:
            name, ctype = FILES[path]
            body = (STATIC / name).read_bytes()
            resp = connection.respond(HTTPStatus.OK, "")
            resp.body = body
            for h in ("Content-Type", "Content-Length"):  # respond() set these for an empty text body
                del resp.headers[h]
            resp.headers["Content-Type"] = ctype
            resp.headers["Content-Length"] = str(len(body))
            resp.headers["Cache-Control"] = "no-cache"
            return resp
        if path in ("/", "/index.html"):
            resp = connection.respond(HTTPStatus.OK, (STATIC / "index.html").read_text())
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            resp.headers["Cache-Control"] = "no-store"
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
                "connect-src 'self' ws: wss:; img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'")
            return resp
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def _handle(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        if self._on_first_connect:
            callback, self._on_first_connect = self._on_first_connect, None
            callback()
        try:
            for msg in self._snapshot.values():
                await ws.send(json.dumps(msg))
            async for raw in ws:
                if isinstance(raw, bytes):  # microphone audio from a companion device
                    if ws in self._audio_clients and self._on_audio:
                        self._on_audio(raw)
                    continue
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(cmd, dict) and cmd.get("t") == "audio_hello":
                    self._audio_clients = {ws}  # the latest device to start audio owns the mic and speaker
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
            self._audio_clients.discard(ws)

    @property
    def audio_connected(self) -> bool:
        return bool(self._audio_clients)

    def send_audio(self, pcm16: bytes) -> None:
        for ws in list(self._audio_clients):
            asyncio.ensure_future(self._safe_send(ws, pcm16))

    def send(self, msg: dict[str, Any], keep: str | None = None) -> None:
        """Broadcast to every open tab. ``keep`` stores it as the latest of its kind for tabs that open later."""
        if keep:
            self._snapshot[keep] = msg
        if not self._clients:
            return
        data = json.dumps(msg, default=str)
        for ws in list(self._clients):
            asyncio.ensure_future(self._safe_send(ws, data))

    async def _safe_send(self, ws: ServerConnection, data: str | bytes) -> None:
        try:
            await ws.send(data)
        except ConnectionClosed:
            self._clients.discard(ws)
            self._audio_clients.discard(ws)
