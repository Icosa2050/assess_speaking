"""Per-launch authorization for the installed loopback backend."""
from __future__ import annotations

import hmac
import re

from fastapi import Request
from starlette.responses import JSONResponse

TOKEN_ENV = "VOSTAVO_SESSION_TOKEN"
MEDIA_TOKEN_ENV = "VOSTAVO_MEDIA_TOKEN"
TOKEN_HEADER = "X-Vostavo-Session"
MEDIA_PATH = re.compile(r"^/v1/history/[^/]+/audio$")


def install_desktop_session(app, token: str, port: int, media_token: str = "") -> None:
    if not token:
        return

    @app.middleware("http")
    async def session_guard(request: Request, call_next):
        if request.headers.get("host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return JSONResponse(status_code=403, content={"detail": "Use the local Vostavo app."})
        if request.method != "OPTIONS":
            supplied = request.headers.get(TOKEN_HEADER, "")
            authorized = hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))
            if not supplied and media_token and request.method in {"GET", "HEAD"} and MEDIA_PATH.fullmatch(request.url.path):
                authorized = hmac.compare_digest(request.query_params.get("session", "").encode("utf-8"), media_token.encode("utf-8"))
            if not authorized:
                return JSONResponse(status_code=401, content={"detail": "Desktop session required."})
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
