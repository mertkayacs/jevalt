"""HTTP server with TypeSafe's `/v1/systemone` contract.

Point any TypeSafe client at it by changing the base URL. Extensions are
optional request fields: `reasoning` ("off" | "on" | "auto"), `abstain`, `language`.
"""

import os
import secrets
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .engine import DecisionEngine


def create_app(engine: DecisionEngine, *, api_key: str | None = None, release_date: str = "", cors: bool = True):
    app = FastAPI(title="JevAlt", version="0.1.0", docs_url="/docs", redoc_url=None)
    lock = threading.Lock()  # one decode at a time: backends are not thread safe
    key = api_key or os.environ.get("JEVALT_API_KEY")

    if cors:
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

        @app.middleware("http")
        async def private_network(request: Request, call_next):
            # Chrome asks before a public page may call a server on localhost.
            response = await call_next(request)
            if request.headers.get("access-control-request-private-network") == "true":
                response.headers["Access-Control-Allow-Private-Network"] = "true"
            return response

    def check(request: Request):
        if key:
            sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not secrets.compare_digest(sent, key):
                raise HTTPException(401, "missing or invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model, "backend": engine.backend.name}

    @app.get("/v1/models")
    def models(request: Request):
        check(request)
        return {"models": [{"name": engine.model, "description": "JevAlt decision model (local)", "release_date": release_date}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        check(request)
        try:
            body: dict[str, Any] = await request.json()
        except ValueError:
            raise HTTPException(422, "body must be JSON") from None
        if "state" not in body or not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise HTTPException(422, "request needs `state` and a non-empty `questions` object")
        try:
            with lock:
                return JSONResponse(engine.predict(body))
        except (ValueError, KeyError, TypeError) as err:
            raise HTTPException(422, str(err)) from None

    return app
