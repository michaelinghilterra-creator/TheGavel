"""HTTP API intended for localhost use or deployment behind a proxy."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .judge import check_single_token_labels, judge
from .rubric import Rubric, rubric_from_dict


class JudgeItem(BaseModel):
    """One item submitted for judging."""

    id: str
    text: str


class JudgeRequest(BaseModel):
    """A batch judging request."""

    rubric: str | dict[str, Any]
    mode: Literal["lean", "explain"] = "lean"
    concurrency: int = Field(default=8, gt=0)
    items: list[JudgeItem]


async def _close_backend(backend: object) -> None:
    close = getattr(backend, "aclose", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await result


def create_app(
    backend_factory,
    rubrics: dict[str, Rubric] | None = None,
    *,
    max_items: int = 256,
    label_check: bool = True,
) -> FastAPI:
    """Create an API meant to sit on localhost or behind a trusted proxy."""

    loaded_rubrics = dict(rubrics or {})
    checked_rubrics: set[tuple[str, bytes]] = set()
    label_check_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        backend = backend_factory()
        app.state.backend = backend
        try:
            yield
        finally:
            await _close_backend(backend)

    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "rubrics": list(loaded_rubrics)}

    @app.post("/v1/judge")
    async def judge_batch(request: JudgeRequest) -> dict:
        if not request.items:
            raise HTTPException(status_code=400, detail="items must not be empty")
        if len(request.items) > max_items:
            raise HTTPException(
                status_code=400,
                detail=f"items must contain at most {max_items} entries",
            )

        item_ids = [item.id for item in request.items]
        if len(item_ids) != len(set(item_ids)):
            raise HTTPException(status_code=400, detail="duplicate item ids")

        if isinstance(request.rubric, str):
            try:
                rubric = loaded_rubrics[request.rubric]
            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown rubric: {request.rubric}",
                ) from exc
        else:
            if "preamble_file" in request.rubric:
                raise HTTPException(
                    status_code=400,
                    detail="inline rubrics cannot use preamble_file",
                )
            try:
                rubric = rubric_from_dict(request.rubric)
                if not isinstance(rubric.name, str) or not isinstance(
                    rubric.preamble, str
                ):
                    raise ValueError("rubric name and preamble must be strings")
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        if label_check:
            cache_key = (
                rubric.name,
                hashlib.sha256(rubric.preamble.encode("utf-8")).digest(),
            )
            if cache_key not in checked_rubrics:
                async with label_check_lock:
                    if cache_key not in checked_rubrics:
                        try:
                            await check_single_token_labels(app.state.backend, rubric)
                        except ValueError as exc:
                            raise HTTPException(status_code=400, detail=str(exc)) from exc
                        checked_rubrics.add(cache_key)

        results, stats = await judge(
            app.state.backend,
            rubric,
            [item.model_dump() for item in request.items],
            mode=request.mode,
            concurrency=request.concurrency,
        )
        return {"results": results, "stats": asdict(stats)}

    return app
