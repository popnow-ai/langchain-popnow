"""Shared fixtures: a mocked PopNow API (no network, no charges)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import pytest
import respx
from popnow import _base_client

BASE = "https://api.popnow.ai"
API_KEY = "sk-popnow-" + "0" * 48


def envelope(data: Any = None, code: int = 200, msg: str = "success") -> Dict[str, Any]:
    body: Dict[str, Any] = {"code": code, "msg": msg, "request_id": "req_test"}
    if data is not None:
        body["data"] = data
    return body


def quote(model: str, max_charge: str = "0.048", quote_id: str = "q_1") -> Dict[str, Any]:
    return {
        "quoteId": quote_id,
        "model": model,
        "maxChargeUsd": max_charge,
        "expectedChargeUsd": max_charge,
        "currency": "USD",
        "quantity": "1",
        "unit": "per_image",
        "expiresAt": "2026-11-05T09:25:00Z",
    }


def created(task_id: str = "job_1", estimated: str = "0.048") -> Dict[str, Any]:
    return {"taskId": task_id, "state": "queued", "estimatedCostUsd": estimated, "deadlineAt": "2026-11-05T09:40:00Z"}


def task_record(
    state: str = "succeeded",
    task_id: str = "job_1",
    model: str = "google/nano-banana-2.1/text-to-image",
    **extra: Any,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "taskId": task_id,
        "model": model,
        "state": state,
        "costUsd": "0.048",
        "settled": state not in ("queued", "running"),
        "createdAt": "2026-11-05T09:00:00Z",
    }
    if state == "succeeded":
        record["output"] = {
            "assets": [
                {
                    "key": "out_0",
                    "url": "https://media.example.com/signed/out_0.png",
                    "mime": "image/png",
                    "width": 1024,
                    "height": 1024,
                    "expiresAt": "2026-11-05T10:00:00Z",
                }
            ]
        }
    record.update(extra)
    return record


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep proxies and real credentials out of the tests."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("POPNOW_BASE_URL", raising=False)
    monkeypatch.setenv("POPNOW_API_KEY", API_KEY)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Record polling back-off sleeps instead of sleeping."""
    calls: List[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    async def fake_async_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(_base_client, "_sleep", fake_sleep)
    monkeypatch.setattr(_base_client, "_async_sleep", fake_async_sleep)
    return calls


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router
