"""Model catalog, task lookup, balance, toolkit and configuration."""

from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import pytest
import respx

from langchain_popnow import (
    PopNowCheckBalance,
    PopNowGetTask,
    PopNowImageGeneration,
    PopNowListModels,
    PopNowToolkit,
    __version__,
)

from .conftest import API_KEY, envelope, task_record

MODEL: Dict[str, Any] = {
    "model": "kling/3.0/text-to-video",
    "displayName": "Kling 3.0",
    "provider": "kling",
    "modality": "video",
    "tasks": ["text-to-video"],
    "available": True,
    "usd": {"unit": "per_second", "perUnit": "0.084", "currency": "USD"},
    "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
}


def test_list_models_filters_and_summarizes(api: respx.MockRouter) -> None:
    route = api.get("/api/v1/models").respond(200, json=envelope({"total": 2, "items": [MODEL, MODEL]}))
    out = PopNowListModels().invoke({"modality": "video", "search": "kling", "limit": 1})
    params = route.calls.last.request.url.params
    assert params["modality"] == "video" and params["search"] == "kling"
    data = json.loads(out)
    assert data["total"] == 2 and data["returned"] == 1
    assert data["models"][0] == {
        "model": "kling/3.0/text-to-video",
        "name": "Kling 3.0",
        "modality": "video",
        "tasks": ["text-to-video"],
        "usd": {"unit": "per_second", "perUnit": "0.084", "currency": "USD"},
        "available": True,
    }


def test_list_models_detail_includes_input_schema(api: respx.MockRouter) -> None:
    api.get("/api/v1/models/kling/3.0/text-to-video").respond(200, json=envelope(MODEL))
    data = json.loads(PopNowListModels().invoke({"model": "kling/3.0/text-to-video"}))
    assert data["input_schema"]["required"] == ["prompt"]


async def test_list_models_async(api: respx.MockRouter) -> None:
    api.get("/api/v1/models").respond(200, json=envelope({"total": 1, "items": [MODEL]}))
    data = json.loads(await PopNowListModels().ainvoke({}))
    assert data["models"][0]["model"] == "kling/3.0/text-to-video"


def test_get_task_waits_until_done(api: respx.MockRouter) -> None:
    api.get("/api/v1/jobs/recordInfo").mock(
        side_effect=[
            httpx.Response(200, json=envelope(task_record("running"))),
            httpx.Response(200, json=envelope(task_record("succeeded"))),
        ]
    )
    tool = PopNowGetTask()
    msg = tool.invoke({"name": tool.name, "args": {"task_id": "job_1"}, "id": "1", "type": "tool_call"})
    assert json.loads(msg.content)["status"] == "succeeded"
    assert msg.artifact["taskId"] == "job_1"


def test_get_task_returns_running_when_wait_is_zero(api: respx.MockRouter) -> None:
    api.get("/api/v1/jobs/recordInfo").respond(200, json=envelope(task_record("running")))
    content, artifact = PopNowGetTask()._run(task_id="job_1", wait_seconds=0)
    assert json.loads(content)["status"] == "running"


async def test_get_task_async(api: respx.MockRouter) -> None:
    api.get("/api/v1/jobs/recordInfo").respond(200, json=envelope(task_record("succeeded")))
    tool = PopNowGetTask()
    msg = await tool.ainvoke({"name": tool.name, "args": {"task_id": "job_1"}, "id": "1", "type": "tool_call"})
    assert json.loads(msg.content)["outputs"][0]["url"].endswith("out_0.png")


def test_get_task_not_found_is_readable(api: respx.MockRouter) -> None:
    api.get("/api/v1/jobs/recordInfo").respond(404, json=envelope(code=404, msg="Task not found"))
    tool = PopNowGetTask(max_retries=0)
    msg = tool.invoke({"name": tool.name, "args": {"task_id": "nope"}, "id": "1", "type": "tool_call"})
    assert msg.status == "error" and "Task not found" in msg.content


def test_balance(api: respx.MockRouter) -> None:
    api.get("/api/v1/chat/credit").respond(
        200, json=envelope({"availableUsd": "12.4", "heldUsd": "0.37", "totalUsd": "12.77"})
    )
    assert json.loads(PopNowCheckBalance().invoke({})) == {
        "currency": "USD",
        "available_usd": "12.4",
        "held_usd": "0.37",
        "total_usd": "12.77",
    }


async def test_balance_async(api: respx.MockRouter) -> None:
    api.get("/api/v1/chat/credit").respond(200, json=envelope({"availableUsd": "1", "heldUsd": "0", "totalUsd": "1"}))
    assert json.loads(await PopNowCheckBalance().ainvoke({}))["available_usd"] == "1"


def test_integration_header_and_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    with respx.mock(base_url="https://api.example.test") as mock:
        route = mock.get("/api/v1/chat/credit").respond(200, json=envelope({"availableUsd": "1"}))
        PopNowCheckBalance(base_url="https://api.example.test").invoke({})
    assert route.calls.last.request.headers["X-PopNow-Integration"] == f"langchain-popnow/{__version__}"


def test_missing_api_key_points_to_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPNOW_API_KEY")
    with pytest.raises(ValueError, match="popnow.ai/api/console/keys"):
        PopNowImageGeneration()


def test_api_key_is_secret() -> None:
    tool = PopNowImageGeneration(api_key="sk-popnow-secret")
    assert "sk-popnow-secret" not in repr(tool)
    assert tool.api_key.get_secret_value() == "sk-popnow-secret"


def test_toolkit_shares_settings() -> None:
    toolkit = PopNowToolkit(api_key="sk-popnow-k", max_usd_per_call=1, auto_approve_usd="0.05", wait_timeout=120)
    tools = toolkit.get_tools()
    names = [t.name for t in tools]
    assert names == [
        "popnow_image_generation",
        "popnow_video_generation",
        "popnow_music_generation",
        "popnow_text_to_speech",
        "popnow_create_task",
        "popnow_list_models",
        "popnow_get_task",
        "popnow_check_balance",
    ]
    for tool in tools:
        assert tool.api_key.get_secret_value() == "sk-popnow-k"  # type: ignore[attr-defined]
    image = tools[0]
    assert image.max_usd_per_call == "1" and image.auto_approve_usd == "0.05" and image.wait_timeout == 120


def test_toolkit_optional_tools() -> None:
    names = [t.name for t in PopNowToolkit(include_create_task=False, include_balance=False).get_tools()]
    assert "popnow_create_task" not in names and "popnow_check_balance" not in names


def test_injected_client_is_used(api: respx.MockRouter) -> None:
    import popnow

    api.get("/api/v1/chat/credit").respond(200, json=envelope({"availableUsd": "3"}))
    client = popnow.PopNow(api_key=API_KEY)
    tool = PopNowCheckBalance(api_key="ignored", client=client)
    assert json.loads(tool.invoke({}))["available_usd"] == "3"
    assert api.calls.last.request.headers["Authorization"] == f"Bearer {API_KEY}"
