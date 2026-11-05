"""Quote, approval, spend cap, creation, polling and failure handling of the paid tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
import respx
from langchain_core.messages import ToolMessage

from langchain_popnow import (
    PopNowCreateTask,
    PopNowImageGeneration,
    PopNowMusicGeneration,
    PopNowSpeech,
    PopNowVideoGeneration,
)
from langchain_popnow.tools import (
    DEFAULT_IMAGE_EDIT_MODEL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    DEFAULT_MUSIC_MODEL,
    DEFAULT_SPEECH_MODEL,
    DEFAULT_VIDEO_MODEL,
)

from .conftest import API_KEY, created, envelope, quote, task_record

QUOTE_PATH = "/api/v1/jobs/quote"
CREATE_PATH = "/api/v1/jobs/createTask"
RECORD_PATH = "/api/v1/jobs/recordInfo"


def call(tool: Any, args: Dict[str, Any]) -> ToolMessage:
    """Invoke like an agent does: with a ToolCall, returning a ToolMessage."""
    return tool.invoke({"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"})


def body(route: respx.Route, index: int = -1) -> Dict[str, Any]:
    return json.loads(route.calls[index].request.content)


def test_first_call_only_quotes_and_charges_nothing(api: respx.MockRouter) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))
    tool = PopNowImageGeneration()

    msg = call(tool, {"prompt": "A paper boat", "aspect_ratio": "16:9"})

    assert q.call_count == 1 and c.call_count == 0
    request = q.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert body(q) == {"model": DEFAULT_IMAGE_MODEL, "input": {"prompt": "A paper boat", "aspect_ratio": "16:9"}}
    content = json.loads(msg.content)
    assert content["status"] == "needs_confirmation"
    assert content["cost_usd"] == "0.048"
    assert content["currency"] == "USD"
    assert 'max_usd="0.048"' in content["note"]
    assert msg.artifact["type"] == "quote" and msg.artifact["quoteId"] == "q_1"


def test_approved_call_creates_with_locked_quote_and_polls(api: respx.MockRouter, sleeps: List[float]) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))
    r = api.get(RECORD_PATH).mock(
        side_effect=[
            httpx.Response(200, json=envelope(task_record("running"))),
            httpx.Response(200, json=envelope(task_record("succeeded"))),
        ]
    )
    tool = PopNowImageGeneration(poll_interval=1.0)

    msg = call(tool, {"prompt": "A paper boat", "max_usd": "0.05"})

    assert c.call_count == 1
    request = c.calls.last.request
    assert body(c) == {
        "model": DEFAULT_IMAGE_MODEL,
        "input": {"prompt": "A paper boat"},
        "quoteId": "q_1",
        "expectedCostUsd": "0.048",
    }
    assert request.headers["Idempotency-Key"]
    assert request.url.params["wait"] == "30"
    assert r.call_count == 2 and r.calls.last.request.url.params["taskId"] == "job_1"
    assert sleeps == [1.0]
    content = json.loads(msg.content)
    assert content["status"] == "succeeded"
    assert content["task_id"] == "job_1"
    assert content["cost_usd"] == "0.048"
    assert content["outputs"][0]["url"] == "https://media.example.com/signed/out_0.png"
    assert content["outputs"][0]["uri"] == "popnow://g/job_1/0"
    assert msg.artifact["type"] == "task" and msg.artifact["state"] == "succeeded"


def test_server_side_wait_result_skips_polling(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    api.post(CREATE_PATH).respond(200, json=envelope(task_record("succeeded")))
    r = api.get(RECORD_PATH).respond(200, json=envelope(task_record("succeeded")))

    msg = call(PopNowImageGeneration(), {"prompt": "A paper boat", "max_usd": "0.048"})

    assert json.loads(msg.content)["status"] == "succeeded"
    assert r.call_count == 0


def test_numeric_max_usd_is_accepted(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    c = api.post(CREATE_PATH).respond(200, json=envelope(task_record("succeeded")))
    call(PopNowImageGeneration(), {"prompt": "A paper boat", "max_usd": 0.05})
    assert c.call_count == 1


def test_price_above_approval_is_not_charged(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_VIDEO_MODEL, "0.37")))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))

    msg = call(PopNowVideoGeneration(), {"prompt": "Coastal drive", "max_usd": "0.185"})

    assert c.call_count == 0
    content = json.loads(msg.content)
    assert content["status"] == "needs_confirmation"
    assert content["cost_usd"] == "0.37"
    assert content["approved_usd"] == "0.185"


def test_developer_ceiling_overrides_agent_approval(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_VIDEO_MODEL, "2.32")))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))

    msg = call(PopNowVideoGeneration(max_usd_per_call="1"), {"prompt": "Epic", "max_usd": "100"})

    assert c.call_count == 0
    content = json.loads(msg.content)
    assert content["status"] == "over_limit"
    assert content["max_usd_per_call"] == "1"


def test_auto_approve_runs_small_quotes_without_max_usd(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_SPEECH_MODEL, "0.0016")))
    c = api.post(CREATE_PATH).respond(200, json=envelope(task_record("succeeded", model=DEFAULT_SPEECH_MODEL)))

    msg = call(PopNowSpeech(auto_approve_usd="0.01"), {"text": "Hello"})

    assert c.call_count == 1
    assert body(c)["expectedCostUsd"] == "0.0016"
    assert json.loads(msg.content)["status"] == "succeeded"


def test_auto_approve_does_not_cover_larger_quotes(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_SPEECH_MODEL, "0.02")))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))
    msg = call(PopNowSpeech(auto_approve_usd="0.01"), {"text": "Hello"})
    assert c.call_count == 0
    assert json.loads(msg.content)["status"] == "needs_confirmation"


def test_auto_approve_above_ceiling_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto_approve_usd"):
        PopNowSpeech(auto_approve_usd="2", max_usd_per_call="1")


@pytest.mark.parametrize("bad", ["abc", "-1", "1e3", "0.0000000001"])
def test_invalid_max_usd_is_reported_to_the_agent(api: respx.MockRouter, bad: str) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    msg = call(PopNowImageGeneration(), {"prompt": "x", "max_usd": bad})
    assert q.call_count == 0
    assert msg.status == "error"
    assert "max_usd" in msg.content and "nine decimal places" in msg.content


def test_failed_task_is_reported(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_MUSIC_MODEL, "0.072")))
    api.post(CREATE_PATH).respond(202, json=envelope(created()))
    failed = task_record("failed", model=DEFAULT_MUSIC_MODEL, errorCode="CONTENT_REJECTED", errorMessage="Rejected")
    api.get(RECORD_PATH).respond(200, json=envelope(failed))

    msg = call(PopNowMusicGeneration(), {"prompt": "A song", "max_usd": "0.072"})

    content = json.loads(msg.content)
    assert content["status"] == "failed"
    assert content["error_code"] == "CONTENT_REJECTED"
    assert "outputs" not in content


def test_api_error_becomes_a_readable_tool_message(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    api.post(CREATE_PATH).respond(402, json=envelope(code=40201, msg="Insufficient API balance"))

    msg = call(PopNowImageGeneration(max_retries=0), {"prompt": "x", "max_usd": "1"})

    assert msg.status == "error"
    assert "Insufficient API balance" in msg.content
    assert "popnow.ai/api/console/billing" in msg.content


def test_bad_request_on_quote_is_not_charged(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(400, json=envelope(code=400, msg="input.foo is not a field of this model"))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))
    msg = call(PopNowCreateTask(max_retries=0), {"model": "m/x/text-to-image", "input": {"foo": 1}, "max_usd": "1"})
    assert c.call_count == 0
    assert "input.foo" in msg.content


def test_wait_timeout_returns_running_task_id(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_VIDEO_MODEL, "0.185")))
    api.post(CREATE_PATH).respond(202, json=envelope(created(estimated="0.185")))
    api.get(RECORD_PATH).respond(200, json=envelope(task_record("running", model=DEFAULT_VIDEO_MODEL)))

    msg = call(PopNowVideoGeneration(wait_timeout=0), {"prompt": "Clip", "max_usd": "0.185"})

    content = json.loads(msg.content)
    assert content["status"] == "running"
    assert content["task_id"] == "job_1"
    assert "popnow_get_task" in content["note"]


def test_image_urls_switch_to_edit_model(api: respx.MockRouter) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_EDIT_MODEL)))
    call(PopNowImageGeneration(), {"prompt": "Make it night", "image_urls": ["https://example.com/a.png"]})
    assert body(q)["model"] == DEFAULT_IMAGE_EDIT_MODEL
    assert body(q)["input"]["image_urls"] == ["https://example.com/a.png"]


def test_image_url_switches_to_image_to_video_and_drops_aspect_ratio(api: respx.MockRouter) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_TO_VIDEO_MODEL)))
    call(
        PopNowVideoGeneration(),
        {"prompt": "Waves", "image_url": "popnow://g/job_0/0", "aspect_ratio": "9:16", "duration_seconds": 5},
    )
    assert body(q) == {
        "model": DEFAULT_IMAGE_TO_VIDEO_MODEL,
        "input": {"prompt": "Waves", "image_url": "popnow://g/job_0/0", "duration_seconds": 5},
    }


def test_model_override_and_extra_input(api: respx.MockRouter) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote("kling/3.0/text-to-video")))
    call(
        PopNowVideoGeneration(),
        {"prompt": "Koi pond", "model": "kling/3.0/text-to-video", "extra_input": {"seed": 7, "prompt": "ignored"}},
    )
    assert body(q) == {"model": "kling/3.0/text-to-video", "input": {"seed": 7, "prompt": "Koi pond"}}


def test_model_override_can_be_disabled(api: respx.MockRouter) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    msg = call(PopNowImageGeneration(allow_model_override=False), {"prompt": "x", "model": "openai/gpt-image-2/text-to-image"})
    assert q.call_count == 0
    assert DEFAULT_IMAGE_MODEL in msg.content


def test_music_and_speech_request_shapes(api: respx.MockRouter) -> None:
    q = api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_MUSIC_MODEL)))
    call(PopNowMusicGeneration(), {"prompt": "Lo-fi", "instrumental": True, "title": "Steam"})
    assert body(q) == {"model": DEFAULT_MUSIC_MODEL, "input": {"prompt": "Lo-fi", "instrumental": True, "title": "Steam"}}
    call(PopNowSpeech(), {"text": "Hi there", "voice_id": "Puck", "style": "newscaster"})
    assert body(q) == {
        "model": DEFAULT_SPEECH_MODEL,
        "input": {"text": "Hi there", "voice_id": "Puck", "style": "newscaster"},
    }


async def test_async_approved_flow(api: respx.MockRouter, sleeps: List[float]) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))
    api.get(RECORD_PATH).mock(
        side_effect=[
            httpx.Response(200, json=envelope(task_record("queued"))),
            httpx.Response(200, json=envelope(task_record("succeeded"))),
        ]
    )
    tool = PopNowImageGeneration()

    msg = await tool.ainvoke({"name": tool.name, "args": {"prompt": "A boat", "max_usd": "0.048"}, "id": "1", "type": "tool_call"})

    assert c.call_count == 1 and body(c)["quoteId"] == "q_1"
    assert json.loads(msg.content)["outputs"][0]["uri"] == "popnow://g/job_1/0"
    assert sleeps == [2.0]


async def test_async_first_call_only_quotes(api: respx.MockRouter) -> None:
    api.post(QUOTE_PATH).respond(200, json=envelope(quote(DEFAULT_MUSIC_MODEL, "0.072")))
    c = api.post(CREATE_PATH).respond(202, json=envelope(created()))
    tool = PopNowMusicGeneration()
    msg = await tool.ainvoke({"name": tool.name, "args": {"prompt": "Jazz"}, "id": "1", "type": "tool_call"})
    assert c.call_count == 0
    assert json.loads(msg.content)["status"] == "needs_confirmation"
