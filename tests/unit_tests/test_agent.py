"""The tools inside a LangGraph agent, driven by a scripted chat model (no LLM, no network)."""

from __future__ import annotations

import json
from typing import Any, Iterator, List, Optional, Sequence

import pytest
import respx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_popnow import PopNowToolkit
from langchain_popnow.tools import DEFAULT_IMAGE_MODEL

from .conftest import created, envelope, quote, task_record


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted messages in order and accepts any tools."""

    script: List[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.script[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _script() -> List[AIMessage]:
    args = {"prompt": "A red fox in a teacup"}
    return [
        AIMessage(content="", tool_calls=[{"name": "popnow_image_generation", "args": args, "id": "c1"}]),
        AIMessage(
            content="",
            tool_calls=[{"name": "popnow_image_generation", "args": {**args, "max_usd": "0.048"}, "id": "c2"}],
        ),
        AIMessage(content="Done."),
    ]


def _tool_messages(messages: Iterator[BaseMessage]) -> List[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _mock_api(api: respx.MockRouter) -> respx.Route:
    api.post("/api/v1/jobs/quote").respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    api.get("/api/v1/jobs/recordInfo").respond(200, json=envelope(task_record("succeeded")))
    return api.post("/api/v1/jobs/createTask").respond(202, json=envelope(created()))


def test_create_agent_quotes_then_runs(api: respx.MockRouter) -> None:
    agents = pytest.importorskip("langchain.agents")
    create = _mock_api(api)

    agent = agents.create_agent(model=ScriptedChatModel(script=_script()), tools=PopNowToolkit().get_tools())
    result = agent.invoke({"messages": [{"role": "user", "content": "Draw a fox"}]})

    first, second = _tool_messages(result["messages"])
    assert json.loads(first.content)["status"] == "needs_confirmation"
    assert json.loads(second.content)["status"] == "succeeded"
    assert create.call_count == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_create_react_agent_quotes_then_runs(api: respx.MockRouter) -> None:
    from langgraph.prebuilt import create_react_agent

    api.post("/api/v1/jobs/quote").respond(200, json=envelope(quote(DEFAULT_IMAGE_MODEL)))
    create = api.post("/api/v1/jobs/createTask").respond(202, json=envelope(created()))
    api.get("/api/v1/jobs/recordInfo").respond(200, json=envelope(task_record("succeeded")))

    agent = create_react_agent(ScriptedChatModel(script=_script()), PopNowToolkit().get_tools())
    result = agent.invoke({"messages": [{"role": "user", "content": "Draw a fox"}]})

    first, second = _tool_messages(result["messages"])
    assert json.loads(first.content)["status"] == "needs_confirmation"
    assert json.loads(second.content)["status"] == "succeeded"
    assert create.call_count == 1
    assert result["messages"][-1].content == "Done."
