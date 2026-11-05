"""LangChain standard integration tests against the live PopNow API.

They only call free endpoints: the model catalog, the balance and quotes. Paid
tools are invoked without ``max_usd``, so the API returns a price and nothing is
created or charged. Run with a real key:

    POPNOW_API_KEY=sk-popnow-... pytest tests/integration_tests
"""

from __future__ import annotations

import os
from typing import Any, Dict, Type

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_popnow import (
    PopNowCheckBalance,
    PopNowImageGeneration,
    PopNowListModels,
    PopNowMusicGeneration,
    PopNowSpeech,
    PopNowVideoGeneration,
)

pytestmark = pytest.mark.skipif(not os.environ.get("POPNOW_API_KEY"), reason="POPNOW_API_KEY is not set")


class _Base(ToolsIntegrationTests):
    tool_class: Type[BaseTool]

    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return self.tool_class


class TestListModelsIntegration(_Base):
    tool_class = PopNowListModels

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"modality": "image", "limit": 5}


class TestCheckBalanceIntegration(_Base):
    tool_class = PopNowCheckBalance


class TestImageQuoteIntegration(_Base):
    tool_class = PopNowImageGeneration

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"prompt": "A paper boat on a calm lake at dawn"}


class TestVideoQuoteIntegration(_Base):
    tool_class = PopNowVideoGeneration

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"prompt": "A lighthouse in a storm", "duration_seconds": 5, "resolution": "480p"}


class TestMusicQuoteIntegration(_Base):
    tool_class = PopNowMusicGeneration

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"prompt": "Calm piano loop", "instrumental": True}


class TestSpeechQuoteIntegration(_Base):
    tool_class = PopNowSpeech

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"text": "Hello from PopNow.", "voice_id": "Kore"}
