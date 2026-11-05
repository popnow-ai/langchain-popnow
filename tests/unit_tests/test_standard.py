"""LangChain standard unit tests for every PopNow tool."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_popnow import (
    PopNowCheckBalance,
    PopNowCreateTask,
    PopNowGetTask,
    PopNowImageGeneration,
    PopNowListModels,
    PopNowMusicGeneration,
    PopNowSpeech,
    PopNowVideoGeneration,
)

from .conftest import API_KEY

ENV = {"POPNOW_API_KEY": "sk-popnow-from-env", "POPNOW_BASE_URL": "https://api.example.test"}
EXPECTED = {"api_key": "sk-popnow-from-env", "base_url": "https://api.example.test"}


class _Base(ToolsUnitTests):
    tool_class: Type[BaseTool]

    @property
    def tool_constructor(self) -> Type[BaseTool]:
        return self.tool_class

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        return {"api_key": API_KEY}

    @property
    def init_from_env_params(self) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Any]]:
        return ENV, {}, EXPECTED


class TestImageGenerationStandard(_Base):
    tool_class = PopNowImageGeneration

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"prompt": "A paper boat on a calm lake at dawn", "aspect_ratio": "16:9"}


class TestVideoGenerationStandard(_Base):
    tool_class = PopNowVideoGeneration

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"prompt": "A lighthouse in a storm, slow push-in", "duration_seconds": 5, "resolution": "768p"}


class TestMusicGenerationStandard(_Base):
    tool_class = PopNowMusicGeneration

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"prompt": "Upbeat lo-fi beat for a cooking video", "instrumental": True}


class TestSpeechStandard(_Base):
    tool_class = PopNowSpeech

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"text": "Welcome back to the channel!", "voice_id": "Kore"}


class TestCreateTaskStandard(_Base):
    tool_class = PopNowCreateTask

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"model": "alibaba/z-image-turbo/text-to-image", "input": {"prompt": "A red fox in a teacup"}}


class TestListModelsStandard(_Base):
    tool_class = PopNowListModels

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"modality": "video", "search": "kling"}


class TestGetTaskStandard(_Base):
    tool_class = PopNowGetTask

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        return {"task_id": "job_1", "wait_seconds": 0}


class TestCheckBalanceStandard(_Base):
    tool_class = PopNowCheckBalance
