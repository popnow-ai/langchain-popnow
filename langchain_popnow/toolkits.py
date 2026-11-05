"""PopNowToolkit: every PopNow tool with one shared configuration."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, field_validator

from ._base import API_KEY_ENV, BASE_URL_ENV, DEFAULT_BASE_URL, KEYS_URL, normalize_usd_field
from .tools import (
    PopNowCheckBalance,
    PopNowCreateTask,
    PopNowGetTask,
    PopNowImageGeneration,
    PopNowListModels,
    PopNowMusicGeneration,
    PopNowSpeech,
    PopNowVideoGeneration,
)

__all__ = ["PopNowToolkit"]


class PopNowToolkit(BaseToolkit):
    """All PopNow tools for LangChain and LangGraph agents.

    Setup:
        .. code-block:: bash

            pip install -U langchain-popnow
            export POPNOW_API_KEY="sk-popnow-..."

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowToolkit

            toolkit = PopNowToolkit(max_usd_per_call="1.00")
            tools = toolkit.get_tools()

    Args:
        api_key: PopNow API key. Defaults to ``POPNOW_API_KEY``.
        base_url: API base URL. Defaults to ``POPNOW_BASE_URL`` or ``https://api.popnow.ai``.
        max_usd_per_call: Hard per-call USD ceiling for every paid tool.
        auto_approve_usd: Run paid calls at or below this USD amount without asking.
            ``None`` (default) means every paid call needs an explicit ``max_usd``.
        wait_timeout: Seconds a paid tool waits for its result before returning the task id.
        include_create_task: Include ``popnow_create_task`` (run any model id).
        include_balance: Include ``popnow_check_balance``.
        client / async_client: Optional pre-built ``popnow`` SDK clients.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            API_KEY_ENV,
            error_message=(
                f"PopNow API key not found. Pass api_key=... or set the {API_KEY_ENV} environment "
                f"variable. Create a key at {KEYS_URL}."
            ),
        )
    )
    base_url: str = Field(default_factory=from_env(BASE_URL_ENV, default=DEFAULT_BASE_URL))
    max_usd_per_call: Optional[str] = None
    auto_approve_usd: Optional[str] = None
    wait_timeout: float = 600.0
    include_create_task: bool = True
    include_balance: bool = True
    client: Optional[Any] = Field(default=None, exclude=True)
    async_client: Optional[Any] = Field(default=None, exclude=True)

    @field_validator("max_usd_per_call", "auto_approve_usd", mode="before")
    @classmethod
    def _check_usd(cls, value: Any) -> Optional[str]:
        return normalize_usd_field(value)

    def get_tools(self) -> List[BaseTool]:
        """Return the PopNow tools, sharing one connection configuration."""
        common = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "client": self.client,
            "async_client": self.async_client,
        }
        paid = {
            **common,
            "max_usd_per_call": self.max_usd_per_call,
            "auto_approve_usd": self.auto_approve_usd,
            "wait_timeout": self.wait_timeout,
        }
        tools: List[BaseTool] = [
            PopNowImageGeneration(**paid),
            PopNowVideoGeneration(**paid),
            PopNowMusicGeneration(**paid),
            PopNowSpeech(**paid),
        ]
        if self.include_create_task:
            tools.append(PopNowCreateTask(**paid))
        tools.append(PopNowListModels(**common))
        tools.append(PopNowGetTask(**common))
        if self.include_balance:
            tools.append(PopNowCheckBalance(**common))
        return tools
