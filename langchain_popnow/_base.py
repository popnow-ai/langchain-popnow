"""Shared settings, spending guard and task execution for every PopNow tool."""

from __future__ import annotations

import decimal
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import popnow
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator, model_validator

from ._version import __version__

DEFAULT_BASE_URL = "https://api.popnow.ai"
API_KEY_ENV = "POPNOW_API_KEY"
BASE_URL_ENV = "POPNOW_BASE_URL"
KEYS_URL = "https://popnow.ai/api/console/keys"

#: Server-side wait used on createTask; the server clamps it to 60 seconds.
SERVER_WAIT_SECONDS = 30

_USD_RE = re.compile(r"^\d+(\.\d{1,9})?$")

ToolOutput = Tuple[str, Dict[str, Any]]


def parse_usd(value: Any, name: str = "max_usd") -> Optional[decimal.Decimal]:
    """Parse a nonnegative USD amount with at most nine decimal places.

    Accepts decimal strings (preferred), ints, floats (converted through ``str``)
    and :class:`decimal.Decimal`. Returns ``None`` for ``None`` or an empty string.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a USD amount such as '0.25', not a boolean")
    if isinstance(value, decimal.Decimal):
        text = format(value, "f")
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip().lstrip("$").strip()
    else:
        raise ValueError(f"{name} must be a USD amount such as '0.25'")
    if text == "":
        return None
    if not _USD_RE.match(text):
        raise ValueError(
            f"{name} must be a nonnegative USD amount with at most nine decimal places, such as '0.25' (got {value!r})"
        )
    return decimal.Decimal(text)


def usd_text(value: Optional[decimal.Decimal]) -> Optional[str]:
    if value is None:
        return None
    return format(value.normalize(), "f") if value != 0 else "0"


def normalize_usd_field(value: Any) -> Optional[str]:
    """Pydantic ``before`` validator body: turn any accepted amount into a decimal string."""
    return usd_text(parse_usd(value))


def validation_error_message(error: Exception) -> str:
    """Tell the agent exactly which argument was invalid so it can correct the call."""
    errors = getattr(error, "errors", None)
    if callable(errors):
        parts = []
        for item in errors():
            where = ".".join(str(p) for p in item.get("loc", ())) or "input"
            parts.append(f"{where}: {item.get('msg', 'invalid value')}")
        if parts:
            return "Invalid tool arguments. " + "; ".join(parts)
    return f"Invalid tool arguments: {error}"


class PopNowBaseTool(BaseTool):
    """Base class with the PopNow connection settings shared by all tools.

    Args:
        api_key: PopNow API key (``sk-popnow-...``). Defaults to ``POPNOW_API_KEY``.
        base_url: API base URL. Defaults to ``POPNOW_BASE_URL`` or ``https://api.popnow.ai``.
        timeout: Per-request HTTP timeout in seconds.
        max_retries: Automatic retries for network errors and retryable responses.
        client: Optional pre-built :class:`popnow.PopNow` client (sync calls).
        async_client: Optional pre-built :class:`popnow.AsyncPopNow` client (async calls).
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
    timeout: float = 60.0
    max_retries: int = 2
    client: Optional[Any] = Field(default=None, exclude=True)
    async_client: Optional[Any] = Field(default=None, exclude=True)

    handle_tool_error: Union[bool, str, Any] = True
    handle_validation_error: Union[bool, str, Any] = validation_error_message

    _sync_client: Optional[popnow.PopNow] = PrivateAttr(default=None)

    def _client_kwargs(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key.get_secret_value(),
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": {"X-PopNow-Integration": f"langchain-popnow/{__version__}"},
        }

    def _get_client(self) -> popnow.PopNow:
        if self.client is not None:
            return self.client
        if self._sync_client is None:
            self._sync_client = popnow.PopNow(**self._client_kwargs())
        return self._sync_client

    def _new_async_client(self) -> Tuple[popnow.AsyncPopNow, bool]:
        """Return an async client and whether this call owns (and must close) it.

        A fresh client per call keeps the underlying ``httpx.AsyncClient`` bound to
        the running event loop.
        """
        if self.async_client is not None:
            return self.async_client, False
        return popnow.AsyncPopNow(**self._client_kwargs()), True


def api_error(err: popnow.PopNowError) -> ToolException:
    """Wrap an SDK error as a ToolException the agent can read and act on."""
    if isinstance(err, popnow.InsufficientBalanceError):
        hint = " Add funds at https://popnow.ai/api/console/billing."
    elif isinstance(err, popnow.SpendCapExceededError):
        hint = " A spend cap on this API key was reached."
    elif isinstance(err, popnow.AuthenticationError):
        hint = f" Check the API key ({KEYS_URL})."
    elif isinstance(err, popnow.QuoteMismatchError):
        hint = " The quote expired or the price changed; call the tool again to get a new quote."
    elif isinstance(err, popnow.BadRequestError):
        hint = " Check the model's input fields with popnow_list_models(model=...)."
    else:
        hint = ""
    return ToolException(f"PopNow API error: {err}.{hint}")


def output_summary(task: popnow.Task) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for index, asset in enumerate(task.assets):
        item: Dict[str, Any] = {"url": asset.url, "uri": f"popnow://g/{task.task_id}/{index}"}
        for key in ("mime", "width", "height", "duration_seconds", "expires_at"):
            value = getattr(asset, key)
            if value is not None:
                item[key] = value
        outputs.append(item)
    return outputs


def task_result(task: popnow.Task) -> ToolOutput:
    """Content (compact JSON for the model) and artifact (the full task record)."""
    content: Dict[str, Any] = {
        "status": task.state,
        "task_id": task.task_id,
        "model": task.model,
        "cost_usd": task.cost_usd,
    }
    if task.is_succeeded:
        content["outputs"] = output_summary(task)
        if task.text:
            content["text"] = task.text
        content["note"] = (
            "Signed download URLs expire. The popnow:// URI never expires and can be passed to another PopNow tool."
        )
    elif task.is_terminal:
        content["error_code"] = task.error_code
        content["error_message"] = task.error_message
        content["note"] = "The task did not succeed. Its funds hold is released; nothing more is charged."
    else:
        content["note"] = "Still running. Call popnow_get_task with this task_id to collect the result."
    content = {k: v for k, v in content.items() if v is not None}
    return json.dumps(content, ensure_ascii=False), {"type": "task", **task.raw}


class PopNowPaidTool(PopNowBaseTool):
    """Base class for tools that create billed tasks.

    Every call is quoted first. A task only starts when the quoted maximum charge
    is within an explicit approval: the ``max_usd`` argument of the tool call or,
    for unattended pipelines, the ``auto_approve_usd`` setting. Otherwise the tool
    returns ``needs_confirmation`` with the price and nothing is created or charged.

    Args:
        model: Default PopNow model id used when the call does not pass ``model``.
        max_usd_per_call: Hard ceiling set by the developer. A quote above it is
            refused even if the agent approved a higher amount.
        auto_approve_usd: Run without asking when the quote is at or below this
            amount. ``None`` (default) means every paid call needs ``max_usd``.
        allow_model_override: Let the agent choose another model id per call.
        wait_timeout: Seconds to wait for the result before returning ``running``
            with the task id (the task keeps running; use ``popnow_get_task``).
        poll_interval: Initial polling interval in seconds (grows by 1.5x up to 15s).
    """

    model: str = ""
    max_usd_per_call: Optional[str] = None
    auto_approve_usd: Optional[str] = None
    allow_model_override: bool = True
    wait_timeout: float = 600.0
    poll_interval: float = 2.0

    response_format: str = "content_and_artifact"

    @field_validator("max_usd_per_call", "auto_approve_usd", mode="before")
    @classmethod
    def _check_usd(cls, value: Any) -> Optional[str]:
        return normalize_usd_field(value)

    @model_validator(mode="after")
    def _check_limits(self) -> "PopNowPaidTool":
        ceiling = parse_usd(self.max_usd_per_call)
        auto = parse_usd(self.auto_approve_usd)
        if ceiling is not None and auto is not None and auto > ceiling:
            raise ValueError("auto_approve_usd cannot be higher than max_usd_per_call")
        return self

    # Subclasses build the PopNow model id and input from the tool-call arguments.
    def _build_request(self, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def _resolve_model(self, requested: Optional[str], default: str) -> str:
        if requested and requested.strip():
            if not self.allow_model_override and requested.strip() != default:
                raise ToolException(f"This tool is configured to use {default} only.")
            return requested.strip()
        return default

    @staticmethod
    def _merge_input(fields: Mapping[str, Any], extra: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(extra or {})
        merged.update({k: v for k, v in fields.items() if v is not None})
        return merged

    def _decide(self, quote: popnow.TaskQuote, max_usd: Optional[str]) -> Optional[ToolOutput]:
        """Return a non-running result when the quote is not approved, else ``None``."""
        price = parse_usd(quote.max_charge_usd or "0", "max_charge_usd") or decimal.Decimal(0)
        ceiling = parse_usd(self.max_usd_per_call)
        approved = parse_usd(max_usd)
        auto = parse_usd(self.auto_approve_usd)
        base: Dict[str, Any] = {
            "model": quote.model,
            "cost_usd": quote.max_charge_usd,
            "expected_cost_usd": quote.expected_charge_usd,
            "currency": "USD",
        }
        if ceiling is not None and price > ceiling:
            content = {
                "status": "over_limit",
                **base,
                "max_usd_per_call": self.max_usd_per_call,
                "note": "This exceeds the per-call limit configured for this tool. Nothing was started or charged. "
                "Try a cheaper model, shorter duration or lower resolution.",
            }
            return self._quote_output(content, quote)
        if approved is None:
            if auto is not None and price <= auto:
                return None
            content = {
                "status": "needs_confirmation",
                **base,
                "note": f"Nothing was started or charged. This costs up to ${quote.max_charge_usd} USD from the "
                "PopNow API balance. Ask the user to approve, then call this tool again with the same arguments "
                f"plus max_usd=\"{quote.max_charge_usd}\".",
            }
            return self._quote_output(content, quote)
        if price > approved:
            content = {
                "status": "needs_confirmation",
                **base,
                "approved_usd": usd_text(approved),
                "note": f"The price is now up to ${quote.max_charge_usd} USD, above the approved "
                f"${usd_text(approved)}. Nothing was started or charged. Ask the user to approve the new price.",
            }
            return self._quote_output(content, quote)
        return None

    @staticmethod
    def _quote_output(content: Dict[str, Any], quote: popnow.TaskQuote) -> ToolOutput:
        content = {k: v for k, v in content.items() if v is not None}
        return json.dumps(content, ensure_ascii=False), {"type": "quote", **quote.raw}

    @staticmethod
    def _running(created: popnow.TaskCreated, model: str) -> ToolOutput:
        content = {
            "status": "running",
            "task_id": created.task_id,
            "model": model,
            "cost_usd": created.estimated_cost_usd,
            "note": "Still running. Call popnow_get_task with this task_id to collect the result.",
        }
        content = {k: v for k, v in content.items() if v is not None}
        return json.dumps(content, ensure_ascii=False), {"type": "task_created", **created.raw}

    def _create_kwargs(self, quote: popnow.TaskQuote, model: str, model_input: Dict[str, Any]) -> Dict[str, Any]:
        wait = SERVER_WAIT_SECONDS if self.wait_timeout >= SERVER_WAIT_SECONDS else int(self.wait_timeout)
        return {
            "model": model,
            "input": model_input,
            "quote_id": quote.quote_id or None,
            "expected_cost_usd": quote.max_charge_usd,
            "idempotency_key": popnow.new_idempotency_key(),
            "wait": wait or None,
        }

    def _run(self, *args: Any, run_manager: Any = None, **kwargs: Any) -> ToolOutput:
        max_usd = kwargs.pop("max_usd", None)
        model, model_input = self._build_request(**kwargs)
        client = self._get_client()
        try:
            quote = client.jobs.quote(model=model, input=model_input)
            refused = self._decide(quote, max_usd)
            if refused is not None:
                return refused
            created = client.jobs.create(**self._create_kwargs(quote, model, model_input))
            try:
                task = client.jobs.wait(
                    created, timeout=max(self.wait_timeout, 0.0), poll_interval=self.poll_interval
                )
            except popnow.WaitTimeoutError:
                return self._running(created, model)
        except popnow.PopNowError as err:
            raise api_error(err) from err
        return task_result(task)

    async def _arun(self, *args: Any, run_manager: Any = None, **kwargs: Any) -> ToolOutput:
        max_usd = kwargs.pop("max_usd", None)
        model, model_input = self._build_request(**kwargs)
        client, owned = self._new_async_client()
        try:
            quote = await client.jobs.quote(model=model, input=model_input)
            refused = self._decide(quote, max_usd)
            if refused is not None:
                return refused
            created = await client.jobs.create(**self._create_kwargs(quote, model, model_input))
            try:
                task = await client.jobs.wait(
                    created, timeout=max(self.wait_timeout, 0.0), poll_interval=self.poll_interval
                )
            except popnow.WaitTimeoutError:
                return self._running(created, model)
        except popnow.PopNowError as err:
            raise api_error(err) from err
        finally:
            if owned:
                await client.close()
        return task_result(task)
