"""LangChain tools for PopNow: AI image, video, music and speech generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import popnow
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, field_validator

from ._base import (
    PopNowBaseTool,
    PopNowPaidTool,
    ToolOutput,
    api_error,
    normalize_usd_field,
    task_result,
)

__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_IMAGE_EDIT_MODEL",
    "DEFAULT_VIDEO_MODEL",
    "DEFAULT_IMAGE_TO_VIDEO_MODEL",
    "DEFAULT_MUSIC_MODEL",
    "DEFAULT_SPEECH_MODEL",
    "PopNowImageGeneration",
    "PopNowVideoGeneration",
    "PopNowMusicGeneration",
    "PopNowSpeech",
    "PopNowCreateTask",
    "PopNowListModels",
    "PopNowGetTask",
    "PopNowCheckBalance",
]

DEFAULT_IMAGE_MODEL = "google/nano-banana-2.1/text-to-image"
DEFAULT_IMAGE_EDIT_MODEL = "google/nano-banana-2.1/edit"
DEFAULT_VIDEO_MODEL = "minimax/h3/text-to-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "minimax/h3/image-to-video"
DEFAULT_MUSIC_MODEL = "suno/v6/text-to-audio"
DEFAULT_SPEECH_MODEL = "google/gemini-3.8-flash-tts/text-to-speech"

_MAX_USD_DESCRIPTION = (
    "USD amount the user approved for this call, as a decimal string such as '0.05'. "
    "Omit it on the first call: the tool then returns the exact price with status "
    "'needs_confirmation' and starts nothing. After the user approves, call again with the "
    "same arguments and max_usd set to the quoted cost_usd. Never set it without the user's approval."
)
_MODEL_DESCRIPTION = (
    "Optional PopNow model id (for example 'kling/3.0/text-to-video'). "
    "Leave empty for the default. Use popnow_list_models to find ids."
)
_EXTRA_DESCRIPTION = (
    "Optional extra model-specific input fields, exactly as listed by "
    "popnow_list_models(model=...). Unknown fields are rejected without charge."
)


class _PaidInput(BaseModel):
    # Subclasses declare their own fields first, then the shared model, extra_input and max_usd.

    @field_validator("max_usd", mode="before", check_fields=False)
    @classmethod
    def _usd(cls, value: Any) -> Optional[str]:
        return normalize_usd_field(value)


class ImageGenerationInput(_PaidInput):
    """Arguments of :class:`PopNowImageGeneration`."""

    prompt: str = Field(description="Description of the picture to create, or the edit to make.")
    image_urls: Optional[List[str]] = Field(
        default=None,
        description="Optional source image URLs (or popnow:// URIs) to edit. When set, the default edit model is used.",
    )
    aspect_ratio: Optional[str] = Field(
        default=None, description="Aspect ratio such as '1:1', '16:9', '9:16', '4:3' or '3:4'."
    )
    resolution: Optional[str] = Field(default=None, description="Output size such as '1k', '2k' or '4k'.")
    model: Optional[str] = Field(default=None, description=_MODEL_DESCRIPTION)
    extra_input: Optional[Dict[str, Any]] = Field(default=None, description=_EXTRA_DESCRIPTION)
    max_usd: Optional[str] = Field(default=None, description=_MAX_USD_DESCRIPTION)


class VideoGenerationInput(_PaidInput):
    """Arguments of :class:`PopNowVideoGeneration`."""

    prompt: str = Field(description="Description of the scene, motion and camera.")
    image_url: Optional[str] = Field(
        default=None,
        description="Optional start frame URL (or popnow:// URI) to animate. When set, the default image-to-video model is used.",
    )
    duration_seconds: Optional[int] = Field(default=None, description="Clip length in seconds, for example 5 or 10.")
    resolution: Optional[str] = Field(default=None, description="Resolution such as '480p', '768p' or '2k'.")
    aspect_ratio: Optional[str] = Field(
        default=None, description="Aspect ratio such as '16:9', '9:16' or '1:1' (text-to-video only)."
    )
    model: Optional[str] = Field(default=None, description=_MODEL_DESCRIPTION)
    extra_input: Optional[Dict[str, Any]] = Field(default=None, description=_EXTRA_DESCRIPTION)
    max_usd: Optional[str] = Field(default=None, description=_MAX_USD_DESCRIPTION)


class MusicGenerationInput(_PaidInput):
    """Arguments of :class:`PopNowMusicGeneration`."""

    prompt: str = Field(description="Style, mood, instruments and topic of the song.")
    lyrics: Optional[str] = Field(default=None, description="Optional lyrics to sing.")
    instrumental: Optional[bool] = Field(default=None, description="True for an instrumental track without vocals.")
    title: Optional[str] = Field(default=None, description="Optional song title.")
    model: Optional[str] = Field(default=None, description=_MODEL_DESCRIPTION)
    extra_input: Optional[Dict[str, Any]] = Field(default=None, description=_EXTRA_DESCRIPTION)
    max_usd: Optional[str] = Field(default=None, description=_MAX_USD_DESCRIPTION)


class SpeechInput(_PaidInput):
    """Arguments of :class:`PopNowSpeech`."""

    text: str = Field(description="The text to read aloud.")
    voice_id: Optional[str] = Field(default=None, description="Voice name, for example 'Kore', 'Puck' or 'Charon'.")
    style: Optional[str] = Field(
        default=None, description="Optional delivery style, for example 'newscaster', 'whisper' or 'empathetic'."
    )
    model: Optional[str] = Field(default=None, description=_MODEL_DESCRIPTION)
    extra_input: Optional[Dict[str, Any]] = Field(default=None, description=_EXTRA_DESCRIPTION)
    max_usd: Optional[str] = Field(default=None, description=_MAX_USD_DESCRIPTION)


class CreateTaskInput(BaseModel):
    """Arguments of :class:`PopNowCreateTask`."""

    model: str = Field(description="Exact PopNow model id, for example 'google/nano-banana-2.1/text-to-image'.")
    input: Dict[str, Any] = Field(description="The model's own input fields, as listed by popnow_list_models(model=...).")
    max_usd: Optional[str] = Field(default=None, description=_MAX_USD_DESCRIPTION)

    @field_validator("max_usd", mode="before")
    @classmethod
    def _usd(cls, value: Any) -> Optional[str]:
        return normalize_usd_field(value)


class PopNowImageGeneration(PopNowPaidTool):
    """Create or edit images with PopNow (Nano Banana, GPT Image, Seedream, FLUX and more).

    Setup:
        Install ``langchain-popnow`` and set ``POPNOW_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-popnow
            export POPNOW_API_KEY="sk-popnow-..."

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowImageGeneration

            tool = PopNowImageGeneration()

    Invoke:
        .. code-block:: python

            # First call: returns the price, starts nothing.
            tool.invoke({"prompt": "A paper boat on a calm lake at dawn"})
            # After the user approves the quoted cost_usd:
            tool.invoke({"prompt": "A paper boat on a calm lake at dawn", "max_usd": "0.048"})
    """

    name: str = "popnow_image_generation"
    description: str = (
        "Create a new image from a text prompt, or edit images given by URL, using PopNow. "
        "Paid: the first call returns the USD price; call again with max_usd after the user approves. "
        "Returns download URLs."
    )
    args_schema: Type[BaseModel] = ImageGenerationInput
    model: str = DEFAULT_IMAGE_MODEL
    edit_model: str = DEFAULT_IMAGE_EDIT_MODEL

    def _build_request(
        self,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        model: Optional[str] = None,
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        default = self.edit_model if image_urls else self.model
        fields = {"prompt": prompt, "image_urls": image_urls, "aspect_ratio": aspect_ratio, "resolution": resolution}
        return self._resolve_model(model, default), self._merge_input(fields, extra_input)


class PopNowVideoGeneration(PopNowPaidTool):
    """Create short videos with PopNow (MiniMax H3, Kling, Seedance, Wan, Veo and more).

    Text-to-video by default; pass ``image_url`` to animate a picture. Video usually
    takes one to a few minutes; when ``wait_timeout`` runs out the tool returns the
    task id and ``popnow_get_task`` collects the result.

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowVideoGeneration

            tool = PopNowVideoGeneration(max_usd_per_call="1.00")
    """

    name: str = "popnow_video_generation"
    description: str = (
        "Create a short video from a text prompt, or animate an image given by URL, using PopNow. "
        "Paid: the first call returns the USD price; call again with max_usd after the user approves. "
        "Returns a download URL, or a task_id to check with popnow_get_task if it is still running."
    )
    args_schema: Type[BaseModel] = VideoGenerationInput
    model: str = DEFAULT_VIDEO_MODEL
    image_to_video_model: str = DEFAULT_IMAGE_TO_VIDEO_MODEL

    def _build_request(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        model: Optional[str] = None,
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        default = self.image_to_video_model if image_url else self.model
        chosen = self._resolve_model(model, default)
        if image_url and chosen == DEFAULT_IMAGE_TO_VIDEO_MODEL:
            # MiniMax H3 image-to-video follows the picture's own aspect ratio.
            aspect_ratio = None
        fields = {
            "prompt": prompt,
            "image_url": image_url,
            "duration_seconds": duration_seconds,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }
        return chosen, self._merge_input(fields, extra_input)


class PopNowMusicGeneration(PopNowPaidTool):
    """Create songs and instrumentals with PopNow (Suno and more).

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowMusicGeneration

            tool = PopNowMusicGeneration()
    """

    name: str = "popnow_music_generation"
    description: str = (
        "Create an original song or instrumental track from a description, optionally with lyrics, using PopNow. "
        "Paid: the first call returns the USD price; call again with max_usd after the user approves. "
        "Returns audio download URLs."
    )
    args_schema: Type[BaseModel] = MusicGenerationInput
    model: str = DEFAULT_MUSIC_MODEL

    def _build_request(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        instrumental: Optional[bool] = None,
        title: Optional[str] = None,
        model: Optional[str] = None,
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        fields = {"prompt": prompt, "lyrics": lyrics, "instrumental": instrumental, "title": title}
        return self._resolve_model(model, self.model), self._merge_input(fields, extra_input)


class PopNowSpeech(PopNowPaidTool):
    """Turn text into natural speech with PopNow (Gemini TTS, OpenAI TTS, Qwen TTS and more).

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowSpeech

            tool = PopNowSpeech(auto_approve_usd="0.02")
    """

    name: str = "popnow_text_to_speech"
    description: str = (
        "Read text aloud as a voiceover in many voices and languages using PopNow. "
        "Paid: the first call returns the USD price; call again with max_usd after the user approves. "
        "Returns an audio download URL."
    )
    args_schema: Type[BaseModel] = SpeechInput
    model: str = DEFAULT_SPEECH_MODEL

    def _build_request(
        self,
        text: str,
        voice_id: Optional[str] = None,
        style: Optional[str] = None,
        model: Optional[str] = None,
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        fields = {"text": text, "voice_id": voice_id, "style": style}
        return self._resolve_model(model, self.model), self._merge_input(fields, extra_input)


class PopNowCreateTask(PopNowPaidTool):
    """Run any PopNow model by its exact id with the model's own input (like ``createTask``).

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowCreateTask

            tool = PopNowCreateTask()
            tool.invoke({"model": "alibaba/z-image-turbo/text-to-image", "input": {"prompt": "A red fox"}})
    """

    name: str = "popnow_create_task"
    description: str = (
        "Run any PopNow image, video or audio model by its exact model id with the model's own input fields "
        "(see popnow_list_models). Paid: the first call returns the USD price; call again with max_usd after "
        "the user approves."
    )
    args_schema: Type[BaseModel] = CreateTaskInput

    def _build_request(self, model: str, input: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if self.model and not self.allow_model_override and model != self.model:
            raise ToolException(f"This tool is configured to use {self.model} only.")
        return model, dict(input)


# Free tools


class ListModelsInput(BaseModel):
    """Arguments of :class:`PopNowListModels`."""

    modality: Optional[Literal["image", "video", "audio"]] = Field(
        default=None, description="Filter by output type: 'image', 'video' or 'audio'."
    )
    search: Optional[str] = Field(default=None, description="Free-text search, for example 'kling' or 'upscale'.")
    task: Optional[str] = Field(
        default=None, description="Exact task, for example 'text-to-image', 'image-to-video' or 'text-to-speech'."
    )
    model: Optional[str] = Field(
        default=None, description="A model id: return that model's input fields and pricing instead of a list."
    )
    limit: int = Field(default=30, ge=1, le=200, description="Maximum number of models to return.")


def _model_summary(model: popnow.Model) -> Dict[str, Any]:
    summary = {
        "model": model.model,
        "name": model.display_name,
        "modality": model.modality,
        "tasks": model.tasks or None,
        "usd": model.usd,
        "available": model.available,
    }
    return {k: v for k, v in summary.items() if v is not None}


class PopNowListModels(PopNowBaseTool):
    """List PopNow image, video and audio models with exact ids and USD rates. Free.

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowListModels

            tool = PopNowListModels()
            tool.invoke({"modality": "video", "search": "kling"})
    """

    name: str = "popnow_list_models"
    description: str = (
        "List PopNow AI image, video and audio models with their exact model ids and USD pricing, "
        "or pass model=<id> to get that model's input fields. Free."
    )
    args_schema: Type[BaseModel] = ListModelsInput

    @staticmethod
    def _list_content(models: popnow.ModelList, limit: int) -> str:
        items = [_model_summary(m) for m in list(models)[:limit]]
        return json.dumps({"total": models.total, "returned": len(items), "models": items}, ensure_ascii=False)

    @staticmethod
    def _detail_content(model: popnow.Model) -> str:
        detail = _model_summary(model)
        if model.input_schema is not None:
            detail["input_schema"] = model.input_schema
        if model.max_output_duration_seconds is not None:
            detail["max_output_duration_seconds"] = model.max_output_duration_seconds
        return json.dumps(detail, ensure_ascii=False)

    def _run(
        self,
        modality: Optional[str] = None,
        search: Optional[str] = None,
        task: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 30,
        run_manager: Any = None,
    ) -> str:
        client = self._get_client()
        try:
            if model:
                return self._detail_content(client.models.get(model))
            return self._list_content(client.models.list(modality=modality, search=search, task=task), limit)
        except popnow.PopNowError as err:
            raise api_error(err) from err

    async def _arun(
        self,
        modality: Optional[str] = None,
        search: Optional[str] = None,
        task: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 30,
        run_manager: Any = None,
    ) -> str:
        client, owned = self._new_async_client()
        try:
            if model:
                return self._detail_content(await client.models.get(model))
            return self._list_content(await client.models.list(modality=modality, search=search, task=task), limit)
        except popnow.PopNowError as err:
            raise api_error(err) from err
        finally:
            if owned:
                await client.close()


class GetTaskInput(BaseModel):
    """Arguments of :class:`PopNowGetTask`."""

    task_id: str = Field(description="The task_id returned by a PopNow generation tool.")
    wait_seconds: int = Field(
        default=30, ge=0, le=300, description="Seconds to wait for the task to finish before returning (0 to 300)."
    )


class PopNowGetTask(PopNowBaseTool):
    """Check a PopNow task and collect its result once it is ready. Free.

    Instantiate:
        .. code-block:: python

            from langchain_popnow import PopNowGetTask

            tool = PopNowGetTask()
            tool.invoke({"task_id": "job_..."})
    """

    name: str = "popnow_get_task"
    description: str = (
        "Check a PopNow task that is still running and return its result (download URLs) when ready. Free."
    )
    args_schema: Type[BaseModel] = GetTaskInput
    response_format: str = "content_and_artifact"
    poll_interval: float = 2.0

    def _run(self, task_id: str, wait_seconds: int = 30, run_manager: Any = None) -> ToolOutput:
        client = self._get_client()
        try:
            try:
                task = client.jobs.wait(task_id, timeout=float(wait_seconds), poll_interval=self.poll_interval)
            except popnow.WaitTimeoutError as timeout:
                task = timeout.last or client.jobs.get(task_id)
        except popnow.PopNowError as err:
            raise api_error(err) from err
        return task_result(task)

    async def _arun(self, task_id: str, wait_seconds: int = 30, run_manager: Any = None) -> ToolOutput:
        client, owned = self._new_async_client()
        try:
            try:
                task = await client.jobs.wait(task_id, timeout=float(wait_seconds), poll_interval=self.poll_interval)
            except popnow.WaitTimeoutError as timeout:
                task = timeout.last or await client.jobs.get(task_id)
        except popnow.PopNowError as err:
            raise api_error(err) from err
        finally:
            if owned:
                await client.close()
        return task_result(task)


class _NoInput(BaseModel):
    pass


class PopNowCheckBalance(PopNowBaseTool):
    """Show the available PopNow API balance and funds held for running tasks, in USD. Free."""

    name: str = "popnow_check_balance"
    description: str = "Show the available PopNow API balance and the funds held for running tasks, in USD. Free."
    args_schema: Type[BaseModel] = _NoInput

    @staticmethod
    def _content(balance: popnow.Balance) -> str:
        return json.dumps(
            {
                "currency": "USD",
                "available_usd": balance.available_usd,
                "held_usd": balance.held_usd,
                "total_usd": balance.total_usd,
            }
        )

    def _run(self, run_manager: Any = None) -> str:
        try:
            return self._content(self._get_client().account.balance())
        except popnow.PopNowError as err:
            raise api_error(err) from err

    async def _arun(self, run_manager: Any = None) -> str:
        client, owned = self._new_async_client()
        try:
            return self._content(await client.account.balance())
        except popnow.PopNowError as err:
            raise api_error(err) from err
        finally:
            if owned:
                await client.close()
