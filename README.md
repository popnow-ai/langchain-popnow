# langchain-popnow

[![PyPI](https://img.shields.io/pypi/v/langchain-popnow.svg)](https://pypi.org/project/langchain-popnow/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Website: popnow.ai](https://img.shields.io/badge/site-popnow.ai-ff5a36.svg)](https://popnow.ai)

**LangChain and LangGraph tools for [PopNow](https://popnow.ai): let your agents create AI images, videos, music and speech with Nano Banana, GPT Image, Seedream, MiniMax H3, Kling, Seedance, Wan, Veo, Suno, Gemini TTS and more, through one API key.**

Every paid call is quoted in USD first and only runs after an explicit approval, so an agent cannot spend your [API balance](https://popnow.ai/docs/billing) silently. Compare model prices on the [PopNow AI price index](https://popnow.ai/ai-price-index).

Built on the official [`popnow`](https://pypi.org/project/popnow/) Python SDK.

## Installation

```bash
pip install -U langchain-popnow
```

## Setup

1. Create an API key at [popnow.ai/api](https://popnow.ai/api) (API console, **Keys**). Giving your agent a key of its own with a low daily spend cap is a good habit.
2. Add funds to the API balance in [API billing](https://popnow.ai/api/console/billing). The API balance is in USD and is separate from popnow.ai website credits.
3. Export the key:

```bash
export POPNOW_API_KEY="sk-popnow-..."
```

You can also pass `api_key="sk-popnow-..."` to any tool or to `PopNowToolkit`. `POPNOW_BASE_URL` overrides the API host (default `https://api.popnow.ai`).

## Tools

| Tool class | Tool name | What it does | Default model | Billing |
| --- | --- | --- | --- | --- |
| `PopNowImageGeneration` | `popnow_image_generation` | Text-to-image, or edit images passed in `image_urls` | `google/nano-banana-2.1/text-to-image` (edit: `google/nano-banana-2.1/edit`) | Paid, quoted first |
| `PopNowVideoGeneration` | `popnow_video_generation` | Text-to-video, or animate the picture in `image_url` | `minimax/h3/text-to-video` (image: `minimax/h3/image-to-video`) | Paid, quoted first |
| `PopNowMusicGeneration` | `popnow_music_generation` | Songs and instrumentals, optional lyrics | `suno/v6/text-to-audio` | Paid, quoted first |
| `PopNowSpeech` | `popnow_text_to_speech` | Text-to-speech in many voices | `google/gemini-3.8-flash-tts/text-to-speech` | Paid, quoted first |
| `PopNowCreateTask` | `popnow_create_task` | Any PopNow model by id with its own `input` | (agent picks) | Paid, quoted first |
| `PopNowListModels` | `popnow_list_models` | Model ids and USD rates; with `model`, its input fields | | Free |
| `PopNowGetTask` | `popnow_get_task` | Collect the result of a task that is still running | | Free |
| `PopNowCheckBalance` | `popnow_check_balance` | Available and held API balance in USD | | Free |

`PopNowToolkit().get_tools()` returns all of them with one shared configuration.

The generation tools accept the common fields (`prompt`, `aspect_ratio`, `resolution`, `duration_seconds`, `voice_id` ...), an optional `model` to pick another PopNow model, and `extra_input` for any model-specific field listed by `popnow_list_models(model=...)`.

## How approval works

PopNow jobs are asynchronous: quote, create with the approved price, then poll. The tools follow the same flow as the [PopNow developer MCP server](https://popnow.ai/docs/mcp):

1. **Quote.** The agent calls a paid tool without `max_usd`. The tool asks PopNow for a quote and returns `status: "needs_confirmation"` with `cost_usd`. Nothing is created or held.
2. **Approve.** The agent shows the price to the user. After approval it calls the tool again with the same arguments plus `max_usd` set to the quoted `cost_usd`.
3. **Run.** If the current quote is within `max_usd`, the task is created with that quote locked (`quoteId` + `expectedCostUsd`), and the tool waits for the result. If the price went up, the tool returns a new `needs_confirmation` instead and charges nothing.

Results contain signed download URLs (they expire) and a `popnow://g/<task>/<n>` URI that never expires and can be passed to another PopNow tool, for example to animate a generated image. Long videos may still be running when `wait_timeout` (default 600 s) ends: the tool then returns `status: "running"` with a `task_id`, and `popnow_get_task` collects the result. Failed or expired tasks release their hold and cost nothing.

Developer-side guards, on any paid tool or on `PopNowToolkit`:

| Setting | Effect |
| --- | --- |
| `max_usd_per_call="1.00"` | Hard ceiling. A quote above it returns `status: "over_limit"` even if the agent passed a higher `max_usd`. |
| `auto_approve_usd="0.05"` | Unattended pipelines: quotes at or below this amount run without `max_usd`. Default `None`: every paid call needs approval. |
| `allow_model_override=False` | The agent cannot switch away from the configured `model`. |

API key spend caps and model allow-lists set in the [API console](https://popnow.ai/api/console/keys) apply as well.

## Quick start

```python
from langchain_popnow import PopNowImageGeneration

tool = PopNowImageGeneration()

# 1. Quote: returns the price, starts nothing.
print(tool.invoke({"prompt": "A small red fox curled up in a teacup, soft morning light"}))
# {"status": "needs_confirmation", "model": "google/nano-banana-2.1/text-to-image", "cost_usd": "0.048", ...}

# 2. Approved: creates the task and waits for it.
print(tool.invoke({"prompt": "A small red fox curled up in a teacup, soft morning light", "max_usd": "0.048"}))
# {"status": "succeeded", "task_id": "...", "cost_usd": "0.048", "outputs": [{"url": "https://...", "uri": "popnow://g/.../0"}], ...}
```

When the tool is called with a `ToolCall` (as agents do), it returns a `ToolMessage` whose `content` is the JSON summary above and whose `artifact` is the full PopNow task or quote record.

## Use with a LangChain agent

```python
from langchain.agents import create_agent
from langchain_popnow import PopNowToolkit

tools = PopNowToolkit(max_usd_per_call="1.00").get_tools()

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    system_prompt=(
        "You create media with PopNow tools. Paid tools return a price first: "
        "show it to the user and only call again with max_usd after they approve."
    ),
)

result = agent.invoke({"messages": [{"role": "user", "content": "Make a 9:16 clip of a vintage car on a coastal road"}]})
print(result["messages"][-1].content)
```

## Use with LangGraph `create_react_agent`

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_popnow import PopNowToolkit

model = init_chat_model("anthropic:claude-sonnet-4-6")
agent = create_react_agent(model, PopNowToolkit(max_usd_per_call="1.00").get_tools())

for step in agent.stream(
    {"messages": [{"role": "user", "content": "Write a short lo-fi track for a cooking video and tell me the price first"}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

`create_react_agent` still works in LangGraph 1.x; new projects can use `langchain.agents.create_agent` as shown above. For a hard human approval step, combine the tools with LangChain's `HumanInTheLoopMiddleware` or a LangGraph `interrupt`.

## More examples

```python
from langchain_popnow import PopNowListModels, PopNowSpeech, PopNowVideoGeneration

# Find a model and its inputs (free).
PopNowListModels().invoke({"modality": "video", "search": "kling"})
PopNowListModels().invoke({"model": "kling/3.0/text-to-video"})

# Animate a picture with a specific model and an extra model field.
PopNowVideoGeneration().invoke({
    "prompt": "Gentle waves, slow push-in",
    "image_url": "https://example.com/beach.jpg",
    "model": "kling/3.0/image-to-video",
    "duration_seconds": 5,
    "max_usd": "0.50",
})

# Voiceovers that cost a fraction of a cent can run without asking.
speech = PopNowSpeech(auto_approve_usd="0.02")
speech.invoke({"text": "Welcome back to the channel!", "voice_id": "Kore"})
```

Async is supported on every tool (`await tool.ainvoke(...)`).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" langchain
pytest                                   # unit tests, mocked HTTP, no network
POPNOW_API_KEY=sk-popnow-... pytest tests/integration_tests   # free calls only: catalog, balance, quotes
```

Unit tests include the LangChain standard tool tests (`langchain-tests`). Integration tests never pass `max_usd`, so they only request quotes and never create a paid task.

## Links

- PopNow: [popnow.ai](https://popnow.ai)
- API keys and console: [popnow.ai/api](https://popnow.ai/api)
- API docs: [popnow.ai/docs](https://popnow.ai/docs) ([quickstart](https://popnow.ai/docs/quickstart), [models](https://popnow.ai/docs/models), [billing](https://popnow.ai/docs/billing), [webhooks](https://popnow.ai/docs/webhooks))
- Model prices: [AI price index](https://popnow.ai/ai-price-index)
- Python SDK: [popnow-python](https://github.com/popnow-ai/popnow-python)

## License

[MIT](LICENSE)
