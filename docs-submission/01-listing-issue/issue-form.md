# Integration listing issue

Form: https://github.com/langchain-ai/docs/issues/new?template=06-integration-submission.yml

Title (prefilled by the form, replace the placeholder):

```
[integration]: langchain-popnow
```

| Field | Value |
| --- | --- |
| Display or class name | `PopNowToolkit` |
| Language | `Python` |
| Component type | `tools` |
| PyPI package name | `langchain-popnow` |
| npm package name | (leave empty) |
| Docs URL | `https://popnow.ai/docs/langchain` (fallback: `https://github.com/popnow-ai/langchain-popnow#readme`) |
| Source repository | `popnow-ai/langchain-popnow` |
| Short provider description | see below |
| Capability flags (optional) | (leave empty; flags only apply to chat models and vector stores) |
| Confirmations | tick "The package is already published on PyPI and/or npm." only after `pip install langchain-popnow` works |

Short provider description (137 characters, one factual sentence):

```
PopNow API tools for generating images, videos, music and speech with models such as Nano Banana, Kling, MiniMax and Suno, billed in USD.
```

After a maintainer applies `integration-run`, the bot opens a PR. Compare it with [`expected-automated-pr.patch`](expected-automated-pr.patch): the row goes under `python.tools` in `scripts/data/integration_external_docs.yaml`, there is a `packages.yml` entry, a provider card in `all_providers.mdx`, and a row in `src/snippets/oss/python-tools-downloads.mdx`. Exact positions may differ, because the bot places rows itself.
