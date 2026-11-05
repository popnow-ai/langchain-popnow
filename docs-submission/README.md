# Listing langchain-popnow in the LangChain docs

Drafts for getting `langchain-popnow` listed on [docs.langchain.com](https://docs.langchain.com/oss/python/integrations/providers/overview), based on the `langchain-ai/docs` repository as of 2026-11-05:

- `src/oss/contributing/integrations-langchain.mdx`
- `src/oss/contributing/publish-langchain.mdx`
- `.github/ISSUE_TEMPLATE/06-integration-submission.yml`
- the automated listing PR [#7360](https://github.com/langchain-ai/docs/pull/7360), from issue [#7356](https://github.com/langchain-ai/docs/issues/7356)
- the manual category-table PR [#7082](https://github.com/langchain-ai/docs/pull/7082)

## How listing works now

1. **No integration code goes to `langchain-ai`.** The package lives in its own repository (`popnow-ai/langchain-popnow`) and is published to PyPI.
2. **New listings start as an issue, not a PR.** File the "Integration listing" issue form after the package is on PyPI. A maintainer applies the `integration-run` label, and a bot opens a PR that edits four files:
   - `packages.yml`
   - `scripts/data/integration_external_docs.yaml` (the `python.tools` list)
   - `src/oss/python/integrations/providers/all_providers.mdx` (a provider card)
   - `src/snippets/oss/python-tools-downloads.mdx` (the "All tools and toolkits" downloads table)

   Both the table row and the card link to the `docs_url` from the form, so that URL is where LangChain sends people. Automation closes manual listing PRs from external contributors and points them back to the issue form.
3. **Hosted pages on docs.langchain.com are only for packages with 50,000+ monthly PyPI downloads, or ones maintainers mark as featured.** Until then, only the external listing above is allowed. Do not add a new MDX page or set `featured: true`.
4. **Category tables on the tools index take manual PRs.** `src/oss/python/integrations/tools/index.mdx` has a "Media generation" table (Magic Hour is the only row today). External contributors have had one-row PRs to these tables merged (#7082 Bowmark, #7141 Pidgeon). Bowmark's row was added after its automated listing, which is the order to follow here.

## What is in this folder

| Path | Use | When |
| --- | --- | --- |
| [`01-listing-issue/issue-form.md`](01-listing-issue/issue-form.md) | Values for the issue form fields | Step 1, after the PyPI release |
| [`01-listing-issue/expected-automated-pr.patch`](01-listing-issue/expected-automated-pr.patch) | What the bot's PR should contain, for checking it | When the bot PR opens |
| [`02-media-generation-row-pr/`](02-media-generation-row-pr) | One-row PR to the "Media generation" table, with a PR body that follows the repository template | Step 2, after the listing PR is merged |
| [`03-hosted-guide-if-eligible/`](03-hosted-guide-if-eligible) | Provider page and tool page drafted from `tools/TEMPLATE.mdx` and the Privy pages | Only after 50k monthly downloads, or if a maintainer asks |

Both patches apply cleanly to `langchain-ai/docs` main as of 2026-11-05 (`git apply --check`).

## Prerequisites

- [ ] `langchain-popnow` is published on PyPI. The issue form requires this, and the downloads badge reads the PyPI name. Right now `https://pypi.org/pypi/langchain-popnow/json` returns 404, so the name is free.
- [ ] The public repository `popnow-ai/langchain-popnow` exists, with this README.
- [ ] The `docs_url` page is live. The drafts use **`https://popnow.ai/docs/langchain`**, a partner docs page, which the form recommends and which sends visitors to popnow.ai. That page does not exist yet (404 on 2026-11-05). The repository README can be adapted for it. If the page is not ready in time, use `https://github.com/popnow-ai/langchain-popnow#readme` and update the patch URLs to match.
- [ ] Every link in the drafts returns 200. The docs CI runs a link checker (`make broken-links-with-anchors`).

## Policies to keep in mind

- Everything must be in English.
- AI-assisted contributions are allowed if every change is verified. The PR template asks you to disclose AI help in one plain sentence (for example "Prepared with Claude Code.") and to leave out "Generated with" badges.
- The provider description is one factual sentence of at most 160 characters, with no marketing language.
- Don't tag maintainers repeatedly.
