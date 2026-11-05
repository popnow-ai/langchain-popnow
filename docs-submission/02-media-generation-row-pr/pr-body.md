Title: docs: add PopNow to the Media generation tools table

## Why
`langchain-popnow` (`PopNowToolkit`) is already listed in the "All tools and toolkits" table on the Python tools index through #<listing PR>, but it's missing from the "Media generation" category table. This PR adds one row so people browsing that category can find it.

## Changes
- `src/oss/python/integrations/tools/index.mdx`: add a PopNow row to the "Media generation" table, after Magic Hour.

## Review notes
- The row links to the partner docs page (https://popnow.ai/docs/langchain), like most external rows in this table.
- The pricing cell says pay-as-you-go in USD. PopNow has no free tier on the API, and every paid tool call is quoted before it runs.
- Affiliation: I maintain the PopNow integration.

## Related
- #<listing issue>
- #<listing PR>

## Checklist
- [x] I read the [contributing guidelines](https://docs.langchain.com/oss/contributing/documentation), including the [language policy](https://docs.langchain.com/oss/python/contributing/overview#language-policy)
- [ ] I checked the rendered pages locally with `docs dev` or on the preview deployment
- [x] I ran every changed code example, or none changed
- [ ] `make lint_prose` passes on the changed files
- [x] `make broken-links-with-anchors` passes, or no links, pages, or navigation changed
- [x] I updated `src/docs.json` navigation and redirects, or no page was added, moved, or removed

Prepared with Claude Code.
