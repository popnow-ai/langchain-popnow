# Hosted guide (only if eligible)

Use these pages only when `langchain-popnow` has at least 50,000 monthly PyPI downloads, or when a LangChain maintainer marks it as featured or asks for a hosted page. Until then, automation closes PRs that add new MDX pages.

- `src/oss/python/integrations/tools/popnow.mdx` follows `src/oss/python/integrations/tools/TEMPLATE.mdx` and the Privy tool page, including its `integration:` frontmatter.
- `src/oss/python/integrations/providers/popnow.mdx` follows the Privy provider page.

The same PR also has to:

1. Remove the `PopNowToolkit` row from `scripts/data/integration_external_docs.yaml`, so the table does not show the package twice.
2. Point the provider card in `all_providers.mdx` to `/oss/integrations/providers/popnow`, and the "Media generation" row in `tools/index.mdx` to `/oss/integrations/tools/popnow`.
3. Add both pages to `src/docs.json` navigation, next to the other tool and provider pages.
4. Leave out `featured: true`. Only maintainers set it.

Before opening the PR, run every code block against the published package.
