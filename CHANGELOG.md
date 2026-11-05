# Changelog

## 0.1.0 (unreleased)

- First release.
- Paid tools: `PopNowImageGeneration`, `PopNowVideoGeneration`, `PopNowMusicGeneration`, `PopNowSpeech`, `PopNowCreateTask`. Each call is quoted in USD first and runs only after an explicit `max_usd` approval, or within `auto_approve_usd`. `max_usd_per_call` sets a hard per-call ceiling.
- Free tools: `PopNowListModels`, `PopNowGetTask`, `PopNowCheckBalance`.
- `PopNowToolkit` returns all tools with one shared configuration.
- Sync and async, built on the `popnow` Python SDK.
