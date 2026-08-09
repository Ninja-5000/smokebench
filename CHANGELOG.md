# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-08-10

### Added

- Exponential-backoff retry helper (`retry_request`) for transient HTTP/server errors (429, 5xx, timeouts), applied to judge calls while preserving the JSON-mode fallback.
- Pause/resume support for the benchmark runner via an `asyncio.Event`, with worker cancellation on the cancel action and live pause/resume status in the run screen.
- Two-phase benchmark pipeline: Phase 1 collects model outputs concurrently (bounded by `max_concurrency`), Phase 2 grades samples serially so the judge never receives overlapping requests.
- Per-benchmark token overrides plus a global default, wired through the config and benchmark engine.
- Persistence of the separate judge model and protocol across screens, with the detected protocol stored after a successful connection test and `openai` as the fallback default.
- TUI CSS files and dataset JSONL included as runtime package data in built distributions.

### Changed

- Redesigned the Advanced screen with a grid layout, inline reset, and global settings.
- Increased default `max_tokens` limits to better support thinking models.
- Refined endpoint connection testing and error handling, with a grid-based endpoint form and improved button styling for visibility and interaction.
- Simplified and clarified test assertions across the suite.

### Fixed

- Latency is now calculated from the response's reported `latency_s` instead of wall-clock time, so TPS reflects actual generation throughput; output tokens are estimated via tiktoken with a whitespace/word fallback when usage is omitted.
- Endpoint screen probe method now returns form values and updates state assignment in advance.
- HelpBar shortcuts and protocol value handling on the endpoint screen.
- Improved rubric clarity (e.g. semantic details for SQL query evaluation) and corrected expected values in benchmark datasets.
- Advanced screen separator line and padding.
- AppState attribute access type-checking.
- Corrected the PyPI version badge URL in the README.

[1.1.0]: https://github.com/Ninja-5000/smokebench/releases/tag/v1.1.0
