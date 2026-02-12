# CLI Activation Pipeline

## Built

- Python package scaffold with CLI entrypoint (`glassbox` / `python -m glassbox.cli`).
- Unified forward runner:
  - Hugging Face causal LM path (optional dependency on `transformers`).
  - Toy causal transformer fallback for local-first execution.
- Layer analytics for:
  - Activation distributions.
  - Residual stream L2 norms per token.
  - Attention pattern summaries (entropy, max weights, strongest edge per head).
- Test coverage for analytics + CLI output generation.

## Key decisions

- CLI-first implementation to validate signal extraction before any visualization UI.
- JSON output as canonical contract for future 2D/3D renderer ingestion.
- Soft dependency on Hugging Face:
  - Tool still runs without `transformers`.
  - Explicit warning is attached when fallback occurs.
- Toy model kept intentionally small and deterministic (`seed=7`) to provide stable local behavior.

## Next steps toward visualization

- Add `--dump-raw` mode for optional full tensor exports (with size guards).
- Add temporal token flow structure for animation timelines.
- Add web transport format (binary + metadata) for fast 3D particle rendering.
- Add per-layer/per-head filtering flags for targeted inspection.
