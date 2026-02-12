# Glassbox Agent Notes

## Project goal

Build an internal-visualization pipeline for LLM inference:

1. Prompt input.
2. Forward pass through model internals.
3. Structured signals for visualization (attention, activations, residual stream norms).
4. Later: 2D/3D interactive rendering.

## Current implementation contract

- Primary interface is CLI.
- JSON output is the integration boundary for any UI renderer.
- CLI supports:
  - Hugging Face model introspection (when available).
  - Toy model fallback for local deterministic execution.

## Development rules (project-specific)

- Keep CLI and data contract stable while iterating visualization layers.
- Prefer adding fields over renaming existing output keys.
- Add tests for core math/aggregation logic whenever adding new metrics.
- Keep outputs interpretable and compact by default; add explicit flags for large dumps.

## Suggested immediate roadmap

1. Add optional raw tensor dumps with per-layer controls.
2. Add token-transition graph extraction for animation systems.
3. Add lightweight local web viewer (Three.js) consuming the JSON contract.
