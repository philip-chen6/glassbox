# Viewer V0 (Next.js)

## Built

- Next.js + React + TypeScript viewer under `/Users/philip/Projects/glassbox/viewer`.
- Client-side trace ingestion:
  - Run prompt directly in UI via Next API route (`/api/trace`) that calls Python CLI.
  - Upload any CLI output JSON file.
  - Load bundled sample trace from `public/samples/trace.json`.
- Interactive controls:
  - Layer selector.
  - Head selector (when attention tensors are available).
  - Token timeline scrubber.
  - Play/pause token playback with speed control.
  - Prompt options (toy/HF, max new tokens, include hidden, include attention).
- Visualization panels:
  - Attention heatmap (selected layer/head) with percentile normalization + contrast mapping.
  - Layer overview heatmap (residual norms across all layers and tokens).
  - Token connection arcs (top edges from selected query token) with ranked edge table.
  - Residual norm flow chart across layers for selected token.

## Key decisions

- React/Next app-router implementation with a single client component for v0 speed.
- No external charting/rendering libraries for first pass:
  - Canvas for heatmap and residual flow.
  - SVG for token edge arcs.
- Reused CLI JSON as the direct frontend contract (no extra backend required yet).
- Added a lightweight backend route only for local UX: users can run prompts from UI without manual file export/import.
- UI defaults to real model path (`distilgpt2`), with toy-model as an explicit debug fallback.
- Sample trace is generated from CLI with both `--include-hidden` and `--include-attention` to validate full-path UI.

## Notes

- This environment could not install pnpm packages due blocked npm registry access, so runtime verification of `pnpm dev` was not possible here.
- Python CLI outputs and tests remain fully verified.

## Next steps

1. Split visual panels into isolated React components with shared trace context.
2. Add animation interpolation (token-to-token transitions) rather than step-wise jumps.
3. Add camera-space 3D mode with Three.js while keeping 2D panels as source-of-truth diagnostics.
