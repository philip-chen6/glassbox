# Viewer V0 (Next.js)

## Built

- Next.js + React + TypeScript viewer under `/Users/philip/Projects/glassbox/viewer`.
- Client-side trace ingestion:
  - Upload any CLI output JSON file.
  - Load bundled sample trace from `public/samples/trace.json`.
- Interactive controls:
  - Layer selector.
  - Head selector (when attention tensors are available).
  - Token timeline scrubber.
  - Play/pause token playback with speed control.
- Visualization panels:
  - Attention heatmap (selected layer/head).
  - Token connection arcs (top edges from selected query token).
  - Residual norm flow chart across layers for selected token.

## Key decisions

- React/Next app-router implementation with a single client component for v0 speed.
- No external charting/rendering libraries for first pass:
  - Canvas for heatmap and residual flow.
  - SVG for token edge arcs.
- Reused CLI JSON as the direct frontend contract (no extra backend required yet).
- Sample trace is generated from CLI with both `--include-hidden` and `--include-attention` to validate full-path UI.

## Notes

- This environment could not install pnpm packages due blocked npm registry access, so runtime verification of `pnpm dev` was not possible here.
- Python CLI outputs and tests remain fully verified.

## Next steps

1. Split visual panels into isolated React components with shared trace context.
2. Add animation interpolation (token-to-token transitions) rather than step-wise jumps.
3. Add camera-space 3D mode with Three.js while keeping 2D panels as source-of-truth diagnostics.
