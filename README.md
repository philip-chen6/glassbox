# glassbox

CLI-first tooling for generating an answer and inspecting internal transformer signals (attention, activation distributions, residual stream norms) from a prompt.

## What exists now

- `glassbox` CLI that accepts a prompt, generates an answer, and outputs JSON summaries.
- Hugging Face path (when `transformers` is installed) for real model introspection.
- Built-in toy causal transformer fallback for zero-setup local runs.
- Core tests for metrics and CLI behavior.

## Quick start

### 1) Run with the built-in toy model

```bash
PYTHONPATH=src python3 -m glassbox.cli --prompt "why attention works" --use-toy
```

### 2) Run with a Hugging Face model

Install optional dependency:

```bash
pip install ".[hf]"
```

Then run:

```bash
PYTHONPATH=src python3 -m glassbox.cli --prompt "why attention works" --model distilgpt2
```

If HF dependencies are missing or model loading fails, the CLI automatically falls back to the toy model and includes a `warning` in output.

### 3) Save output to JSON

```bash
PYTHONPATH=src python3 -m glassbox.cli \
  --prompt "explain residual stream norms" \
  --use-toy \
  --output outputs/report.json
```

### 4) Include raw activations/attention tensors

```bash
PYTHONPATH=src python3 -m glassbox.cli \
  --prompt "show me internals" \
  --use-toy \
  --max-new-tokens 24 \
  --include-hidden \
  --include-attention
```

## Output schema (high-level)

- `tokens`: token index, id, text.
- `answer_text`: generated answer text.
- `prompt_token_count` and `generated_token_count`: sequence split metadata.
- `layers[*].activation_distribution`: mean/std/min/max/p01/p99.
- `layers[*].residual_norms`: per-token L2 norms.
- `layers[*].attention`: entropy/max-weight summaries + strongest query/key edge per head.
- `hidden_states` (optional): raw per-layer hidden states `[layer][token][hidden_dim]`.
- `attentions` (optional): raw per-layer attention maps `[layer][head][query][key]`.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Viewer (Next.js)

Interactive v0 viewer lives in `/Users/philip/Projects/glassbox/viewer` and supports:
- in-app prompt runner (no manual JSON export required)
- JSON upload
- sample trace loading
- token timeline playback
- percentile-normalized attention heatmap
- layer overview heatmap (layers x tokens, residual norms)
- token connections with top-edge table
- residual norm flow plot

Run:

```bash
cd viewer
nvm use 22
pnpm install
pnpm dev
```

Then open `http://localhost:3000`.

Use the `Run Prompt` card in the UI:
- type a prompt
- default path is real model (`distilgpt2`); use toy only for debugging
- keep `Include hidden states` + `Include attention maps` enabled
- click `Run Prompt`
