"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { parseTrace } from "../lib/trace";
import { TraceReport } from "../lib/types";

const INITIAL_ERROR = "";
const TOP_EDGE_COUNT = 10;
const MODEL_PRESETS = [
  "HuggingFaceTB/SmolLM2-360M-Instruct",
  "Qwen/Qwen2.5-0.5B-Instruct",
  "distilgpt2"
];

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const pos = clamp(q, 0, 1) * (sorted.length - 1);
  const low = Math.floor(pos);
  const high = Math.ceil(pos);
  if (low === high) {
    return sorted[low];
  }
  const t = pos - low;
  return sorted[low] * (1 - t) + sorted[high] * t;
}

function toHeatColor(normalized: number): string {
  const t = clamp(normalized, 0, 1);
  const r = Math.round(32 + 223 * t);
  const g = Math.round(224 - 120 * t);
  const b = Math.round(227 - 180 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function shortToken(text: string, limit = 10): string {
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit - 1)}…`;
}

function dot(a: number[], b: number[]): number {
  const size = Math.min(a.length, b.length);
  let total = 0;
  for (let index = 0; index < size; index += 1) {
    total += a[index] * b[index];
  }
  return total;
}

function norm(a: number[]): number {
  return Math.sqrt(Math.max(dot(a, a), 0));
}

function cosine(a: number[], b: number[]): number {
  const denom = norm(a) * norm(b);
  if (denom <= 1e-12) {
    return 0;
  }
  return dot(a, b) / denom;
}

type FlowNeuron = {
  dim: number;
  value: number;
  absValue: number;
};

type FlowLink = {
  fromLayer: number;
  fromRank: number;
  toLayer: number;
  toRank: number;
  strength: number;
  positive: boolean;
};

export function TraceViewer() {
  const [trace, setTrace] = useState<TraceReport | null>(null);
  const [error, setError] = useState<string>(INITIAL_ERROR);
  const [promptInput, setPromptInput] = useState("hi how are you");
  const [modelInput, setModelInput] = useState("HuggingFaceTB/SmolLM2-360M-Instruct");
  const [maxNewTokens, setMaxNewTokens] = useState(24);
  const [useToyModel, setUseToyModel] = useState(false);
  const [includeHidden, setIncludeHidden] = useState(true);
  const [includeAttention, setIncludeAttention] = useState(true);
  const [isRunningPrompt, setIsRunningPrompt] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);

  const heatmapRef = useRef<HTMLCanvasElement | null>(null);
  const flowRef = useRef<HTMLCanvasElement | null>(null);
  const layerMapRef = useRef<HTMLCanvasElement | null>(null);
  const neuronFlowRef = useRef<HTMLCanvasElement | null>(null);

  const headCount = useMemo(() => {
    if (!trace?.attentions?.[selectedLayer]) {
      return 0;
    }
    return trace.attentions[selectedLayer].length;
  }, [trace, selectedLayer]);

  useEffect(() => {
    if (!trace) {
      return;
    }
    setSelectedLayer((current) => clamp(current, 0, trace.num_layers - 1));
    setSelectedTokenIndex((current) => clamp(current, 0, trace.num_tokens - 1));
  }, [trace]);

  useEffect(() => {
    setSelectedHead((current) => clamp(current, 0, Math.max(headCount - 1, 0)));
  }, [headCount]);

  useEffect(() => {
    if (!isPlaying || !trace) {
      return;
    }
    const intervalMs = Math.max(70, Math.floor(1000 / speed));
    const id = window.setInterval(() => {
      setSelectedTokenIndex((current) => (current + 1) % trace.num_tokens);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [isPlaying, speed, trace]);

  useEffect(() => {
    const canvas = heatmapRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!trace?.attentions?.[selectedLayer]?.[selectedHead]) {
      ctx.fillStyle = "#5c6c7a";
      ctx.font = "16px ui-sans-serif";
      ctx.fillText("Attention tensors not available.", 20, 36);
      return;
    }

    const matrix = trace.attentions[selectedLayer][selectedHead];
    const qLen = matrix.length;
    const kLen = matrix[0]?.length ?? 0;
    if (qLen === 0 || kLen === 0) {
      return;
    }

    const allValues = matrix.flat().slice().sort((a, b) => a - b);
    const lowQ = quantile(allValues, 0.05);
    const highQ = quantile(allValues, 0.995);
    const range = Math.max(highQ - lowQ, 1e-9);

    const pad = 28;
    const plotW = canvas.width - pad * 2;
    const plotH = canvas.height - pad * 2;
    const cellW = plotW / kLen;
    const cellH = plotH / qLen;

    for (let q = 0; q < qLen; q += 1) {
      for (let k = 0; k < kLen; k += 1) {
        const clipped = clamp(matrix[q][k], lowQ, highQ);
        const normalized = (clipped - lowQ) / range;
        const contrasted = Math.pow(normalized, 0.63);
        ctx.fillStyle = toHeatColor(contrasted);
        ctx.fillRect(pad + k * cellW, pad + q * cellH, cellW, cellH);
      }
    }

    ctx.strokeStyle = "#9cb0bf";
    ctx.strokeRect(pad, pad, plotW, plotH);

    ctx.strokeStyle = "#0e1e2b";
    ctx.lineWidth = 2;
    ctx.strokeRect(pad, pad + selectedTokenIndex * cellH, plotW, cellH);

    ctx.strokeStyle = "rgba(14,30,43,0.45)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(pad + selectedTokenIndex * cellW, pad, cellW, plotH);
  }, [trace, selectedLayer, selectedHead, selectedTokenIndex]);

  useEffect(() => {
    const canvas = flowRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!trace?.layers?.length) {
      return;
    }

    const values = trace.layers.map((layer) => layer.residual_norms[selectedTokenIndex] ?? 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = Math.max(max - min, 1e-9);

    const padX = 44;
    const padY = 26;
    const w = canvas.width - padX * 2;
    const h = canvas.height - padY * 2;

    ctx.strokeStyle = "#d5dce4";
    ctx.lineWidth = 1;
    ctx.strokeRect(padX, padY, w, h);

    ctx.strokeStyle = "#0f8f8a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = padX + (index / Math.max(values.length - 1, 1)) * w;
      const y = padY + h - ((value - min) / range) * h;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    values.forEach((value, index) => {
      const x = padX + (index / Math.max(values.length - 1, 1)) * w;
      const y = padY + h - ((value - min) / range) * h;
      ctx.fillStyle = index === selectedLayer ? "#ff7a59" : "#0f8f8a";
      ctx.beginPath();
      ctx.arc(x, y, index === selectedLayer ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [trace, selectedTokenIndex, selectedLayer]);

  useEffect(() => {
    const canvas = layerMapRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!trace?.layers?.length) {
      ctx.fillStyle = "#5c6c7a";
      ctx.font = "16px ui-sans-serif";
      ctx.fillText("Layer data not available.", 20, 36);
      return;
    }

    const layerCount = trace.layers.length;
    const tokenCount = trace.num_tokens;
    const matrix = trace.layers.map((layer) => layer.residual_norms);
    const allValues = matrix.flat().slice().sort((a, b) => a - b);
    const lowQ = quantile(allValues, 0.05);
    const highQ = quantile(allValues, 0.95);
    const range = Math.max(highQ - lowQ, 1e-9);

    const padLeft = 44;
    const padTop = 20;
    const plotW = canvas.width - padLeft - 16;
    const plotH = canvas.height - padTop - 22;
    const cellW = plotW / Math.max(tokenCount, 1);
    const cellH = plotH / Math.max(layerCount, 1);

    for (let layer = 0; layer < layerCount; layer += 1) {
      for (let token = 0; token < tokenCount; token += 1) {
        const value = matrix[layer]?.[token] ?? 0;
        const normalized = (clamp(value, lowQ, highQ) - lowQ) / range;
        const contrasted = Math.pow(normalized, 0.72);
        ctx.fillStyle = toHeatColor(contrasted);
        ctx.fillRect(padLeft + token * cellW, padTop + layer * cellH, cellW, cellH);
      }
    }

    ctx.strokeStyle = "#9cb0bf";
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    ctx.strokeStyle = "#0e1e2b";
    ctx.lineWidth = 1.8;
    ctx.strokeRect(padLeft, padTop + selectedLayer * cellH, plotW, cellH);
    ctx.strokeRect(padLeft + selectedTokenIndex * cellW, padTop, cellW, plotH);

    ctx.fillStyle = "#3f5368";
    ctx.font = "11px ui-sans-serif";
    ctx.fillText("L0", 8, padTop + 10);
    ctx.fillText(`L${layerCount - 1}`, 8, padTop + plotH - 6);
  }, [trace, selectedLayer, selectedTokenIndex]);

  const tokenConnections = useMemo(() => {
    if (!trace?.attentions?.[selectedLayer]?.[selectedHead]) {
      return [];
    }
    const matrix = trace.attentions[selectedLayer][selectedHead];
    const row = matrix[selectedTokenIndex] ?? [];
    const ranked = row
      .map((weight, keyIndex) => ({
        keyIndex,
        weight,
        token: trace.tokens[keyIndex]?.text ?? `[${keyIndex}]`
      }))
      .sort((a, b) => b.weight - a.weight);
    const dynamicCutoff = Math.max((ranked[0]?.weight ?? 0) * 0.18, 0.01);
    const points = ranked.filter((item) => item.weight >= dynamicCutoff).slice(0, TOP_EDGE_COUNT);
    return points;
  }, [trace, selectedLayer, selectedHead, selectedTokenIndex, trace?.tokens]);

  const flowValue = useMemo(() => {
    if (!trace?.layers?.[selectedLayer]) {
      return "-";
    }
    const value = trace.layers[selectedLayer].residual_norms[selectedTokenIndex] ?? 0;
    return `${value.toFixed(4)} (token=${selectedTokenIndex}, layer=${selectedLayer})`;
  }, [trace, selectedLayer, selectedTokenIndex]);

  const neuronFlowData = useMemo(() => {
    if (!trace?.hidden_states || trace.hidden_states.length < 3) {
      return null;
    }

    const neuronsPerLayer = 12;
    const maxLinksPerHop = 20;
    const vectors = trace.hidden_states.slice(1).map((layerState) => layerState[selectedTokenIndex] ?? []);
    if (vectors.length < 2 || vectors.some((vec) => vec.length === 0)) {
      return null;
    }

    const layers: FlowNeuron[][] = vectors.map((vector) =>
      vector
        .map((value, dim) => ({ dim, value, absValue: Math.abs(value) }))
        .sort((a, b) => b.absValue - a.absValue)
        .slice(0, neuronsPerLayer)
    );

    const allAbs = layers.flat().map((node) => node.absValue);
    const globalMaxAbs = Math.max(...allAbs, 1e-9);

    const links: FlowLink[] = [];
    for (let layer = 0; layer < layers.length - 1; layer += 1) {
      const left = layers[layer];
      const right = layers[layer + 1];
      const ranked: FlowLink[] = [];
      left.forEach((a, fromRank) => {
        right.forEach((b, toRank) => {
          const signed = a.value * b.value;
          ranked.push({
            fromLayer: layer,
            fromRank,
            toLayer: layer + 1,
            toRank,
            strength: Math.abs(signed),
            positive: signed >= 0
          });
        });
      });
      ranked.sort((a, b) => b.strength - a.strength);
      links.push(...ranked.slice(0, maxLinksPerHop));
    }

    const maxLinkStrength = Math.max(...links.map((link) => link.strength), 1e-9);
    return {
      layers,
      links,
      globalMaxAbs,
      maxLinkStrength
    };
  }, [trace, selectedTokenIndex]);

  useEffect(() => {
    const canvas = neuronFlowRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!neuronFlowData) {
      ctx.fillStyle = "#5c6c7a";
      ctx.font = "16px ui-sans-serif";
      ctx.fillText("Hidden states required for neuron-flow view.", 20, 36);
      return;
    }

    const { layers, links, globalMaxAbs, maxLinkStrength } = neuronFlowData;
    const layerCount = layers.length;
    const nodeCount = Math.max(...layers.map((nodes) => nodes.length), 1);
    const padX = 68;
    const padY = 40;
    const w = canvas.width - padX * 2;
    const h = canvas.height - padY * 2;

    const bg = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    bg.addColorStop(0, "#0b1020");
    bg.addColorStop(1, "#111827");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const nodePositions: Array<Array<{ x: number; y: number; node: FlowNeuron }>> = layers.map(
      (nodes, layerIndex) => {
        const x = padX + (layerIndex / Math.max(layerCount - 1, 1)) * w;
        return nodes.map((node, rank) => {
          const y = padY + ((rank + 0.5) / Math.max(nodeCount, 1)) * h;
          return { x, y, node };
        });
      }
    );

    links.forEach((link) => {
      const from = nodePositions[link.fromLayer]?.[link.fromRank];
      const to = nodePositions[link.toLayer]?.[link.toRank];
      if (!from || !to) {
        return;
      }
      const t = clamp(link.strength / maxLinkStrength, 0, 1);
      const alpha = 0.06 + t * 0.42;
      const width = 0.6 + t * 2.2;
      ctx.strokeStyle = link.positive
        ? `rgba(255,136,125,${alpha})`
        : `rgba(105,203,255,${alpha})`;
      ctx.lineWidth = width;
      ctx.beginPath();
      const midX = (from.x + to.x) / 2;
      const pull = Math.abs(to.x - from.x) * 0.24;
      ctx.moveTo(from.x, from.y);
      ctx.bezierCurveTo(midX - pull, from.y, midX + pull, to.y, to.x, to.y);
      ctx.stroke();
    });

    nodePositions.forEach((nodes, layerIndex) => {
      nodes.forEach(({ x, y, node }) => {
        const intensity = clamp(node.absValue / globalMaxAbs, 0, 1);
        const radius = 3 + intensity * 8;
        const isSelectedLayer = layerIndex === selectedLayer;
        const core = node.value >= 0 ? "#ffd9d2" : "#d7f1ff";
        const glow = node.value >= 0 ? "rgba(255,136,125,0.7)" : "rgba(105,203,255,0.72)";

        ctx.shadowBlur = isSelectedLayer ? 18 : 10;
        ctx.shadowColor = glow;
        ctx.fillStyle = core;
        ctx.beginPath();
        ctx.arc(x, y, isSelectedLayer ? radius + 1.1 : radius, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = 1;
    nodePositions.forEach((nodes, layerIndex) => {
      if (nodes.length === 0) {
        return;
      }
      const x = nodes[0].x;
      ctx.beginPath();
      ctx.moveTo(x, padY - 8);
      ctx.lineTo(x, canvas.height - padY + 8);
      ctx.stroke();

      ctx.fillStyle = layerIndex === selectedLayer ? "#f8fafc" : "#9fb2c8";
      ctx.font = layerIndex === selectedLayer ? "12px ui-sans-serif" : "11px ui-sans-serif";
      ctx.fillText(`L${layerIndex}`, x - 8, 20);
    });
  }, [neuronFlowData, selectedLayer]);

  function applyTrace(parsed: TraceReport): void {
    setTrace(parsed);
    setSelectedLayer(0);
    setSelectedHead(0);
    setSelectedTokenIndex(0);
    setIsPlaying(false);
  }

  async function runPrompt(): Promise<void> {
    setError(INITIAL_ERROR);
    if (!promptInput.trim()) {
      setError("Prompt is required.");
      return;
    }
    setIsRunningPrompt(true);
    try {
      const response = await fetch("/api/trace", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: promptInput,
          model: modelInput,
          useToy: useToyModel,
          includeHidden,
          includeAttention,
          maxNewTokens
        })
      });
      const payload = (await response.json()) as { trace?: unknown; error?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? `Request failed with status ${response.status}`);
      }
      if (!payload.trace) {
        throw new Error("Missing trace in response");
      }
      const parsed = parseTrace(JSON.stringify(payload.trace));
      applyTrace(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run prompt");
    } finally {
      setIsRunningPrompt(false);
    }
  }

  async function loadSample(): Promise<void> {
    setError(INITIAL_ERROR);
    try {
      const response = await fetch("/samples/trace.json");
      if (!response.ok) {
        throw new Error(`Failed to load sample: ${response.status}`);
      }
      const text = await response.text();
      const parsed = parseTrace(text);
      applyTrace(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sample trace");
    }
  }

  async function onFileChange(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setError(INITIAL_ERROR);
    try {
      const text = await file.text();
      const parsed = parseTrace(text);
      applyTrace(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse trace");
    }
  }

  return (
    <main className="app-shell">
      <div className="bg-orb orb-a" />
      <div className="bg-orb orb-b" />

      <header className="app-header">
        <div>
          <h1>glassbox trace viewer</h1>
          <p>Interactive playback for attention and residual stream flow.</p>
        </div>
        <div className="io-row">
          <label className="button ghost">
            <input type="file" accept="application/json" onChange={onFileChange} />
            Load Trace JSON
          </label>
          <button className="button" onClick={loadSample} type="button">
            Load Sample
          </button>
        </div>
      </header>

      <section className="card runner-card">
        <h2>Run Prompt</h2>
        <div className="runner-grid">
          <div className="control-group wide">
            <label htmlFor="prompt-input">Prompt</label>
            <textarea
              id="prompt-input"
              rows={3}
              value={promptInput}
              onChange={(event) => setPromptInput(event.target.value)}
              placeholder="Type prompt text..."
            />
          </div>

          <div className="control-group">
            <label htmlFor="model-input">Model</label>
            <input
              id="model-input"
              type="text"
              list="model-presets"
              value={modelInput}
              onChange={(event) => setModelInput(event.target.value)}
              disabled={useToyModel}
            />
            <datalist id="model-presets">
              {MODEL_PRESETS.map((model) => (
                <option key={model} value={model} />
              ))}
            </datalist>
          </div>

          <div className="control-group">
            <label htmlFor="new-token-input">Max New Tokens</label>
            <input
              id="new-token-input"
              type="number"
              min={0}
              max={256}
              value={maxNewTokens}
              onChange={(event) => setMaxNewTokens(Math.max(0, Number(event.target.value) || 0))}
            />
          </div>

          <div className="control-group">
            <label>Options</label>
            <label className="toggle-inline">
              <input
                type="checkbox"
                checked={useToyModel}
                onChange={(event) => setUseToyModel(event.target.checked)}
              />
              Force toy model (debug)
            </label>
            <label className="toggle-inline">
              <input
                type="checkbox"
                checked={includeHidden}
                onChange={(event) => setIncludeHidden(event.target.checked)}
              />
              Include hidden states
            </label>
            <label className="toggle-inline">
              <input
                type="checkbox"
                checked={includeAttention}
                onChange={(event) => setIncludeAttention(event.target.checked)}
              />
              Include attention maps
            </label>
          </div>

          <div className="control-group">
            <label>Run</label>
            <button
              className="button"
              onClick={runPrompt}
              type="button"
              disabled={isRunningPrompt || !promptInput.trim()}
            >
              {isRunningPrompt ? "Running..." : "Run Prompt"}
            </button>
          </div>
        </div>
      </section>

      <section className="card summary-card">
        <div className="meta-grid">
          <div>
            <h2>Prompt</h2>
            <p>{trace?.prompt ?? "No trace loaded."}</p>
          </div>
          <div>
            <h2>Answer</h2>
            <p>{trace?.answer_text || "-"}</p>
          </div>
        </div>

        <div className="stats-row">
          <span className="pill">model: {trace?.model ?? "-"}</span>
          <span className="pill">tokens: {trace?.num_tokens ?? "-"}</span>
          <span className="pill">layers: {trace?.num_layers ?? "-"}</span>
          <span className="pill">generated: {trace?.generated_token_count ?? "-"}</span>
          <span className="pill">
            attentions: {trace?.attentions ? "included" : "missing"}
          </span>
          <span className="pill">
            hidden: {trace?.hidden_states ? "included" : "missing"}
          </span>
        </div>

        {trace?.warning ? <p className="warning">{trace.warning}</p> : null}
        {error ? <p className="warning">{error}</p> : null}
      </section>

      <section className="card controls-card">
        <div className="control-group">
          <label htmlFor="layer-select">Layer</label>
          <select
            id="layer-select"
            value={selectedLayer}
            onChange={(event) => setSelectedLayer(Number(event.target.value))}
          >
            {Array.from({ length: trace?.num_layers ?? 0 }).map((_, index) => (
              <option value={index} key={`layer-${index}`}>
                Layer {index}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="head-select">Head</label>
          <select
            id="head-select"
            value={selectedHead}
            onChange={(event) => setSelectedHead(Number(event.target.value))}
            disabled={headCount === 0}
          >
            {Array.from({ length: headCount }).map((_, index) => (
              <option value={index} key={`head-${index}`}>
                Head {index}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group wide">
          <label htmlFor="token-slider">Token Timeline</label>
          <input
            id="token-slider"
            type="range"
            min={0}
            max={Math.max((trace?.num_tokens ?? 1) - 1, 0)}
            value={selectedTokenIndex}
            onChange={(event) => setSelectedTokenIndex(Number(event.target.value))}
          />
        </div>

        <div className="control-group">
          <label htmlFor="speed-slider">Speed</label>
          <input
            id="speed-slider"
            type="range"
            min={1}
            max={8}
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Playback</label>
          <button className="button" onClick={() => setIsPlaying((v) => !v)} type="button">
            {isPlaying ? "Pause" : "Play"}
          </button>
        </div>
      </section>

      <section className="card">
        <h2>Token Timeline</h2>
        <div className="token-rail">
          {trace?.tokens.map((token, index) => {
            const className =
              index === selectedTokenIndex
                ? "token-pill active"
                : index < selectedTokenIndex
                  ? "token-pill before"
                  : "token-pill";
            return (
              <button
                key={`token-${token.index}-${token.id}`}
                className={className}
                onClick={() => setSelectedTokenIndex(index)}
                type="button"
              >
                {token.text}
              </button>
            );
          })}
        </div>
      </section>

      <section className="panel-grid">
        <article className="card panel">
          <h2>Attention Heatmap</h2>
          <p className="subtitle">
            Layer {selectedLayer}, head {selectedHead} (percentile-normalized)
          </p>
          <div className="canvas-wrap">
            <canvas ref={heatmapRef} width={640} height={640} />
          </div>
          <div className="legend-row">
            <span>Low</span>
            <div className="legend-gradient" />
            <span>High</span>
          </div>
        </article>

        <article className="card panel">
          <h2>Layer Overview</h2>
          <p className="subtitle">Residual norm map (layers x tokens)</p>
          <div className="canvas-wrap">
            <canvas ref={layerMapRef} width={800} height={300} />
          </div>
        </article>

        <article className="card panel">
          <h2>Token Connections</h2>
          <p className="subtitle">Top edges from query token index {selectedTokenIndex}</p>
          <div className="svg-wrap">
            <svg viewBox="0 0 800 320" preserveAspectRatio="xMidYMid meet">
              <circle cx={400} cy={52} r={10} fill="#0f8f8a" />
              <text x={400} y={28} textAnchor="middle" fontSize={12} fill="#26435c">
                query[{selectedTokenIndex}]
              </text>
              <text x={400} y={78} textAnchor="middle" fontSize={13} fill="#17344a" fontWeight="600">
                {shortToken(trace?.tokens[selectedTokenIndex]?.text ?? "-", 20)}
              </text>

              {tokenConnections.map(({ keyIndex, weight, token }, index) => {
                if (!trace?.num_tokens || tokenConnections.length === 0) {
                  return null;
                }
                const x1 = 400;
                const x2 = 70 + (index / Math.max(tokenConnections.length - 1, 1)) * 660;
                const y2 = 228;
                const midY = 112 + Math.abs(x2 - x1) * 0.16;
                const opacity = 0.24 + weight * 0.76;
                return (
                  <g key={`edge-${keyIndex}`}>
                    <path
                      d={`M ${x1} 66 Q ${(x1 + x2) / 2} ${midY} ${x2} ${y2 - 10}`}
                      stroke={`rgba(255,122,89,${opacity})`}
                      strokeWidth={1.5 + weight * 5}
                      fill="none"
                    />
                    <circle cx={x2} cy={y2} r={6} fill="#157f9f" />
                    <text x={x2} y={247} textAnchor="middle" fontSize={11} fill="#26435c">
                      {shortToken(token, 11)}
                    </text>
                    <text x={x2} y={264} textAnchor="middle" fontSize={10} fill="#5a7084">
                      {keyIndex}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="edge-table-wrap">
            <table className="edge-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>key token</th>
                  <th>index</th>
                  <th>weight</th>
                </tr>
              </thead>
              <tbody>
                {tokenConnections.map((edge, index) => (
                  <tr key={`edge-row-${edge.keyIndex}`}>
                    <td>{index + 1}</td>
                    <td>{edge.token}</td>
                    <td>{edge.keyIndex}</td>
                    <td>{edge.weight.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card panel span-2">
          <h2>Neuron Flow Corridor</h2>
          <p className="subtitle">
            3b1b-style layered neuron activation + signed co-activation links (selected token)
          </p>
          <div className="canvas-wrap">
            <canvas ref={neuronFlowRef} width={1100} height={380} />
          </div>
          <div className="flow-value">
            Selected token: {trace?.tokens[selectedTokenIndex]?.text ?? "-"} (index {selectedTokenIndex})
          </div>
        </article>

        <article className="card panel span-2">
          <h2>Residual Norm Flow</h2>
          <p className="subtitle">Selected token through all layers</p>
          <div className="canvas-wrap flow-wrap">
            <canvas ref={flowRef} width={800} height={320} />
          </div>
          <div className="flow-value">{flowValue}</div>
        </article>
      </section>
    </main>
  );
}
