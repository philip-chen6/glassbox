"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { parseTrace } from "../lib/trace";
import { TraceReport } from "../lib/types";

const INITIAL_ERROR = "";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

function toHeatColor(normalized: number): string {
  const t = clamp(normalized, 0, 1);
  const r = Math.round(32 + 223 * t);
  const g = Math.round(224 - 120 * t);
  const b = Math.round(227 - 180 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function TraceViewer(): JSX.Element {
  const [trace, setTrace] = useState<TraceReport | null>(null);
  const [error, setError] = useState<string>(INITIAL_ERROR);
  const [promptInput, setPromptInput] = useState("hi how are you");
  const [modelInput, setModelInput] = useState("distilgpt2");
  const [maxNewTokens, setMaxNewTokens] = useState(24);
  const [useToyModel, setUseToyModel] = useState(true);
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

    const allValues = matrix.flat();
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = Math.max(max - min, 1e-9);

    const pad = 28;
    const plotW = canvas.width - pad * 2;
    const plotH = canvas.height - pad * 2;
    const cellW = plotW / kLen;
    const cellH = plotH / qLen;

    for (let q = 0; q < qLen; q += 1) {
      for (let k = 0; k < kLen; k += 1) {
        const value = matrix[q][k];
        const normalized = (value - min) / range;
        ctx.fillStyle = toHeatColor(normalized);
        ctx.fillRect(pad + k * cellW, pad + q * cellH, cellW, cellH);
      }
    }

    ctx.strokeStyle = "#9cb0bf";
    ctx.strokeRect(pad, pad, plotW, plotH);

    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 2;
    ctx.strokeRect(pad, pad + selectedTokenIndex * cellH, plotW, cellH);
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

  const tokenConnections = useMemo(() => {
    if (!trace?.attentions?.[selectedLayer]?.[selectedHead]) {
      return [];
    }
    const matrix = trace.attentions[selectedLayer][selectedHead];
    const row = matrix[selectedTokenIndex] ?? [];
    const points = row
      .map((weight, keyIndex) => ({ keyIndex, weight }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 8);
    return points;
  }, [trace, selectedLayer, selectedHead, selectedTokenIndex]);

  const flowValue = useMemo(() => {
    if (!trace?.layers?.[selectedLayer]) {
      return "-";
    }
    const value = trace.layers[selectedLayer].residual_norms[selectedTokenIndex] ?? 0;
    return `${value.toFixed(4)} (token=${selectedTokenIndex}, layer=${selectedLayer})`;
  }, [trace, selectedLayer, selectedTokenIndex]);

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
              value={modelInput}
              onChange={(event) => setModelInput(event.target.value)}
              disabled={useToyModel}
            />
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
              Use toy model
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
          <p className="subtitle">Layer {selectedLayer}, head {selectedHead}</p>
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
          <h2>Token Connections</h2>
          <p className="subtitle">Top edges from query token index {selectedTokenIndex}</p>
          <div className="svg-wrap">
            <svg viewBox="0 0 800 320" preserveAspectRatio="xMidYMid meet">
              {trace?.tokens.map((token, index) => {
                const x = 40 + (index / Math.max((trace.num_tokens ?? 1) - 1, 1)) * 720;
                return (
                  <g key={`label-${token.index}`}>
                    <circle cx={x} cy={42} r={index === selectedTokenIndex ? 8 : 5} fill="#0f8f8a" />
                    <text x={x} y={72} textAnchor="middle" fontSize={12} fill="#26435c">
                      {token.text.slice(0, 6)}
                    </text>
                  </g>
                );
              })}

              {tokenConnections.map(({ keyIndex, weight }) => {
                if (!trace?.num_tokens) {
                  return null;
                }
                const x1 = 40 + (selectedTokenIndex / Math.max(trace.num_tokens - 1, 1)) * 720;
                const x2 = 40 + (keyIndex / Math.max(trace.num_tokens - 1, 1)) * 720;
                const midY = 170 - Math.abs(x2 - x1) * 0.12;
                const opacity = 0.2 + weight * 0.8;
                return (
                  <path
                    key={`edge-${keyIndex}`}
                    d={`M ${x1} 96 Q ${(x1 + x2) / 2} ${midY} ${x2} 96`}
                    stroke={`rgba(255,122,89,${opacity})`}
                    strokeWidth={1.5 + weight * 4}
                    fill="none"
                  />
                );
              })}
            </svg>
          </div>
        </article>

        <article className="card panel">
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
