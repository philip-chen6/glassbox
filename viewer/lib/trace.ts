import { TraceReport } from "./types";

function ensureNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || Number.isNaN(value)) {
    throw new Error(`Invalid ${name}: expected number`);
  }
  return value;
}

export function parseTrace(json: string): TraceReport {
  const parsed = JSON.parse(json) as Partial<TraceReport>;
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Invalid trace format");
  }

  if (!Array.isArray(parsed.tokens) || !Array.isArray(parsed.layers)) {
    throw new Error("Trace must include tokens and layers");
  }

  const numTokens = ensureNumber(parsed.num_tokens, "num_tokens");
  const numLayers = ensureNumber(parsed.num_layers, "num_layers");

  if (parsed.tokens.length !== numTokens) {
    throw new Error("tokens length does not match num_tokens");
  }
  if (parsed.layers.length !== numLayers) {
    throw new Error("layers length does not match num_layers");
  }

  return {
    prompt: String(parsed.prompt ?? ""),
    answer_text: String(parsed.answer_text ?? ""),
    model: String(parsed.model ?? ""),
    source: String(parsed.source ?? ""),
    warning: parsed.warning ? String(parsed.warning) : undefined,
    num_tokens: numTokens,
    prompt_token_count: ensureNumber(parsed.prompt_token_count, "prompt_token_count"),
    generated_token_count: ensureNumber(parsed.generated_token_count, "generated_token_count"),
    num_layers: numLayers,
    tokens: parsed.tokens,
    layers: parsed.layers,
    hidden_states: parsed.hidden_states,
    attentions: parsed.attentions
  };
}
