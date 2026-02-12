export type TokenEntry = {
  index: number;
  id: number;
  text: string;
};

export type LayerSummary = {
  layer_index: number;
  residual_norms: number[];
  activation_distribution: {
    mean: number;
    std: number;
    min: number;
    max: number;
    p01: number;
    p99: number;
  };
};

export type TraceReport = {
  prompt: string;
  answer_text: string;
  model: string;
  source: string;
  warning?: string;
  num_tokens: number;
  prompt_token_count: number;
  generated_token_count: number;
  num_layers: number;
  tokens: TokenEntry[];
  layers: LayerSummary[];
  hidden_states?: number[][][];
  attentions?: number[][][][];
};
