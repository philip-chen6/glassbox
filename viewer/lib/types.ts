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

export type LayerInternals = {
  layer_index: number;
  residual_delta_norms: number[];
  residual_state_norms: number[];
  residual_delta_distribution: {
    mean: number;
    std: number;
    min: number;
    max: number;
    p01: number;
    p99: number;
  };
  attention_output_norms?: number[];
  attention_output_distribution?: {
    mean: number;
    std: number;
    min: number;
    max: number;
    p01: number;
    p99: number;
  };
  mlp_output_norms?: number[];
  mlp_output_distribution?: {
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
  layer_internals?: LayerInternals[];
  hidden_states?: number[][][];
  attentions?: number[][][][];
};
