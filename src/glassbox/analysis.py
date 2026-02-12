from __future__ import annotations

from typing import Any

import torch


def tensor_distribution_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Return compact distribution stats for a tensor."""
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p01": 0.0,
            "p99": 0.0,
        }

    percentiles = torch.quantile(flat, torch.tensor([0.01, 0.99], device=flat.device))
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "p01": float(percentiles[0].item()),
        "p99": float(percentiles[1].item()),
    }


def residual_stream_norms(hidden_state: torch.Tensor) -> list[float]:
    """Per-token L2 norm for a layer hidden state."""
    if hidden_state.ndim != 2:
        raise ValueError(f"Expected [seq, hidden], got shape {tuple(hidden_state.shape)}")
    norms = torch.linalg.vector_norm(hidden_state.detach().float(), dim=-1)
    return [float(v.item()) for v in norms]


def attention_summary(attn: torch.Tensor) -> dict[str, Any]:
    """
    Summarize attention tensor.

    Expected shape: [heads, query_len, key_len]
    """
    if attn.ndim != 3:
        raise ValueError(f"Expected [heads, q, k], got shape {tuple(attn.shape)}")

    attn = attn.detach().float()
    eps = 1e-9
    probs = torch.clamp(attn, min=eps)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [heads, q]
    max_per_query = attn.max(dim=-1).values  # [heads, q]

    top_edges: list[dict[str, float | int]] = []
    for head_idx in range(attn.shape[0]):
        head = attn[head_idx]
        max_val = head.max()
        flat_idx = int(torch.argmax(head).item())
        q_idx = flat_idx // head.shape[1]
        k_idx = flat_idx % head.shape[1]
        top_edges.append(
            {
                "head": int(head_idx),
                "query_index": int(q_idx),
                "key_index": int(k_idx),
                "weight": float(max_val.item()),
            }
        )

    return {
        "mean_entropy": float(entropy.mean().item()),
        "mean_max_weight": float(max_per_query.mean().item()),
        "global_max_weight": float(attn.max().item()),
        "top_edges": top_edges,
    }


def summarize_layers(hidden_states: list[torch.Tensor], attentions: list[torch.Tensor]) -> list[dict[str, Any]]:
    """
    Build layer-level summaries.

    hidden_states is [layer_0_embed, layer_1, ... layer_n]
    attentions is [layer_1_attn, ... layer_n_attn]
    """
    summaries: list[dict[str, Any]] = []
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx]
        layer: dict[str, Any] = {
            "layer_index": layer_idx - 1,
            "residual_norms": residual_stream_norms(h),
            "activation_distribution": tensor_distribution_stats(h),
        }
        if layer_idx - 1 < len(attentions):
            layer["attention"] = attention_summary(attentions[layer_idx - 1])
        summaries.append(layer)
    return summaries
