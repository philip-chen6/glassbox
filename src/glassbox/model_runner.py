from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .analysis import summarize_layers
from .toy_model import run_toy_forward


@dataclass
class ForwardArtifacts:
    source: str
    model_name: str
    token_ids: list[int]
    tokens: list[str]
    hidden_states: list[torch.Tensor]  # [seq, hidden] including embedding layer
    attentions: list[torch.Tensor]  # [heads, query, key]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_hf_forward(prompt: str, model_name: str, device: str) -> ForwardArtifacts:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(resolved)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(resolved)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(resolved)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = [h.squeeze(0).detach().cpu() for h in out.hidden_states]
    attentions = [a.squeeze(0).detach().cpu() for a in out.attentions]

    ids = [int(v) for v in input_ids[0].detach().cpu().tolist()]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    return ForwardArtifacts(
        source="huggingface",
        model_name=model_name,
        token_ids=ids,
        tokens=tokens,
        hidden_states=hidden_states,
        attentions=attentions,
    )


def run_forward(
    prompt: str,
    model_name: str,
    device: str = "auto",
    use_toy: bool = False,
) -> tuple[ForwardArtifacts, str | None]:
    if use_toy:
        toy = run_toy_forward(prompt)
        return (
            ForwardArtifacts(
                source="toy",
                model_name="toy-causal-transformer",
                token_ids=toy.token_ids,
                tokens=toy.tokens,
                hidden_states=toy.hidden_states,
                attentions=toy.attentions,
            ),
            None,
        )

    try:
        return run_hf_forward(prompt, model_name, device), None
    except ImportError:
        toy = run_toy_forward(prompt)
        return (
            ForwardArtifacts(
                source="toy",
                model_name="toy-causal-transformer",
                token_ids=toy.token_ids,
                tokens=toy.tokens,
                hidden_states=toy.hidden_states,
                attentions=toy.attentions,
            ),
            "transformers is not installed; using toy fallback model",
        )
    except Exception as exc:
        toy = run_toy_forward(prompt)
        return (
            ForwardArtifacts(
                source="toy",
                model_name="toy-causal-transformer",
                token_ids=toy.token_ids,
                tokens=toy.tokens,
                hidden_states=toy.hidden_states,
                attentions=toy.attentions,
            ),
            f"failed to run {model_name}: {exc}; using toy fallback model",
        )


def build_report(prompt: str, artifacts: ForwardArtifacts, warning: str | None = None) -> dict[str, Any]:
    tokens = [
        {"index": idx, "id": token_id, "text": text}
        for idx, (token_id, text) in enumerate(zip(artifacts.token_ids, artifacts.tokens))
    ]
    layers = summarize_layers(artifacts.hidden_states, artifacts.attentions)

    report: dict[str, Any] = {
        "prompt": prompt,
        "model": artifacts.model_name,
        "source": artifacts.source,
        "num_tokens": len(tokens),
        "num_layers": len(layers),
        "tokens": tokens,
        "layers": layers,
    }
    if warning:
        report["warning"] = warning
    return report
