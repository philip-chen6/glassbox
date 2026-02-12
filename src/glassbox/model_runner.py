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
    prompt_token_count: int
    generated_token_count: int
    answer_text: str


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_hf_forward(prompt: str, model_name: str, device: str, max_new_tokens: int) -> ForwardArtifacts:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.to(resolved)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(resolved)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(resolved)

    if max_new_tokens > 0:
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
    else:
        generated_ids = input_ids

    full_attention_mask = torch.ones_like(generated_ids, device=generated_ids.device)
    with torch.no_grad():
        out = model(
            input_ids=generated_ids,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if out.hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")
    if out.attentions is None or any(attn is None for attn in out.attentions):
        raise RuntimeError(
            "Model did not return attention tensors. Try eager attention implementation."
        )

    hidden_states = [h.squeeze(0).detach().cpu() for h in out.hidden_states]
    attentions = [a.squeeze(0).detach().cpu() for a in out.attentions if a is not None]

    prompt_token_count = int(input_ids.shape[1])
    ids = [int(v) for v in generated_ids[0].detach().cpu().tolist()]
    answer_ids = ids[prompt_token_count:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    return ForwardArtifacts(
        source="huggingface",
        model_name=model_name,
        token_ids=ids,
        tokens=tokens,
        hidden_states=hidden_states,
        attentions=attentions,
        prompt_token_count=prompt_token_count,
        generated_token_count=len(answer_ids),
        answer_text=answer_text,
    )


def run_forward(
    prompt: str,
    model_name: str,
    device: str = "auto",
    use_toy: bool = False,
    max_new_tokens: int = 24,
) -> tuple[ForwardArtifacts, str | None]:
    if use_toy:
        toy = run_toy_forward(prompt, max_new_tokens=max_new_tokens)
        return (
            ForwardArtifacts(
                source="toy",
                model_name="toy-causal-transformer",
                token_ids=toy.token_ids,
                tokens=toy.tokens,
                hidden_states=toy.hidden_states,
                attentions=toy.attentions,
                prompt_token_count=toy.prompt_token_count,
                generated_token_count=toy.generated_token_count,
                answer_text=toy.answer_text,
            ),
            None,
        )

    try:
        return run_hf_forward(prompt, model_name, device, max_new_tokens=max_new_tokens), None
    except ImportError:
        toy = run_toy_forward(prompt, max_new_tokens=max_new_tokens)
        return (
            ForwardArtifacts(
                source="toy",
                model_name="toy-causal-transformer",
                token_ids=toy.token_ids,
                tokens=toy.tokens,
                hidden_states=toy.hidden_states,
                attentions=toy.attentions,
                prompt_token_count=toy.prompt_token_count,
                generated_token_count=toy.generated_token_count,
                answer_text=toy.answer_text,
            ),
            "transformers is not installed; using toy fallback model",
        )
    except Exception as exc:
        toy = run_toy_forward(prompt, max_new_tokens=max_new_tokens)
        return (
            ForwardArtifacts(
                source="toy",
                model_name="toy-causal-transformer",
                token_ids=toy.token_ids,
                tokens=toy.tokens,
                hidden_states=toy.hidden_states,
                attentions=toy.attentions,
                prompt_token_count=toy.prompt_token_count,
                generated_token_count=toy.generated_token_count,
                answer_text=toy.answer_text,
            ),
            f"failed to run {model_name}: {exc}; using toy fallback model",
        )


def build_report(
    prompt: str,
    artifacts: ForwardArtifacts,
    warning: str | None = None,
    include_hidden: bool = False,
    include_attention: bool = False,
) -> dict[str, Any]:
    tokens = [
        {"index": idx, "id": token_id, "text": text}
        for idx, (token_id, text) in enumerate(zip(artifacts.token_ids, artifacts.tokens))
    ]
    layers = summarize_layers(artifacts.hidden_states, artifacts.attentions)

    report: dict[str, Any] = {
        "prompt": prompt,
        "answer_text": artifacts.answer_text,
        "model": artifacts.model_name,
        "source": artifacts.source,
        "num_tokens": len(tokens),
        "prompt_token_count": artifacts.prompt_token_count,
        "generated_token_count": artifacts.generated_token_count,
        "num_layers": len(layers),
        "tokens": tokens,
        "layers": layers,
    }
    if warning:
        report["warning"] = warning
    if include_hidden:
        report["hidden_states"] = [layer.tolist() for layer in artifacts.hidden_states]
    if include_attention:
        report["attentions"] = [layer.tolist() for layer in artifacts.attentions]
    return report
