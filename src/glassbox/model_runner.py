from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .analysis import summarize_layers, tensor_distribution_stats
from .toy_model import run_toy_forward


@dataclass
class ForwardArtifacts:
    source: str
    model_name: str
    token_ids: list[int]
    tokens: list[str]
    hidden_states: list[torch.Tensor]  # [seq, hidden] including embedding layer
    attentions: list[torch.Tensor]  # [heads, query, key]
    attention_outputs: list[torch.Tensor] | None  # [seq, hidden] per layer
    mlp_outputs: list[torch.Tensor] | None  # [seq, hidden] per layer
    prompt_token_count: int
    generated_token_count: int
    answer_text: str


def _format_token_piece(piece: str) -> str:
    """
    Make per-token display strings readable in the UI.

    We intentionally preserve token boundaries while making whitespace visible.
    """
    if piece == "":
        return "<empty>"
    piece = piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    piece = piece.replace(" ", "␠")
    if piece.strip("␠") == "":
        return "␠"
    return piece


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_first_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            found = _extract_first_tensor(item)
            if found is not None:
                return found
    return None


def _build_output_hook(storage: list[torch.Tensor | None]) -> Callable[..., None]:
    def _hook(_module: Any, _inputs: Any, output: Any) -> None:
        storage.append(_extract_first_tensor(output))

    return _hook


def _l2_per_token(tensor_2d: torch.Tensor) -> list[float]:
    norms = torch.linalg.vector_norm(tensor_2d.detach().float(), dim=-1)
    return [float(v.item()) for v in norms]


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
    attn_raw_outputs: list[torch.Tensor | None] = []
    mlp_raw_outputs: list[torch.Tensor | None] = []
    hook_handles: list[Any] = []

    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is not None:
        for block in blocks:
            if hasattr(block, "attn"):
                hook_handles.append(block.attn.register_forward_hook(_build_output_hook(attn_raw_outputs)))
            if hasattr(block, "mlp"):
                hook_handles.append(block.mlp.register_forward_hook(_build_output_hook(mlp_raw_outputs)))

    with torch.no_grad():
        try:
            out = model(
                input_ids=generated_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in hook_handles:
                handle.remove()

    if out.hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")
    if out.attentions is None or any(attn is None for attn in out.attentions):
        raise RuntimeError(
            "Model did not return attention tensors. Try eager attention implementation."
        )

    hidden_states = [h.squeeze(0).detach().cpu() for h in out.hidden_states]
    attentions = [a.squeeze(0).detach().cpu() for a in out.attentions if a is not None]
    attention_outputs: list[torch.Tensor] | None = None
    mlp_outputs: list[torch.Tensor] | None = None

    expected_layers = len(hidden_states) - 1
    if (
        expected_layers > 0
        and len(attn_raw_outputs) >= expected_layers
        and all(t is not None for t in attn_raw_outputs[:expected_layers])
    ):
        attention_outputs = []
        for tensor in attn_raw_outputs[:expected_layers]:
            assert tensor is not None
            if tensor.ndim == 3:
                attention_outputs.append(tensor.squeeze(0).detach().cpu())
            elif tensor.ndim == 2:
                attention_outputs.append(tensor.detach().cpu())

    if (
        expected_layers > 0
        and len(mlp_raw_outputs) >= expected_layers
        and all(t is not None for t in mlp_raw_outputs[:expected_layers])
    ):
        mlp_outputs = []
        for tensor in mlp_raw_outputs[:expected_layers]:
            assert tensor is not None
            if tensor.ndim == 3:
                mlp_outputs.append(tensor.squeeze(0).detach().cpu())
            elif tensor.ndim == 2:
                mlp_outputs.append(tensor.detach().cpu())

    prompt_token_count = int(input_ids.shape[1])
    ids = [int(v) for v in generated_ids[0].detach().cpu().tolist()]
    answer_ids = ids[prompt_token_count:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    tokens = [
        _format_token_piece(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in ids
    ]

    return ForwardArtifacts(
        source="huggingface",
        model_name=model_name,
        token_ids=ids,
        tokens=tokens,
        hidden_states=hidden_states,
        attentions=attentions,
        attention_outputs=attention_outputs,
        mlp_outputs=mlp_outputs,
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
                attention_outputs=toy.attention_outputs,
                mlp_outputs=toy.mlp_outputs,
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
                attention_outputs=toy.attention_outputs,
                mlp_outputs=toy.mlp_outputs,
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
                attention_outputs=toy.attention_outputs,
                mlp_outputs=toy.mlp_outputs,
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
    layer_internals: list[dict[str, Any]] = []
    for layer_idx in range(len(artifacts.hidden_states) - 1):
        prev = artifacts.hidden_states[layer_idx]
        curr = artifacts.hidden_states[layer_idx + 1]
        delta = curr - prev
        entry: dict[str, Any] = {
            "layer_index": layer_idx,
            "residual_delta_norms": _l2_per_token(delta),
            "residual_state_norms": _l2_per_token(curr),
            "residual_delta_distribution": tensor_distribution_stats(delta),
        }

        if artifacts.attention_outputs and layer_idx < len(artifacts.attention_outputs):
            attn_out = artifacts.attention_outputs[layer_idx]
            entry["attention_output_norms"] = _l2_per_token(attn_out)
            entry["attention_output_distribution"] = tensor_distribution_stats(attn_out)
        if artifacts.mlp_outputs and layer_idx < len(artifacts.mlp_outputs):
            mlp_out = artifacts.mlp_outputs[layer_idx]
            entry["mlp_output_norms"] = _l2_per_token(mlp_out)
            entry["mlp_output_distribution"] = tensor_distribution_stats(mlp_out)

        layer_internals.append(entry)

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
        "layer_internals": layer_internals,
    }
    if warning:
        report["warning"] = warning
    if include_hidden:
        report["hidden_states"] = [layer.tolist() for layer in artifacts.hidden_states]
    if include_attention:
        report["attentions"] = [layer.tolist() for layer in artifacts.attentions]
    return report
