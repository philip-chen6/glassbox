from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ToyForwardOutput:
    token_ids: list[int]
    tokens: list[str]
    hidden_states: list[torch.Tensor]  # [seq, hidden] per layer, including embeddings
    attentions: list[torch.Tensor]  # [heads, query, key] per layer


class ToyTokenizer:
    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return tokens or ["<empty>"]

    def build_vocab(self, tokens: list[str]) -> dict[str, int]:
        vocab = {"<pad>": 0, "<unk>": 1}
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    def encode(self, text: str) -> tuple[list[int], list[str], dict[str, int]]:
        tokens = self.tokenize(text)
        vocab = self.build_vocab(tokens)
        ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
        return ids, tokens, vocab


class ToyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.ln1(x)
        attn_output, attn_weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


class ToyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([ToyBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, token_ids: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size, seq_len = token_ids.shape
        if batch_size != 1:
            raise ValueError(f"Toy model currently supports batch=1, got {batch_size}")

        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_emb(token_ids) + self.pos_emb(pos_ids)

        hidden_states: list[torch.Tensor] = [x.squeeze(0)]
        attentions: list[torch.Tensor] = []

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x, attn_weights = block(x, mask)
            hidden_states.append(x.squeeze(0))
            attentions.append(attn_weights.squeeze(0))

        return hidden_states, attentions


def run_toy_forward(prompt: str, seed: int = 7) -> ToyForwardOutput:
    tokenizer = ToyTokenizer()
    token_ids, tokens, vocab = tokenizer.encode(prompt)

    torch.manual_seed(seed)
    model = ToyCausalTransformer(vocab_size=len(vocab))
    model.eval()

    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        hidden_states, attentions = model(ids)

    return ToyForwardOutput(
        token_ids=token_ids,
        tokens=tokens,
        hidden_states=hidden_states,
        attentions=attentions,
    )
