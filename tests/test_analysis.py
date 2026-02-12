import torch

from glassbox.analysis import attention_summary, residual_stream_norms, tensor_distribution_stats


def test_tensor_distribution_stats_basic():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stats = tensor_distribution_stats(tensor)
    assert stats["mean"] == 2.5
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["p01"] <= stats["mean"] <= stats["p99"]


def test_residual_stream_norms_shape_check():
    bad = torch.zeros(1, 2, 3)
    try:
        residual_stream_norms(bad)
        assert False, "Expected ValueError for invalid shape"
    except ValueError:
        pass


def test_attention_summary_reports_top_edges():
    attn = torch.tensor(
        [
            [[0.6, 0.4], [0.1, 0.9]],
            [[0.2, 0.8], [0.3, 0.7]],
        ],
        dtype=torch.float32,
    )
    summary = attention_summary(attn)
    assert summary["global_max_weight"] == 0.9
    assert len(summary["top_edges"]) == 2
    assert all("head" in edge for edge in summary["top_edges"])
