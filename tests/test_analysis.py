import unittest

import torch

from glassbox.analysis import attention_summary, residual_stream_norms, tensor_distribution_stats


class AnalysisTests(unittest.TestCase):
    def test_tensor_distribution_stats_basic(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        stats = tensor_distribution_stats(tensor)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertLessEqual(stats["p01"], stats["mean"])
        self.assertLessEqual(stats["mean"], stats["p99"])

    def test_residual_stream_norms_shape_check(self):
        bad = torch.zeros(1, 2, 3)
        with self.assertRaises(ValueError):
            residual_stream_norms(bad)

    def test_attention_summary_reports_top_edges(self):
        attn = torch.tensor(
            [
                [[0.6, 0.4], [0.1, 0.9]],
                [[0.2, 0.8], [0.3, 0.7]],
            ],
            dtype=torch.float32,
        )
        summary = attention_summary(attn)
        self.assertAlmostEqual(summary["global_max_weight"], 0.9, places=6)
        self.assertEqual(len(summary["top_edges"]), 2)
        self.assertTrue(all("head" in edge for edge in summary["top_edges"]))


if __name__ == "__main__":
    unittest.main()
