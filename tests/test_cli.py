import json
import tempfile
import unittest
from pathlib import Path

from glassbox.cli import main


class CLITests(unittest.TestCase):
    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "report.json"
            exit_code = main(
                [
                    "--prompt",
                    "hello world",
                    "--use-toy",
                    "--output",
                    str(out_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(report["source"], "toy")
            self.assertGreaterEqual(report["num_tokens"], 1)
            self.assertGreaterEqual(report["num_layers"], 1)
            self.assertEqual(len(report["layers"]), report["num_layers"])

    def test_cli_can_include_raw_hidden_states(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "report_with_hidden.json"
            exit_code = main(
                [
                    "--prompt",
                    "token flow",
                    "--use-toy",
                    "--include-hidden",
                    "--output",
                    str(out_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertIn("hidden_states", report)
            self.assertEqual(len(report["hidden_states"]), report["num_layers"] + 1)


if __name__ == "__main__":
    unittest.main()
