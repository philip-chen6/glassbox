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


if __name__ == "__main__":
    unittest.main()
