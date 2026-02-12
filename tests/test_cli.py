import json

from glassbox.cli import main


def test_cli_writes_report(tmp_path):
    out_file = tmp_path / "report.json"
    exit_code = main(
        [
            "--prompt",
            "hello world",
            "--use-toy",
            "--output",
            str(out_file),
        ]
    )
    assert exit_code == 0
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["source"] == "toy"
    assert report["num_tokens"] >= 1
    assert report["num_layers"] >= 1
    assert len(report["layers"]) == report["num_layers"]
