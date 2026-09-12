import json

from benchmarks import report


def _payload(commit="abc1234", f1=(0.9, 0.5)):
    return {
        "schema": 1,
        "commit": commit,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "suites": [
            {
                "name": "synthetic-boundaries",
                "variant": "alpha",
                "metrics": {
                    "boundary_f1": f1[0],
                    "boundary_f1_p05": f1[0] - 0.2,
                    "boundary_f1_p50": f1[0],
                    "boundary_f1_p95": 1.0,
                    "pk_p50": 0.1,
                    "pk_p95": 0.3,
                    "windowdiff_p50": 0.12,
                    "windowdiff_p95": 0.35,
                    "doc_s_p50": 0.04,
                    "doc_s_p95": 0.09,
                    "encoder_texts_requested": 620,
                },
            },
            {
                "name": "synthetic-boundaries",
                "variant": "beta",
                "metrics": {
                    "boundary_f1": f1[1],
                    "boundary_f1_p05": 0.1,
                    "boundary_f1_p50": f1[1],
                    "boundary_f1_p95": 0.8,
                    "pk_p50": 0.3,
                    "pk_p95": 0.6,
                    "windowdiff_p50": 0.4,
                    "windowdiff_p95": 0.7,
                    "doc_s_p50": 0.2,
                    "doc_s_p95": 0.5,
                    "encoder_texts_requested": 1180,
                },
            },
        ],
    }


def test_nice_ticks_start_at_zero_and_cover_the_max():
    for scale_max in (1.0, 0.35, 1180, 0.0043):
        ticks = report.nice_ticks(scale_max)
        assert ticks[0] == 0
        assert ticks[-1] >= scale_max
        assert len(ticks) >= 2


def test_report_writes_self_contained_html(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_payload()))
    out = tmp_path / "report.html"
    history = tmp_path / "history.jsonl"

    assert (
        report.main(
            [
                "--results",
                str(results),
                "--out",
                str(out),
                "--history",
                str(history),
            ]
        )
        == 0
    )
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert "<svg" in page
    # no external assets, so the file opens offline
    assert "http://" not in page and "<script" not in page
    # the highest F1 leads, and every variant is in the table view
    assert "alpha" in page and "beta" in page
    assert page.index(">alpha<") < page.index(">beta<")
    assert "0.900" in page


def test_report_records_one_row_per_commit(tmp_path):
    results = tmp_path / "results.json"
    history = tmp_path / "history.jsonl"
    out = tmp_path / "report.html"
    args = ["--results", str(results), "--out", str(out), "--history", str(history)]

    results.write_text(json.dumps(_payload(commit="aaa")))
    report.main(args)
    results.write_text(json.dumps(_payload(commit="bbb", f1=(0.95, 0.4))))
    report.main(args)
    # re-running the same commit replaces its row rather than appending a duplicate
    report.main(args)

    runs = report.load_history(history)
    assert [r["commit"] for r in runs] == ["aaa", "bbb"]
    page = out.read_text()
    # both runs on the trend chart's x-axis, joined by a line per variant
    assert "aaa" in page and "bbb" in page
    assert page.count("<polyline") == 2
    assert "One run recorded so far" not in page


def test_trend_chart_says_so_when_there_is_only_one_run(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_payload()))
    out = tmp_path / "report.html"
    report.main(
        [
            "--results",
            str(results),
            "--out",
            str(out),
            "--history",
            str(tmp_path / "h.jsonl"),
        ]
    )
    page = out.read_text()
    assert "One run recorded so far" in page
    assert "<polyline" not in page


def test_report_skips_failed_variants_in_charts_but_keeps_them_in_the_table(tmp_path):
    payload = _payload()
    payload["suites"].append(
        {
            "name": "synthetic-boundaries",
            "variant": "broken",
            "metrics": {},
            "error": "TypeError: no such param",
        }
    )
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload))
    out = tmp_path / "report.html"
    report.main(
        [
            "--results",
            str(results),
            "--out",
            str(out),
            "--history",
            str(tmp_path / "h.jsonl"),
        ]
    )
    page = out.read_text()
    assert "Failed variants" in page
    assert "TypeError: no such param" in page
    assert "2 variants scored" in page


def test_report_exits_nonzero_without_results(tmp_path):
    assert (
        report.main(
            [
                "--results",
                str(tmp_path / "missing.json"),
                "--out",
                str(tmp_path / "o.html"),
                "--history",
                str(tmp_path / "h.jsonl"),
            ]
        )
        == 1
    )
