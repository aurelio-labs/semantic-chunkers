"""Render ``benchmarks/results.json`` as a single self-contained HTML report.

    uv run python -m benchmarks.report [--results ...] [--out ...] [--history ...]

The report is one file with no external assets, no JavaScript and no extra
dependencies: open ``benchmarks/report.html`` in a browser. Charts are inline
SVG, hover values come from SVG ``<title>``, and every number is also in the
table at the bottom, so nothing is reachable only by hovering.

Each run is appended to ``benchmarks/history.jsonl`` (one JSON object per run,
keyed by commit) and the trend chart plots boundary F1 across those runs. With
a single run it draws that run's points and says so; it fills in as runs
accumulate.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).parent
DEFAULT_RESULTS = ROOT / "results.json"
DEFAULT_OUT = ROOT / "report.html"
DEFAULT_HISTORY = ROOT / "history.jsonl"

# Categorical slots from the data-viz reference palette, in the documented
# order. That order is the colour-vision-deficiency safety mechanism, not a
# style choice: do not re-order or extend it.
SERIES_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
SERIES_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
MAX_SERIES = len(SERIES_LIGHT)


def _series_vars(hexes: Sequence[str]) -> str:
    return "\n  ".join(f"--series-{i}: {h};" for i, h in enumerate(hexes, start=1))


# Chart geometry, in the SVG user units the viewBox scales from.
LABEL_W = 240
PLOT_W = 430
PAD_R = 58
CHART_W = LABEL_W + PLOT_W + PAD_R
TICK_BAND = 24
BAR_MAX_H = 18
GAP = 2  # the surface gap between touching marks


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def fmt_seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:,.0f} ms"
    return f"{value:,.2f} s"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def nice_ticks(scale_max: float, count: int = 4) -> list[float]:
    """Round tick values from 0 to at least ``scale_max``."""
    if scale_max <= 0:
        return [0.0, 1.0]
    raw = scale_max / count
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(
        m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw * 0.999999
    )
    ticks = []
    value = 0.0
    while value < scale_max - step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    ticks.append(round(value, 10))
    return ticks


def truncate(text: str, limit: int = 32) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rounded_bar(x: float, y: float, width: float, height: float) -> str:
    """A bar with a 4px rounded data-end and a square baseline end."""
    r = min(4.0, max(0.0, width), height / 2)
    if width <= 0.5:
        return f'<path d="M {x} {y} h 0.5 v {height} h -0.5 Z"/>'
    return (
        f'<path d="M {x} {y} h {width - r} a {r} {r} 0 0 1 {r} {r} '
        f'v {height - 2 * r} a {r} {r} 0 0 1 {-r} {r} h {-(width - r)} Z"/>'
    )


def _known(values: Iterable[float | None]) -> list[float]:
    """The measured values, dropping the ``None``s that mean "not measured"."""
    return [v for v in values if v is not None]


class Series:
    """A named row of values. ``None`` is "not measured", never zero."""

    def __init__(self, name: str, values: Sequence[float | None], slot: int) -> None:
        self.name = name
        self.values: list[float | None] = list(values)
        self.slot = slot


def hbar_chart(
    labels: Sequence[str],
    series: Sequence[Series],
    *,
    fmt: Callable[[float], str],
    scale_max: float | None = None,
    better: str = "up",
    label_values: str = "all",
    ranges: Sequence[tuple[float | None, float | None]] | None = None,
    markers: Sequence[float | None] | None = None,
    marker_slot: int = 2,
) -> str:
    """A horizontal bar chart.

    ``ranges`` draws a tone-on-tone spread line per row on top of the first
    series' bar; ``markers`` draws a tick per row in ``marker_slot``'s colour.
    A ``None`` value is not measured: it draws an em dash, not a zero bar.
    """
    highs = [max(_known(s.values), default=0.0) for s in series]
    if ranges:
        highs.append(max(_known(hi for _, hi in ranges), default=0.0))
    if markers:
        highs.append(max(_known(markers), default=0.0))
    top = scale_max if scale_max is not None else max(highs + [0.0])
    ticks = nice_ticks(top)
    top = max(ticks[-1], 1e-9)

    n = len(series)
    bar_h = min(BAR_MAX_H, max(6.0, (BAR_MAX_H * 2 - GAP * (n - 1)) / n))
    row_h = n * bar_h + (n - 1) * GAP + 18
    plot_h = row_h * len(labels)
    height = plot_h + TICK_BAND

    def x_of(value: float) -> float:
        return LABEL_W + max(0.0, min(1.0, value / top)) * PLOT_W

    parts: list[str] = [
        f'<svg class="chart" viewBox="0 0 {CHART_W} {height:.0f}" '
        f'width="100%" height="{height:.0f}" role="img" '
        f'preserveAspectRatio="xMinYMin meet">'
    ]
    for tick in ticks:
        x = x_of(tick)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{plot_h:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{plot_h + 15:.0f}" '
            f'text-anchor="middle">{esc(fmt(tick))}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{LABEL_W}" y1="0" x2="{LABEL_W}" y2="{plot_h:.1f}"/>'
    )

    best = {}
    for s in series:
        known = _known(s.values)
        if known:
            best[s.name] = min(known) if better == "down" else max(known)

    for row, label in enumerate(labels):
        y0 = row * row_h + 9
        parts.append(
            f'<text class="rowlabel" x="{LABEL_W - 10}" '
            f'y="{y0 + (row_h - 18) / 2 + 4:.1f}" text-anchor="end">'
            f"{esc(truncate(label))}<title>{esc(label)}</title></text>"
        )
        for i, s in enumerate(series):
            value = s.values[row]
            y = y0 + i * (bar_h + GAP)
            if value is None:
                # Not measured. An em dash, because a zero-length bar reads as
                # a real zero and this is the same lie `run.py` refuses to tell.
                parts.append(
                    f'<text class="value" x="{LABEL_W + 6}" '
                    f'y="{y + bar_h - 4:.1f}">—'
                    f"<title>{esc(f'{label} — {s.name}: not measured')}</title>"
                    f"</text>"
                )
                continue
            width = x_of(value) - LABEL_W
            title = f"{label} — {s.name}: {fmt(value)}"
            parts.append(
                f'<g fill="var(--series-{s.slot})">'
                f"{_rounded_bar(LABEL_W, y, width, bar_h)}"
                f"<title>{esc(title)}</title></g>"
            )
            show = label_values == "all" or (
                label_values == "best" and value == best.get(s.name)
            )
            if show:
                parts.append(
                    f'<text class="value" x="{x_of(value) + 6:.1f}" '
                    f'y="{y + bar_h - 4:.1f}">{esc(fmt(value))}</text>'
                )
        lo, hi = ranges[row] if ranges else (None, None)
        if lo is not None and hi is not None:
            mid = y0 + bar_h / 2
            x1, x2 = x_of(lo), x_of(hi)
            parts.append(
                f'<g class="spread"><line x1="{x1:.1f}" y1="{mid:.1f}" '
                f'x2="{x2:.1f}" y2="{mid:.1f}"/>'
                f'<line x1="{x1:.1f}" y1="{mid - 4:.1f}" x2="{x1:.1f}" '
                f'y2="{mid + 4:.1f}"/>'
                f'<line x1="{x2:.1f}" y1="{mid - 4:.1f}" x2="{x2:.1f}" '
                f'y2="{mid + 4:.1f}"/>'
                f"<title>{esc(f'{label} — p05 {fmt(lo)}, p95 {fmt(hi)}')}</title>"
                f"</g>"
            )
        marker = markers[row] if markers else None
        if marker is not None:
            x = x_of(marker)
            parts.append(
                f'<g class="marker" stroke="var(--series-{marker_slot})">'
                f'<line x1="{x:.1f}" y1="{y0 - 3:.1f}" x2="{x:.1f}" '
                f'y2="{y0 + bar_h + 3:.1f}"/>'
                f"<title>{esc(f'{label} — p95 {fmt(marker)}')}</title></g>"
            )
    parts.append("</svg>")
    return "".join(parts)


Point = tuple[int, float, float, float]  # run index, value, x, y


def _stretches(points: Sequence[Point]) -> list[list[Point]]:
    """Split points into runs that are consecutive on the x axis."""
    out: list[list[Point]] = []
    for point in points:
        if out and point[0] == out[-1][-1][0] + 1:
            out[-1].append(point)
        else:
            out.append([point])
    return out


def line_chart(
    x_labels: Sequence[str],
    series: Sequence[Series],
    *,
    fmt: Callable[[float], str],
    scale_max: float | None = None,
) -> str:
    """Boundary F1 across runs: one 2px line per variant, end dot and end label."""
    left, right_pad, top_pad = 44.0, 150.0, 12.0
    plot_h, height = 190.0, 190.0 + TICK_BAND + 12
    plot_w = CHART_W - left - right_pad
    highs = [v for s in series for v in _known(s.values)] or [0.0]
    top = scale_max if scale_max is not None else max(highs)
    ticks = nice_ticks(top)
    top = max(ticks[-1], 1e-9)
    n = max(1, len(x_labels) - 1)

    def x_of(i: int) -> float:
        if len(x_labels) == 1:
            return left + plot_w / 2
        return left + plot_w * i / n

    def y_of(value: float) -> float:
        return top_pad + (1 - max(0.0, min(1.0, value / top))) * (plot_h - top_pad)

    parts: list[str] = [
        f'<svg class="chart" viewBox="0 0 {CHART_W} {height:.0f}" '
        f'width="100%" height="{height:.0f}" role="img" '
        f'preserveAspectRatio="xMinYMin meet">'
    ]
    for tick in ticks:
        y = y_of(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{left + plot_w:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{esc(fmt(tick))}</text>'
        )
    for i, label in enumerate(x_labels):
        parts.append(
            f'<text class="tick" x="{x_of(i):.1f}" y="{plot_h + 16:.0f}" '
            f'text-anchor="middle">{esc(label)}</text>'
        )

    for s in series:
        colour = f"var(--series-{s.slot})"
        # Carry the run index with the point: a variant that did not exist in
        # an earlier run is a gap, not a zero, so the position in this list is
        # not the position on the x axis.
        points = [
            (i, v, x_of(i), y_of(v)) for i, v in enumerate(s.values) if v is not None
        ]
        # One polyline per unbroken stretch, so a gap is drawn as a gap.
        for stretch in _stretches(points):
            if len(stretch) > 1:
                d = " ".join(f"{x:.1f},{y:.1f}" for _, _, x, y in stretch)
                parts.append(f'<polyline class="line" stroke="{colour}" points="{d}"/>')
        for i, value, x, y in points:
            parts.append(
                f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}"/>'
                f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12">'
                f"<title>{esc(f'{s.name} @ {x_labels[i]}: {fmt(value)}')}"
                f"</title></circle>"
            )
        if points:
            x, y = points[-1][2], points[-1][3]
            parts.append(
                f'<text class="endlabel" x="{x + 10:.1f}" y="{y + 4:.1f}">'
                f"{esc(truncate(s.name, 22))}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def legend(items: Sequence[tuple[str, int]], spread: str = "") -> str:
    keys = [
        f'<span class="key"><i style="background:var(--series-{slot})"></i>'
        f"{esc(name)}</span>"
        for name, slot in items
    ]
    if spread:
        keys.append(f'<span class="key spread"><i></i>{esc(spread)}</span>')
    return f'<div class="legend">{"".join(keys)}</div>'


def card(title: str, subtitle: str, body: str) -> str:
    return (
        f'<section class="card"><h2>{esc(title)}</h2>'
        f'<p class="sub">{esc(subtitle)}</p>{body}</section>'
    )


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    runs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    return runs


def record_run(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Append this run to the history, replacing any earlier run of the same commit."""
    entry = {
        "commit": payload.get("commit", "unknown"),
        "timestamp": payload.get("timestamp", ""),
        "suites": [
            {
                "name": s["name"],
                "variant": s["variant"],
                "metrics": s.get("metrics", {}),
            }
            for s in payload.get("suites", [])
        ],
    }
    runs = [r for r in load_history(path) if r.get("commit") != entry["commit"]]
    runs.append(entry)
    path.write_text("".join(json.dumps(r) + "\n" for r in runs))
    return runs


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def metric_table(entries: Sequence[dict[str, Any]]) -> str:
    columns: list[str] = []
    for entry in entries:
        for key in entry.get("metrics", {}):
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    rows = []
    for entry in entries:
        metrics = entry.get("metrics", {})
        cells = "".join(f"<td>{esc(_cell(metrics.get(c)))}</td>" for c in columns)
        error = entry.get("error")
        note = f' <span class="err">{esc(error)}</span>' if error else ""
        rows.append(f"<tr><th>{esc(entry['variant'])}{note}</th>{cells}</tr>")
    return (
        '<div class="tablewrap"><table><thead><tr><th>variant</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


STYLE = (
    """
:root { color-scheme: light dark; }
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --spread: #0d366b;
  """
    + _series_vars(SERIES_LIGHT)
    + """
}
/* Dark is its own set of steps for the dark surface, not an automatic flip. */
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --spread: #cde2fb;
    """
    + _series_vars(SERIES_DARK)
    + """
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --spread: #cde2fb;
  """
    + _series_vars(SERIES_DARK)
    + """
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px;
  background: var(--page); color: var(--text-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.viz-root { max-width: 860px; margin: 0 auto; }
header h1 { font-size: 20px; margin: 0 0 4px; }
header .sub { margin: 0 0 24px; }
.sub { color: var(--text-secondary); font-size: 13px; margin: 0 0 16px; }
.hero { font-size: 48px; font-weight: 600; line-height: 1.1; margin: 0; }
.hero-label { color: var(--text-secondary); font-size: 13px; margin: 0 0 4px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px 22px; margin: 0 0 20px;
}
.card h2 { font-size: 15px; margin: 0 0 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 0 0 12px; }
.key { display: inline-flex; align-items: center; gap: 6px;
       color: var(--text-secondary); font-size: 12px; }
.key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.key.spread i { background: var(--spread); width: 14px; height: 2px; border-radius: 0; }
.chart { display: block; overflow: visible; }
.grid, .axis { stroke-width: 1; }
.grid { stroke: var(--grid); }
.axis { stroke: var(--axis); }
.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.rowlabel { fill: var(--text-secondary); font-size: 12px; }
.value { fill: var(--text-secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
.endlabel { fill: var(--text-secondary); font-size: 11px; }
.spread line { stroke: var(--spread); stroke-width: 2; stroke-linecap: butt; }
.marker line { stroke-width: 2; stroke-linecap: round; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--surface-1); stroke-width: 2; }
.hit { fill: transparent; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12px; width: 100%; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap;
         border-bottom: 1px solid var(--border); }
thead th { color: var(--text-muted); font-weight: 500; }
tbody th, thead th:first-child { text-align: left; font-weight: 500; }
td { font-variant-numeric: tabular-nums; color: var(--text-secondary); }
/* Status critical, reserved: never a series colour, and always with the
   message beside it so state never rides on colour alone. */
.err { color: #d03b3b; font-weight: 400; }
.note { color: var(--text-secondary); font-size: 12px; margin: 12px 0 0; }
"""
)


def build_html(
    payload: dict[str, Any], history: Sequence[dict[str, Any]], suite_name: str
) -> str:
    entries = [s for s in payload.get("suites", []) if s.get("name") == suite_name]
    ok = [e for e in entries if e.get("metrics") and not e.get("error")]
    failed = [e for e in entries if e.get("error")]
    ok.sort(key=lambda e: e["metrics"].get("boundary_f1", 0), reverse=True)
    names = [e["variant"] for e in ok]

    def col(key: str) -> list[float | None]:
        """A metric across the scored variants. Absent means not measured."""
        return [
            None if e["metrics"].get(key) is None else float(e["metrics"][key])
            for e in ok
        ]

    body: list[str] = []
    if ok:
        error_scale = max(_known(col("pk_p95") + col("windowdiff_p95")) + [0.0])
        top = ok[0]
        best_f1 = top["metrics"].get("boundary_f1")
        body.append(
            '<section class="card">'
            '<p class="hero-label">Best boundary F1</p>'
            f'<p class="hero">{esc("—" if best_f1 is None else fmt_rate(best_f1))}</p>'
            f'<p class="sub">{esc(top["variant"])} · '
            f"{esc(len(ok))} variants scored</p></section>"
        )
        body.append(
            card(
                "Boundary F1 — higher is better",
                "Bar is the mean across documents; the line is the p05–p95 spread, "
                "so a variant with a good average but a bad tail shows it here.",
                legend([("mean F1", 1)], spread="p05–p95 across documents")
                + hbar_chart(
                    names,
                    [Series("mean F1", col("boundary_f1"), 1)],
                    fmt=fmt_rate,
                    scale_max=1.0,
                    ranges=list(zip(col("boundary_f1_p05"), col("boundary_f1_p95"))),
                ),
            )
        )
        body.append(
            card(
                "Segmentation error — lower is better",
                "Pk and WindowDiff, median document and the p95 tail. Only the "
                "best value in each series is labelled; the rest are in the table.",
                legend([("p50", 1), ("p95", 2)])
                # One scale across both charts: they measure the same thing in
                # the same units, so a reader may compare them by bar length.
                + '<p class="sub">Pk</p>'
                + hbar_chart(
                    names,
                    [
                        Series("Pk p50", col("pk_p50"), 1),
                        Series("Pk p95", col("pk_p95"), 2),
                    ],
                    fmt=fmt_rate,
                    scale_max=error_scale,
                    better="down",
                    label_values="best",
                )
                + '<p class="sub">WindowDiff</p>'
                + hbar_chart(
                    names,
                    [
                        Series("WindowDiff p50", col("windowdiff_p50"), 1),
                        Series("WindowDiff p95", col("windowdiff_p95"), 2),
                    ],
                    fmt=fmt_rate,
                    scale_max=error_scale,
                    better="down",
                    label_values="best",
                ),
            )
        )
        body.append(
            card(
                "Time per document — lower is better",
                "Chunking one synthetic document, embeddings included. The bar is "
                "p50, the tick is p95.",
                legend([("p50", 1), ("p95", 2)])
                + hbar_chart(
                    names,
                    [Series("p50", col("doc_s_p50"), 1)],
                    fmt=fmt_seconds,
                    better="down",
                    markers=col("doc_s_p95"),
                ),
            )
        )
        body.append(
            card(
                "Encoder texts requested — lower is better",
                "What a user pays: every text the chunker asked the encoder for, "
                "cache hit or not, over the whole suite.",
                hbar_chart(
                    names,
                    [Series("texts requested", col("encoder_texts_requested"), 1)],
                    fmt=fmt_count,
                    better="down",
                ),
            )
        )

    body.append(trend_card(history, suite_name))

    if failed:
        listed = ", ".join(f"{e['variant']} ({e['error']})" for e in failed)
        body.append(
            f'<section class="card"><h2>Failed variants</h2>'
            f'<p class="sub">{esc(listed)}</p></section>'
        )

    body.append(
        card(
            "All metrics",
            "The table view: every number in the charts above, plus the ones that "
            "are not charted.",
            metric_table(entries),
        )
    )

    commit = payload.get("commit", "unknown")
    stamp = payload.get("timestamp", "")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>semantic-chunkers benchmark — {esc(commit)}</title>"
        f'<style>{STYLE}</style></head><body><div class="viz-root">'
        f"<header><h1>Benchmark — {esc(suite_name)}</h1>"
        f'<p class="sub">commit <code>{esc(commit)}</code> · {esc(stamp)}</p>'
        "</header>"
        f"{''.join(body)}</div></body></html>\n"
    )


def trend_card(history: Sequence[dict[str, Any]], suite_name: str) -> str:
    """Boundary F1 per variant across recorded runs."""
    runs = list(history)[-12:]
    variants: list[str] = []
    for run in runs:
        for entry in run.get("suites", []):
            if entry.get("name") == suite_name and entry["variant"] not in variants:
                variants.append(entry["variant"])
    if not runs or not variants:
        return card(
            "Boundary F1 over runs",
            "No history recorded yet. Each `make bench` appends a run to "
            "benchmarks/history.jsonl and this chart fills in.",
            "",
        )
    latest = {
        e["variant"]: e.get("metrics", {}).get("boundary_f1", 0.0)
        for e in runs[-1].get("suites", [])
        if e.get("name") == suite_name
    }
    variants.sort(key=lambda v: latest.get(v, 0.0), reverse=True)
    overflow = variants[MAX_SERIES:]
    variants = variants[:MAX_SERIES]

    series = []
    for slot, variant in enumerate(variants, start=1):
        values: list[float | None] = []
        for run in runs:
            hit = next(
                (
                    e
                    for e in run.get("suites", [])
                    if e.get("name") == suite_name and e["variant"] == variant
                ),
                None,
            )
            # A variant that did not exist in that run, or failed in it, is a
            # gap. Charting it as 0.0 would draw a cliff up from zero every
            # time a variant is added or renamed, which a sweep does often.
            f1 = (hit or {}).get("metrics", {}).get("boundary_f1")
            values.append(None if f1 is None else float(f1))
        series.append(Series(variant, values, slot))

    labels = [r.get("commit", "?") for r in runs]
    note = ""
    if len(runs) == 1:
        note = (
            '<p class="note">One run recorded so far, so there is nothing to '
            "compare against yet. The line appears from the second run on.</p>"
        )
    if overflow:
        note += (
            f'<p class="note">{esc(len(overflow))} further variant(s) are in the '
            "table only; the chart shows the eight with the highest current F1.</p>"
        )
    return card(
        "Boundary F1 over runs — higher is better",
        f"The last {len(runs)} recorded run(s), oldest first, labelled by commit.",
        legend([(truncate(s.name, 22), s.slot) for s in series])
        + line_chart(labels, series, fmt=fmt_rate, scale_max=1.0)
        + note,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument(
        "--no-append",
        action="store_true",
        help="render without recording this run in the history file",
    )
    ap.add_argument(
        "--suite",
        default=None,
        help="suite to chart (default: the first one in the results)",
    )
    args = ap.parse_args(argv)

    results = Path(args.results)
    if not results.exists():
        print(f"no results at {results}; run `make bench` first", file=sys.stderr)
        return 1
    payload = json.loads(results.read_text())
    suites = payload.get("suites", [])
    if not suites:
        print(f"{results} has no suites", file=sys.stderr)
        return 1
    suite_name = args.suite or suites[0]["name"]

    history_path = Path(args.history)
    history = (
        load_history(history_path)
        if args.no_append
        else record_run(history_path, payload)
    )

    out = Path(args.out)
    out.write_text(build_html(payload, history, suite_name))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
