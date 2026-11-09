"""Render a run's metrics log as one self-contained HTML file.

Reads only <run_dir>/metrics.jsonl (written by ml_system.metrics.MetricsLogger)
and writes <run_dir>/dashboard.html. Pure HTML + CSS + inline SVG: no
JavaScript, no external files, no extra dependencies. It therefore looks the
same in a browser, in Jupyter (even an untrusted notebook, which strips
scripts) and in Colab. Hover a point on the chart for its value (native SVG
tooltips).
"""

import html
import json
import math
from pathlib import Path

METRICS_FILE = "metrics.jsonl"
DASHBOARD_FILE = "dashboard.html"

# Target number of slates per rolling-CTR window when the caller does not
# choose a window in ticks.
ROLLING_SESSIONS = 250

# Chart geometry (SVG user units; the SVG scales to its container width).
WIDTH, HEIGHT = 760, 280
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 44, 92, 16, 30
MAX_POINTS = 400


def read_log(run_dir):
    path = Path(run_dir) / METRICS_FILE
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rolling(ticks, window):
    """Trailing-window CTR over `window` ticks: list of (tick, ctr|None)."""
    series = []
    shown_sum = clicks_sum = 0

    for i, row in enumerate(ticks):
        shown_sum += row["shown"]
        clicks_sum += row["clicks"]

        if i >= window:
            shown_sum -= ticks[i - window]["shown"]
            clicks_sum -= ticks[i - window]["clicks"]

        if i + 1 >= window and shown_sum:
            series.append((row["tick"], clicks_sum / shown_sum))

    return series


def summarize(records, window=None):
    """Everything the page shows, computed from the log records."""
    start = next((r for r in records if r["type"] == "run_start"), {})
    ticks = [r for r in records if r["type"] == "tick"]
    events = [r for r in records if r["type"] not in ("tick",)]
    deploys = [r for r in records if r["type"] == "deploy"]
    retrains = [r for r in records if r["type"] == "retrain"]
    evaluations = [r for r in records if r["type"] == "evaluation"]

    shown = sum(r["shown"] for r in ticks)
    clicks = sum(r["clicks"] for r in ticks)

    if window is None:
        per_tick = shown / len(ticks) if ticks else 1
        window = max(1, round(ROLLING_SESSIONS / max(per_tick, 1e-9)))
        window = min(window, max(1, len(ticks) // 4)) if ticks else 1

    last_deploy_tick = deploys[-1]["tick"] if deploys else None

    def ctr_of(rows):
        s = sum(r["shown"] for r in rows)
        return sum(r["clicks"] for r in rows) / s if s else None

    if last_deploy_tick is not None:
        ctr_now = ctr_of([r for r in ticks if r["tick"] > last_deploy_tick])
        ctr_before = ctr_of([r for r in ticks if r["tick"] <= last_deploy_tick])
    else:
        ctr_now, ctr_before = ctr_of(ticks), None

    active = [r["active_users"] for r in ticks]

    return {
        "run_id": start.get("run_id", Path(".").name),
        "seed": start.get("seed"),
        "created_at": start.get("created_at"),
        "config": start.get("config", {}),
        "reference": start.get("reference", {}),
        "ticks": ticks,
        "last_tick": ticks[-1]["tick"] if ticks else 0,
        "sessions": shown,
        "clicks": clicks,
        "overall_ctr": clicks / shown if shown else None,
        "ctr_now": ctr_now,
        "ctr_before": ctr_before,
        "model_version": max(
            [start.get("model_version", 0)] + [r["model_version"] for r in deploys]
        ),
        "retrains": len(retrains),
        "deploys": deploys,
        "evaluation": evaluations[-1] if evaluations else None,
        "active_mean": sum(active) / len(active) if active else 0,
        "active_last": active[-1] if active else 0,
        "window": window,
        "rolling": rolling(ticks, window),
        "events": events,
    }


# ---------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------

def pct(value, digits=1):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def nice_step(span, max_intervals, steps=(1, 2, 2.5, 5)):
    """Smallest round step (1, 2, 2.5 or 5 x 10^n) that splits `span`
    into at most `max_intervals` intervals."""
    span = max(span, 1e-9)
    magnitude = 10 ** math.floor(math.log10(span / max_intervals))
    for multiple in steps + (10,):
        step = multiple * magnitude
        if span / step <= max_intervals:
            return step
    return 10 * magnitude


def esc(value):
    return html.escape(str(value))


CSS = """
.simdash {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --series: #2a78d6;
  --up: #006300; --down: #b42323; --border: #e1e0d9;
  color-scheme: light;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--ink); background: var(--page);
  padding: 16px; border-radius: 10px; max-width: 980px; box-sizing: border-box;
  line-height: 1.4; font-size: 14px;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .simdash {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --series: #3987e5;
    --up: #0ca30c; --down: #e66767; --border: #2c2c2a; color-scheme: dark;
  }
}
:root[data-theme="dark"] .simdash, body.vscode-dark .simdash, html[theme="dark"] .simdash {
  --surface: #1a1a19; --page: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --series: #3987e5;
  --up: #0ca30c; --down: #e66767; --border: #2c2c2a; color-scheme: dark;
}
.simdash * { box-sizing: border-box; }
.simdash h1 { font-size: 18px; margin: 0 0 2px; font-weight: 600; }
.simdash h2 { font-size: 14px; margin: 0 0 8px; font-weight: 600; }
.simdash .meta { color: var(--ink-2); font-size: 13px; display: flex; flex-wrap: wrap; gap: 4px 16px; }
.simdash .meta b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.simdash .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }
.simdash .tile, .simdash .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.simdash .tile .label { color: var(--ink-2); font-size: 12px; }
.simdash .tile .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; margin: 2px 0; }
.simdash .tile .sub { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.simdash .up { color: var(--up); } .simdash .down { color: var(--down); }
.simdash .panel { margin-bottom: 10px; }
.simdash .chartwrap { overflow-x: auto; }
.simdash svg { display: block; width: 100%; min-width: 560px; height: auto; }
.simdash svg text { font: 11px system-ui, sans-serif; fill: var(--muted); }
.simdash svg .lbl { fill: var(--ink-2); }
.simdash .note { color: var(--muted); font-size: 12px; margin-top: 4px; }
.simdash table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.simdash th { text-align: left; color: var(--ink-2); font-weight: 600; border-bottom: 1px solid var(--axis); padding: 4px 8px 4px 0; }
.simdash td { border-bottom: 1px solid var(--grid); padding: 4px 8px 4px 0; vertical-align: top; }
.simdash .tablewrap { max-height: 260px; overflow: auto; }
.simdash .empty { color: var(--muted); padding: 24px 0; text-align: center; }
"""


def tile(label, value, sub=""):
    return (
        f'<div class="tile"><div class="label">{esc(label)}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def chart_svg(s):
    series = s["rolling"]
    ref = s["reference"]
    random_ctr = ref.get("random_ctr")
    oracle_ctr = ref.get("oracle_ctr")

    if not series:
        return '<div class="empty">Not enough ticks yet for a rolling CTR.</div>'

    # Downsample for a light file: keep every n-th point (and the last).
    step = max(1, len(series) // MAX_POINTS)
    points = series[::step]
    if points[-1] != series[-1]:
        points.append(series[-1])

    x0 = s["ticks"][0]["tick"]
    x1 = max(s["last_tick"], x0 + 1)
    top = max([v for _, v in points] + [r for r in (oracle_ctr, random_ctr) if r] + [0.05])
    y_step = nice_step(top, 5)
    top = min(1.0, math.ceil(top * 1.02 / y_step) * y_step)

    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

    def sx(tick):
        return PAD_LEFT + (tick - x0) / (x1 - x0) * plot_w

    def sy(value):
        return PAD_TOP + (1 - value / top) * plot_h

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="Rolling click-through rate by tick">'
    ]

    # Gridlines + y labels at round percentages.
    for i in range(int(round(top / y_step)) + 1):
        value = i * y_step
        y = sy(value)
        parts.append(
            f'<line x1="{PAD_LEFT}" x2="{PAD_LEFT + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var({"--axis" if i == 0 else "--grid"})" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{value * 100:.0f}%</text>'
        )

    # X labels at round tick numbers.
    x_step = nice_step(x1 - x0, 5, steps=(1, 2, 2.5, 5))
    tick = math.ceil(x0 / x_step) * x_step
    while tick <= x1:
        parts.append(
            f'<text x="{sx(tick):.1f}" y="{HEIGHT - 10}" text-anchor="middle">{tick:,.0f}</text>'
        )
        tick += x_step

    # Reference lines, labelled directly at the right edge.
    for name, value, color in (
        ("oracle", oracle_ctr, "--ink-2"),
        ("random", random_ctr, "--muted"),
    ):
        if value is None:
            continue
        y = sy(value)
        parts.append(
            f'<line x1="{PAD_LEFT}" x2="{PAD_LEFT + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var({color})" stroke-width="1.5" stroke-dasharray="5 4">'
            f'<title>{name} slate: {pct(value)} expected CTR</title></line>'
        )
        parts.append(
            f'<text class="lbl" x="{PAD_LEFT + plot_w + 6}" y="{y + 4:.1f}">'
            f'{name} {pct(value, 0)}</text>'
        )

    # Retrain / deploy markers.
    for deploy in s["deploys"]:
        x = sx(deploy["tick"])
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_TOP}" y2="{PAD_TOP + plot_h}" '
            f'stroke="var(--ink-2)" stroke-width="1" stroke-dasharray="2 3">'
            f'<title>tick {deploy["tick"]:,}: retrain, model v{deploy["model_version"]} deployed</title></line>'
        )
        parts.append(
            f'<text class="lbl" x="{x + 4:.1f}" y="{PAD_TOP + plot_h - 6}">v{deploy["model_version"]}</text>'
        )

    # The rolling CTR line.
    path = " ".join(
        f'{"M" if i == 0 else "L"}{sx(t):.1f},{sy(v):.1f}' for i, (t, v) in enumerate(points)
    )
    parts.append(
        f'<path d="{path}" fill="none" stroke="var(--series)" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    last_t, last_v = points[-1]
    parts.append(
        f'<circle cx="{sx(last_t):.1f}" cy="{sy(last_v):.1f}" r="4" fill="var(--series)" '
        f'stroke="var(--surface)" stroke-width="2"/>'
    )

    # Hover targets: one full-height strip per point, wider than the mark.
    strip = plot_w / max(len(points), 1)
    for tick, value in points:
        parts.append(
            f'<rect x="{sx(tick) - strip / 2:.1f}" y="{PAD_TOP}" width="{max(strip, 1):.1f}" '
            f'height="{plot_h}" fill="transparent">'
            f'<title>tick {tick:,}: rolling CTR {pct(value)}</title></rect>'
        )

    parts.append("</svg>")
    return "".join(parts)


def describe_event(record):
    kind = record["type"]

    if kind == "run_start":
        cfg = record.get("config", {})
        return "Run started", (
            f'seed {record.get("seed")}, {cfg.get("users_per_tick", "?")} users/tick, '
            f'slate {cfg.get("slate_size", "?")} of {cfg.get("num_candidates", "?")}, '
            f'outcomes: {cfg.get("outcome_model", "?")}'
        )
    if kind == "retrain":
        n = record.get("training_examples")
        return "Retrain", f'model v{record["model_version"]} trained' + (
            f" on {n:,} examples" if n is not None else ""
        )
    if kind == "deploy":
        return "Deploy", f'model v{record["model_version"]} serving'
    if kind == "evaluation":
        k = record.get("k", 20)
        bits = [f'Recall@{k} {pct(record.get("recall_at_k"))}']
        if record.get("random_recall_at_k") is not None:
            bits.append(f'random {pct(record["random_recall_at_k"])}')
        if record.get("popularity_recall_at_k") is not None:
            bits.append(f'popularity {pct(record["popularity_recall_at_k"])}')
        return "Offline evaluation", ", ".join(bits)

    details = {k: v for k, v in record.items() if k not in ("type", "tick")}
    return kind, ", ".join(f"{k}={v}" for k, v in details.items())


def event_table(s):
    rows = []
    for record in s["events"]:
        what, details = describe_event(record)
        rows.append(
            f'<tr><td>{record.get("tick", 0):,}</td><td>{esc(what)}</td><td>{esc(details)}</td></tr>'
        )
    return (
        '<div class="tablewrap"><table><thead><tr><th>Tick</th><th>Event</th>'
        '<th>Details</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
    )


def render_fragment(run_dir, window=None):
    """The dashboard as an HTML fragment (a styled <div>), for embedding."""
    s = summarize(read_log(run_dir), window=window)
    cfg = s["config"]

    # Tile 1: CTR since the last deploy, vs before it.
    if s["ctr_before"] is not None and s["ctr_now"] is not None:
        delta = (s["ctr_now"] - s["ctr_before"]) * 100
        cls = "up" if delta >= 0 else "down"
        arrow = "▲" if delta >= 0 else "▼"
        ctr_sub = (
            f'<span class="{cls}">{arrow} {abs(delta):.1f} pts</span> vs '
            f'{pct(s["ctr_before"])} before v{s["model_version"]}'
        )
        ctr_label = f'CTR since v{s["model_version"]} deployed'
    else:
        ctr_sub = f'random slate {pct(s["reference"].get("random_ctr"))}'
        ctr_label = "CTR"

    evaluation = s["evaluation"]
    if evaluation:
        k = evaluation.get("k", 20)
        recall_value = pct(evaluation.get("recall_at_k"))
        recall_sub = f'random {pct(evaluation.get("random_recall_at_k"))} · held-out'
        recall_label = f"Recall@{k} vs random"
    else:
        recall_value, recall_sub, recall_label = "—", "no offline evaluation logged", "Recall@20 vs random"

    tiles = "".join([
        tile(ctr_label, pct(s["ctr_now"]), ctr_sub),
        tile(recall_label, recall_value, recall_sub),
        tile("Active users / tick", f'{s["active_mean"]:.1f}'.rstrip("0").rstrip("."),
             f'{s["sessions"]:,} sessions in total'),
        tile("Retrains", f'{s["retrains"]}', f'model v{s["model_version"]} serving'),
    ])

    header = (
        f'<h1>SimPipeline run · {esc(s["run_id"])}</h1>'
        f'<div class="meta"><span>seed <b>{esc(s["seed"])}</b></span>'
        f'<span>simulated time <b>{s["last_tick"]:,} ticks</b></span>'
        f'<span>model <b>v{s["model_version"]}</b></span>'
        f'<span>users/tick <b>{esc(cfg.get("users_per_tick", "?"))}</b></span>'
        f'<span>overall CTR <b>{pct(s["overall_ctr"])}</b></span>'
        + (f'<span>started {esc(s["created_at"])}</span>' if s["created_at"] else "")
        + "</div>"
    )

    window_sessions = s["window"] * (s["sessions"] / max(len(s["ticks"]), 1))
    chart = (
        '<div class="panel"><h2>Rolling click-through rate by tick</h2>'
        + '<div class="chartwrap">' + chart_svg(s) + '</div>'
        + f'<div class="note">Trailing window of {s["window"]:,} ticks '
        f'(≈{window_sessions:,.0f} slates). Dashed lines: expected CTR of a random '
        f'slate and of an oracle slate (most relevant movies first). Dotted '
        f'verticals: retrain + deploy. Hover the chart for values.</div></div>'
    )

    events = '<div class="panel"><h2>Event log</h2>' + event_table(s) + "</div>"

    return (
        f'<div class="simdash"><style>{CSS}</style>{header}'
        f'<div class="tiles">{tiles}</div>{chart}{events}</div>'
    )


def render_dashboard(run_dir, output=None, window=None):
    """Write the dashboard for `run_dir` and return the HTML file's path.

    Reads only <run_dir>/metrics.jsonl. `output` defaults to
    <run_dir>/dashboard.html. `window` is the rolling-CTR window in ticks
    (default: about 250 slates' worth).
    """
    run_dir = Path(run_dir)
    output = Path(output) if output is not None else run_dir / DASHBOARD_FILE

    fragment = render_fragment(run_dir, window=window)
    title = f"SimPipeline run {esc(run_dir.name)}"

    output.write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title}</title>"
        "<style>body{margin:0;padding:16px;background:#f9f9f7}"
        "@media (prefers-color-scheme: dark){body{background:#0d0d0d}}</style>"
        f"</head><body>{fragment}</body></html>\n",
        encoding="utf-8",
    )
    return output


def display_dashboard(run_dir, window=None):
    """Render the dashboard file and show it inline in Jupyter/Colab.
    Returns the path of the written HTML file."""
    from IPython.display import HTML, display

    path = render_dashboard(run_dir, window=window)
    display(HTML(render_fragment(run_dir, window=window)))
    return path
