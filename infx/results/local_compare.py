"""Offline comparison of canonical InferenceX result JSONs.

Compares one local run against one or more baseline result files (other
engines, other GPUs) without any database or network access. Baselines are
plain JSON files in the same canonical format produced by
``infx.results.fixed_sequence`` (a single point dict) or
``infx.results.collect_results`` (a list of point dicts), e.g. exports from
the public InferenceX database or from other local runs.

Points are matched on (model prefix, framework, precision, spec decoding,
disaggregation, ISL, OSL, concurrency). Throughput and intvty are
higher-is-better; latencies and energy are lower-is-better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# (agg JSON key, display label, higher is better)
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("tput_per_gpu", "Total throughput (tok/s per GPU)", True),
    ("output_tput_per_gpu", "Output throughput (tok/s per GPU)", True),
    ("median_ttft", "TTFT median (s)", False),
    ("p99_ttft", "TTFT p99 (s)", False),
    ("median_tpot", "TPOT median (s)", False),
    ("p99_tpot", "TPOT p99 (s)", False),
    ("median_intvty", "Intvty median (tok/s per user)", True),
    ("joules_per_total_token", "Energy per total token (J)", False),
)


def match_key(point: dict[str, Any]) -> tuple[Any, ...]:
    """Identity under which a run point and a baseline point are comparable."""
    return (
        str(point.get("infmax_model_prefix", "")).lower(),
        str(point.get("framework", "")).lower(),
        str(point.get("precision", "")).lower(),
        str(point.get("spec_decoding", "none")).lower(),
        bool(point.get("disagg", False)),
        point.get("isl"),
        point.get("osl"),
        point.get("conc"),
    )


def _as_point_list(data: Any) -> list[dict[str, Any]]:
    """Accept either a single point dict or a list of point dicts."""
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def load_points(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Load one result file as a {match key: point} map.

    The first occurrence of a key wins; non-dict entries in a batch list are
    ignored.
    """
    with open(path) as f:
        data = json.load(f)
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for point in _as_point_list(data):
        key = match_key(point)
        if key not in out:
            out[key] = point
    return out


def compare_points(
    run_point: dict[str, Any], baseline_point: dict[str, Any]
) -> dict[str, Any]:
    """Per-metric run-vs-baseline numbers for one matched pair of points.

    Delta is run minus baseline; delta_pct is relative to the baseline. Both
    are None when either side is missing or the baseline is zero.
    """
    out: dict[str, Any] = {}
    for key, _label, higher_is_better in METRICS:
        run_value = run_point.get(key)
        baseline_value = baseline_point.get(key)
        entry: dict[str, Any] = {
            "run": run_value,
            "baseline": baseline_value,
            "delta": None,
            "delta_pct": None,
            "higher_is_better": higher_is_better,
        }
        if (
            isinstance(run_value, (int, float))
            and not isinstance(run_value, bool)
            and isinstance(baseline_value, (int, float))
            and not isinstance(baseline_value, bool)
            and baseline_value != 0
        ):
            delta = float(run_value) - float(baseline_value)
            entry["delta"] = delta
            entry["delta_pct"] = delta / float(baseline_value) * 100.0
        out[key] = entry
    return out


def _fmt_value(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return "-"


def _fmt_delta(entry: dict[str, Any]) -> str:
    if entry["delta"] is None or entry["delta_pct"] is None:
        return "-"
    return f"{entry['delta']:+.4f} ({entry['delta_pct']:+.1f}%)"


def _baseline_label(path: Path, point: dict[str, Any]) -> str:
    hw = point.get("hw", "unknown-hw")
    framework = point.get("framework", "unknown-fw")
    return f"{hw} {framework} ({path.name})"


def render_markdown(
    run_name: str,
    run_points: list[dict[str, Any]],
    baselines: list[tuple[Path, dict[tuple[Any, ...], dict[str, Any]]]],
) -> tuple[str, bool]:
    """Render the comparison report; the bool reports whether anything matched."""
    lines: list[str] = [f"# Local comparison: {run_name}", ""]
    matched_any = False
    ordered = sorted(
        run_points,
        key=lambda p: (p.get("isl") or 0, p.get("osl") or 0, p.get("conc") or 0),
    )
    for run_point in ordered:
        key = match_key(run_point)
        conc = run_point.get("conc")
        isl = run_point.get("isl")
        osl = run_point.get("osl")
        matches: list[tuple[str, dict[str, Any]]] = []
        for baseline_path, baseline_points in baselines:
            baseline_point = baseline_points.get(key)
            if baseline_point is not None:
                matches.append(
                    (_baseline_label(baseline_path, baseline_point), baseline_point)
                )
        lines.append(f"## ISL {isl} / OSL {osl} / concurrency {conc}")
        lines.append("")
        if not matches:
            lines.append("No baseline matched this point.")
            lines.append("")
            continue
        matched_any = True
        comparisons = [
            compare_points(run_point, baseline_point)
            for _label, baseline_point in matches
        ]
        header = ["Metric", run_name]
        for label, _baseline_point in matches:
            header.append(label)
            header.append(f"\u0394 vs {label}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for metric_key, label, _higher_is_better in METRICS:
            row = [label, _fmt_value(run_point.get(metric_key))]
            for comparison in comparisons:
                entry = comparison[metric_key]
                row.append(_fmt_value(entry["baseline"]))
                row.append(_fmt_delta(entry))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines), matched_any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run result JSON (single point dict or collect_results list)",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        required=True,
        help="Directory of baseline result JSON files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the markdown report here instead of stdout",
    )
    parser.add_argument(
        "--name", default=None, help="Label for this run (default: run file stem)"
    )
    args = parser.parse_args(argv)

    if not args.run.is_file():
        print(f"run file not found: {args.run}", file=sys.stderr)
        return 2
    if not args.baselines.is_dir():
        print(f"baselines directory not found: {args.baselines}", file=sys.stderr)
        return 2
    baseline_files = sorted(args.baselines.glob("*.json"))
    if not baseline_files:
        print(f"no baseline JSON files in {args.baselines}", file=sys.stderr)
        return 2

    baselines = [(path, load_points(path)) for path in baseline_files]
    with open(args.run) as f:
        run_points = _as_point_list(json.load(f))
    if not run_points:
        print(f"no result points in {args.run}", file=sys.stderr)
        return 2

    report, matched = render_markdown(args.name or args.run.stem, run_points, baselines)
    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"report: {args.out}", file=sys.stderr)
    else:
        print(report)
    if not matched:
        print("warning: no baseline points matched the run points", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
