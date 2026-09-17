"""Export published InferenceX benchmark rows as local baseline JSON files.

Fetches the public InferenceX-app API (no authentication) and rewrites each
selected row into the canonical per-point dict format used by
``infx.results.fixed_sequence`` / ``infx.results.collect_results``, so the
files can be dropped into a baselines directory for
``infx.results.local_compare`` on a fully air-gapped host.

One output file is written per (hardware, framework, precision) combination:
``<model>_<isl>x<osl>_<hardware>_<framework>_<precision>.json`` containing a
list of points sorted by concurrency.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://inferencex.semianalysis.com/api/v1"
_USER_AGENT = "inferencex-local-bench/1.0"


def fetch_rows(model: str) -> list[dict[str, Any]]:
    url = f"{API_BASE}/benchmarks?model={model}"
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        data = json.loads(body.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"unexpected API payload: {type(data).__name__}")
    return data


def to_point(row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite one API row into the canonical per-point dict format."""
    metrics = row.get("metrics") or {}
    point: dict[str, Any] = {
        "infmax_model_prefix": str(row.get("model", "")).lower(),
        "framework": row.get("framework"),
        "precision": row.get("precision"),
        "spec_decoding": row.get("spec_method", "none"),
        "disagg": bool(row.get("disagg", False)),
        "isl": row.get("isl"),
        "osl": row.get("osl"),
        "conc": row.get("conc"),
        "hw": row.get("hardware"),
        "image": row.get("image"),
    }
    point.update(metrics)
    return point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help='Frontend model name, e.g. "qwen3.5" or "dsv4"'
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write baseline JSON files into",
    )
    parser.add_argument(
        "--isl", type=int, default=None, help="Keep only rows with this ISL"
    )
    parser.add_argument(
        "--osl", type=int, default=None, help="Keep only rows with this OSL"
    )
    parser.add_argument(
        "--hardware",
        action="append",
        default=None,
        help="Keep only this hardware (repeatable)",
    )
    parser.add_argument(
        "--framework",
        action="append",
        default=None,
        help="Keep only this framework (repeatable)",
    )
    args = parser.parse_args(argv)

    rows = fetch_rows(args.model)
    selected = []
    for row in rows:
        if args.isl is not None and row.get("isl") != args.isl:
            continue
        if args.osl is not None and row.get("osl") != args.osl:
            continue
        if args.hardware and row.get("hardware") not in args.hardware:
            continue
        if args.framework and row.get("framework") not in args.framework:
            continue
        selected.append(row)
    if not selected:
        print("no rows matched the filters", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in selected:
        key = (row.get("hardware"), row.get("framework"), row.get("precision"))
        groups.setdefault(key, []).append(to_point(row))

    written: list[Path] = []
    for (hardware, framework, precision), points in sorted(groups.items()):
        points.sort(key=lambda p: p.get("conc") or 0)
        name = (
            f"{args.model}_{args.isl or '?'}x{args.osl or '?'}_"
            f"{hardware}_{framework}_{precision}.json"
        )
        path = args.out_dir / name
        with open(path, "w") as f:
            json.dump(points, f, indent=2)
        written.append(path)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
