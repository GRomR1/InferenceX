"""Exercise the offline local comparison with controlled result JSONs."""
import json
from pathlib import Path

from infx.results.local_compare import (
    compare_points,
    load_points,
    match_key,
    render_markdown,
)

RUN_POINT = {
    "infmax_model_prefix": "Qwen3.8",
    "framework": "vllm",
    "precision": "int8",
    "spec_decoding": "none",
    "disagg": False,
    "isl": 1024,
    "osl": 512,
    "conc": 1,
    "hw": "metax-c500",
    "tp": 1,
    "tput_per_gpu": 10.0,
    "output_tput_per_gpu": 4.0,
    "median_ttft": 2.0,
    "p99_ttft": 5.0,
    "median_tpot": 0.05,
    "p99_tpot": 0.12,
    "median_intvty": 20.0,
    "joules_per_total_token": 0.5,
}

BASELINE_POINT = {
    **RUN_POINT,
    "infmax_model_prefix": "qwen3.8",
    "hw": "h100",
    "tp": 8,
    "tput_per_gpu": 20.0,
    "output_tput_per_gpu": 8.0,
    "median_ttft": 1.0,
    "p99_ttft": 2.5,
    "median_tpot": 0.04,
    "p99_tpot": 0.10,
    "median_intvty": 25.0,
}
del BASELINE_POINT["joules_per_total_token"]


def test_compare_points_computes_deltas_by_hand():
    comparison = compare_points(RUN_POINT, BASELINE_POINT)

    tput = comparison["tput_per_gpu"]
    assert tput["delta"] == -10.0
    assert tput["delta_pct"] == -50.0
    assert tput["higher_is_better"] is True

    ttft = comparison["median_ttft"]
    assert ttft["delta"] == 1.0
    assert ttft["delta_pct"] == 100.0
    assert ttft["higher_is_better"] is False

    intvty = comparison["median_intvty"]
    assert intvty["delta"] == -5.0
    assert intvty["delta_pct"] == -20.0

    joules = comparison["joules_per_total_token"]
    assert joules["delta"] is None
    assert joules["delta_pct"] is None
    assert joules["baseline"] is None


def test_match_key_is_case_insensitive_and_conc_scoped():
    # Hand-computed; RUN_POINT's prefix is intentionally capitalized to prove
    # the key lowercases it.
    assert match_key(RUN_POINT) == (
        "qwen3.8", "vllm", "int8", "none", False, 1024, 512, 1,
    )
    other_conc = {**RUN_POINT, "conc": 2}
    assert match_key(other_conc) == (
        "qwen3.8", "vllm", "int8", "none", False, 1024, 512, 2,
    )


def test_load_points_and_render_markdown(tmp_path: Path):
    baseline_path = tmp_path / "h100_vllm.json"
    batch = [
        BASELINE_POINT,
        {**BASELINE_POINT, "conc": 2, "tput_per_gpu": 30.0},
    ]
    baseline_path.write_text(json.dumps(batch))

    points = load_points(baseline_path)
    # Hand-computed (prefix, framework, precision, spec, disagg, isl, osl, conc).
    assert set(points) == {
        ("qwen3.8", "vllm", "int8", "none", False, 1024, 512, 1),
        ("qwen3.8", "vllm", "int8", "none", False, 1024, 512, 2),
    }

    unmatched = {**RUN_POINT, "osl": 1024}
    report, matched = render_markdown(
        "run", [unmatched, RUN_POINT], [(baseline_path, points)]
    )

    assert matched is True
    assert "No baseline matched this point." in report
    # tp is surfaced, not a matching key: tp1 run matches the tp8 baseline.
    assert "run [metax-c500 tp1]" in report
    # conc=1 point: run tput 10.0 vs baseline 20.0 -> -10.0000 (-50.0%)
    assert "| Total throughput (tok/s per GPU) | 10 | 20 | -10.0000 (-50.0%) |" in report
    # missing baseline metric renders as dashes, not a number
    assert "| Energy per total token (J) | 0.5 | - | - |" in report
    assert "h100 tp8 vllm (h100_vllm.json)" in report
