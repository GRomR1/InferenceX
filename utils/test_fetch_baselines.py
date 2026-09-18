"""Exercise the public-API baseline exporter with a stubbed network fetch."""
import json

from infx.results import fetch_baselines


def _row(**overrides):
    row = {
        "model": "qwen3.5",
        "hardware": "b200",
        "framework": "sglang",
        "precision": "bf16",
        "spec_method": "none",
        "disagg": False,
        "isl": 8192,
        "osl": 1024,
        "conc": 4,
        "image": "lmsysorg/sglang:v0.5.14-cu130",
        "prefill_tp": 8,
        "decode_tp": 8,
        "num_prefill_gpu": 8,
        "num_decode_gpu": 8,
        "metrics": {
            "tput_per_gpu": 1234.5,
            "median_ttft": 0.318,
            "joules_per_total_token": 0.021,
        },
    }
    row.update(overrides)
    return row


def test_fetch_rows_encodes_reserved_characters_in_model(tmp_path, monkeypatch):
    """A frontend name with space/&/# must be percent-encoded in the query."""
    captured = {}

    class _Response:
        def __init__(self):
            self.headers = {"Content-Encoding": "identity"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"[]"

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _Response()

    monkeypatch.setattr(
        fetch_baselines.urllib.request, "urlopen", _fake_urlopen, raising=False
    )

    fetch_baselines.fetch_rows("Qwen 3.5 & Friends #2 (beta)")

    url = captured["url"]
    # Hand-computed: urlencode("Qwen 3.5 & Friends #2 (beta)") percent-encodes
    # &, # and the parens and plus-encodes the spaces.
    assert url == (
        "https://inferencex.semianalysis.com/api/v1/benchmarks"
        "?model=Qwen+3.5+%26+Friends+%232+%28beta%29"
    )
    value = url.split("model=", 1)[1]
    # No raw reserved character may remain outside its %XX form.
    assert "&" not in value.replace("%26", "")
    assert "#" not in value
    assert " " not in value


def test_main_filters_groups_and_maps_canonical_fields(tmp_path, monkeypatch):
    rows = [
        _row(),
        _row(conc=8, metrics={"tput_per_gpu": 2048.0}),
        _row(hardware="b300", framework="trt", precision="fp8"),
        _row(isl=4096),
    ]
    monkeypatch.setattr(fetch_baselines, "fetch_rows", lambda model: rows)

    rc = fetch_baselines.main(
        [
            "--model", "Qwen-3.5-397B-A17B",
            "--out-dir", str(tmp_path),
            "--isl", "8192", "--osl", "1024",
        ]
    )

    assert rc == 0
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "Qwen-3.5-397B-A17B_8192x1024_b200_sglang_bf16.json",
        "Qwen-3.5-397B-A17B_8192x1024_b300_trt_fp8.json",
    ]
    points = json.loads(
        (tmp_path / "Qwen-3.5-397B-A17B_8192x1024_b200_sglang_bf16.json").read_text()
    )
    assert [p["conc"] for p in points] == [4, 8]
    point = points[0]
    # Hand-checked canonical mapping, field by field.
    assert point["infmax_model_prefix"] == "qwen3.5"
    assert point["framework"] == "sglang"
    assert point["precision"] == "bf16"
    assert point["spec_decoding"] == "none"
    assert point["disagg"] is False
    assert (point["isl"], point["osl"], point["conc"]) == (8192, 1024, 4)
    assert point["hw"] == "b200"
    assert point["image"] == "lmsysorg/sglang:v0.5.14-cu130"
    assert point["tp"] == 8
    assert point["tput_per_gpu"] == 1234.5
    assert point["median_ttft"] == 0.318
    assert point["joules_per_total_token"] == 0.021


def test_main_stamps_explicit_model_prefix(tmp_path, monkeypatch):
    rows = [_row()]
    monkeypatch.setattr(fetch_baselines, "fetch_rows", lambda model: rows)

    rc = fetch_baselines.main(
        [
            "--model", "Qwen-3.5-397B-A17B",
            "--model-prefix", "qwen3.8",
            "--out-dir", str(tmp_path),
            "--isl", "8192", "--osl", "1024",
        ]
    )

    assert rc == 0
    points = json.loads(
        (tmp_path / "Qwen-3.5-397B-A17B_8192x1024_b200_sglang_bf16.json").read_text()
    )
    assert points[0]["infmax_model_prefix"] == "qwen3.8"


def test_main_no_match_exits_1_and_writes_nothing(tmp_path, monkeypatch):
    rows = [_row(isl=4096)]
    monkeypatch.setattr(fetch_baselines, "fetch_rows", lambda model: rows)

    rc = fetch_baselines.main(
        ["--model", "Qwen-3.5-397B-A17B", "--out-dir", str(tmp_path), "--isl", "8192"]
    )

    assert rc == 1
    assert list(tmp_path.glob("*.json")) == []


def test_main_maps_disagg_topology_separately(tmp_path, monkeypatch):
    rows = [_row(disagg=True, prefill_tp=4, decode_tp=8,
                num_prefill_gpu=4, num_decode_gpu=8)]
    monkeypatch.setattr(fetch_baselines, "fetch_rows", lambda model: rows)

    rc = fetch_baselines.main(
        [
            "--model", "Qwen-3.5-397B-A17B",
            "--out-dir", str(tmp_path),
            "--isl", "8192", "--osl", "1024",
        ]
    )

    assert rc == 0
    point = json.loads(
        (tmp_path / "Qwen-3.5-397B-A17B_8192x1024_b200_sglang_bf16.json").read_text()
    )[0]
    assert point["disagg"] is True
    assert point["tp"] is None
    assert (point["prefill_tp"], point["decode_tp"]) == (4, 8)
    assert (point["num_prefill_gpu"], point["num_decode_gpu"]) == (4, 8)
