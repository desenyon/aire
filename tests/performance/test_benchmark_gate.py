import importlib.util
from pathlib import Path


def test_benchmark_gate_detects_material_regression_and_missing_case():
    spec = importlib.util.spec_from_file_location(
        "aire_microbench", Path(__file__).resolve().parents[2] / "benchmarks/microbench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = {"benches": {"search": {"p50_ms": 1.0}}}
    assert module.check_regressions({"benches": {"search": {"p50_ms": 1.2}}}, baseline) == []
    assert module.check_regressions({"benches": {"search": {"p50_ms": 10.0}}}, baseline)
    assert module.check_regressions({"benches": {}}, baseline)
    assert module.check_regressions({"benches": {"search": {"p50_ms": float("nan")}}}, baseline)
