from pathlib import Path

PRE_DECISION = [
    "experiments/runner.py",
    "experiments/registry.py",
    "experiments/promotion.py",
    "experiments/summary.py",
    "experiments/tracking.py",
]


def test_selection_code_never_touches_the_test_segment():
    root = Path(__file__).resolve().parents[2] / "src" / "finanomaly"
    for rel in PRE_DECISION:
        text = (root / rel).read_text(encoding="utf-8")
        for forbidden in ("test_seg", "split.test", "test_start_seg"):
            assert forbidden not in text, f"{rel} references {forbidden}"
