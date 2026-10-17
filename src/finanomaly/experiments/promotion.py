"""Selection rule and the promotion gate. Pure logic, unit tested on hand-written cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Row:
    """One configuration's validation result (seed 0)."""

    config_idx: int
    recall: float
    fa_rate: float
    ap: float | None
    n_parameters: float


def select_best(rows: list[Row]) -> Row:
    """Highest recall, then lower false alarm rate, then higher average precision, then simpler."""
    return sorted(rows, key=lambda r: (-r.recall, r.fa_rate, -(r.ap or 0.0), r.n_parameters))[0]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    version: int
    recall: float
    fa_rate: float
    ap: float
    recall_std: float
    explanations_ok: bool
    contract_ok: bool


@dataclass(frozen=True)
class GateConfig:
    budget_per_year: float = 3.0
    fa_tolerance: float = 0.5
    max_recall_std: float = 0.2


def gate_candidate(
    c: Candidate,
    baseline_recall: float,
    production: Candidate | None,
    gate: GateConfig,
) -> dict[str, Any]:
    ref = baseline_recall if production is None else max(baseline_recall, production.recall)
    checks = {
        "recall_vs_reference": c.recall >= ref,
        "recall_vs_best_baseline": c.recall >= baseline_recall,
        "false_alarm_within_budget": c.fa_rate <= gate.budget_per_year + gate.fa_tolerance,
        "seed_spread_acceptable": c.recall_std <= gate.max_recall_std,
        "explanations_available": c.explanations_ok,
        "serving_contract": c.contract_ok,
    }
    return {
        "candidate": asdict(c),
        "reference_recall": ref,
        "checks": checks,
        "passes": all(checks.values()),
    }


def apply_gate(
    candidates: list[Candidate],
    baseline_recall: float,
    production: Candidate | None,
    gate: GateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate every candidate; the champion is the best passing one (recall, FA rate, AP)."""
    gate = gate or GateConfig()
    results = [gate_candidate(c, baseline_recall, production, gate) for c in candidates]
    passing = [r for r in results if r["passes"]]
    champion = None
    if passing:
        champion = sorted(
            passing,
            key=lambda r: (
                -r["candidate"]["recall"],
                r["candidate"]["fa_rate"],
                -r["candidate"]["ap"],
                r["candidate"]["name"],
            ),
        )[0]["candidate"]
    return {
        "baseline_recall": baseline_recall,
        "production_before": asdict(production) if production else None,
        "results": results,
        "champion": champion,
    }
