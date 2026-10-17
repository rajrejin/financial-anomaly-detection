"""Baseline evaluation on validation and shared helpers for result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from finanomaly.context import Context
from finanomaly.evaluation.episodes import calibrate_threshold
from finanomaly.evaluation.metrics import evaluate_segment
from finanomaly.models.rules import RULES


def calibrate_and_eval(
    ctx: Context, scores: np.ndarray, seg: tuple[int, int], budget: float | None = None
) -> tuple[float, dict]:
    budget = ctx.cfg.alarm.budget_per_year if budget is None else budget
    scores = np.where(ctx.valid, scores, np.nan)
    thr = calibrate_threshold(
        scores,
        seg,
        ctx.wins,
        budget,
        ctx.cfg.alarm.merge_gap_days,
        max_day_share=ctx.cfg.alarm.max_alarm_day_share,
    )
    res = evaluate_segment(scores, thr, seg, ctx.wins, ctx.cfg.alarm.merge_gap_days)
    return thr, res


def evaluate_baselines_validation(ctx: Context) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in RULES.items():
        thr, res = calibrate_and_eval(ctx, fn(ctx), ctx.val_seg)
        res.pop("false_alarm_episode_list", None)
        out[name] = {"threshold": thr, "validation": res}
    path = ctx.cfg.path("results") / "baselines_validation.json"
    path.write_text(json.dumps(out, indent=2))
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
