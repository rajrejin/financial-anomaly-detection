"""Serving contract test: the registered model loads, scores a real recent day, validates input."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np

from finanomaly.context import Context


def serving_contract(ctx: Context, model_uri: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model_uri": model_uri, "ok": False}
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        out["loaded"] = True
        feats = list(ctx.z.columns)
        recent = ctx.z.loc[ctx.valid].iloc[[-1]]
        res = model.predict(recent)
        out["date"] = str(recent.index[-1].date())
        out["score_finite"] = bool(np.isfinite(res["score"].to_numpy()).all())
        out["alarm_binary"] = bool(set(res["alarm"].unique()) <= {0, 1})
        bad = recent.drop(columns=[feats[0]])
        try:
            model.predict(bad)
            out["rejects_bad_schema"] = False
        except Exception:
            out["rejects_bad_schema"] = True
        out["ok"] = bool(
            out["loaded"]
            and out["score_finite"]
            and out["alarm_binary"]
            and out["rejects_bad_schema"]
        )
    except Exception as exc:  # report, do not raise: the gate treats this as a failed check
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out
