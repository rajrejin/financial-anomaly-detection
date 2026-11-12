"""Explanation checks used by the promotion gate and the explain report."""

from __future__ import annotations

from typing import Any

import numpy as np

from finanomaly.models.base import AnomalyModel
from finanomaly.models.trees import TreeClassifierModel


def shap_additivity(model: TreeClassifierModel, X: np.ndarray) -> dict[str, float]:
    """Maximum |sum(SHAP) + base value - model margin| for a TreeExplainer."""
    import shap

    explainer = shap.TreeExplainer(model.booster())
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    base = np.asarray(explainer.expected_value).ravel()[-1]
    recon = np.asarray(sv).sum(axis=1) + base
    gap = float(np.max(np.abs(recon - model.margin(X))))
    return {"max_additivity_gap": gap, "base_value": float(base)}


def check_explanations(model: AnomalyModel, X: np.ndarray) -> dict[str, Any]:
    """Run the family's explanation on real rows X and verify shape and finiteness."""
    out: dict[str, Any] = {"family": model.family}
    contrib = model.explain(X)
    out["shape_ok"] = contrib.shape == X.shape
    out["finite"] = bool(np.isfinite(contrib).all())
    if isinstance(model, TreeClassifierModel):
        add = shap_additivity(model, X)
        out.update(add)
        out["additivity_ok"] = add["max_additivity_gap"] < 1e-3
    else:
        out["additivity_ok"] = True
    out["available"] = bool(out["shape_ok"] and out["finite"] and out["additivity_ok"])
    return out
