"""Supervised gradient boosted tree classifiers (LightGBM and XGBoost) with Platt calibration.

Training protocol: fit on (X, y) with early stopping on a later time-ordered block (X_es, y_es);
fit a Platt calibrator on that block's out-of-sample predictions; refit on both blocks with the
selected number of trees.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from finanomaly.models.base import AnomalyModel


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class TreeClassifierModel(AnomalyModel):
    max_trees = 600
    early_stopping_rounds = 40

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.clf: Any = None
        self.best_iteration = 0
        self.cal_coef = 1.0
        self.cal_intercept = 0.0
        self.calibrated = False
        self.n_features = 0

    # -- implemented by subclasses -----------------------------------------------------------
    def _make(self, seed: int, n_estimators: int, spw: float) -> Any:
        raise NotImplementedError

    def _fit_es(self, clf: Any, X: Any, y: Any, Xe: Any, ye: Any) -> int:
        raise NotImplementedError

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(X)[:, 1])

    def booster(self) -> Any:
        raise NotImplementedError

    def margin(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- common ------------------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        seed: int = 0,
        X_es: np.ndarray | None = None,
        y_es: np.ndarray | None = None,
        **kw: Any,
    ) -> TreeClassifierModel:
        assert y is not None
        self.seed = seed
        self.n_features = X.shape[1]
        pos = max(int(y.sum()), 1)
        spw = float((len(y) - y.sum()) / pos)
        if X_es is not None and y_es is not None and len(X_es) > 0:
            first = self._make(seed, self.max_trees, spw)
            self.best_iteration = max(self._fit_es(first, X, y, X_es, y_es), 5)
            p = np.asarray(first.predict_proba(X_es)[:, 1])
            if y_es.min() != y_es.max():
                lr = LogisticRegression(C=1e6).fit(_logit(p).reshape(-1, 1), y_es)
                self.cal_coef = float(lr.coef_[0, 0])
                self.cal_intercept = float(lr.intercept_[0])
                self.calibrated = True
            X_all = np.vstack([X, X_es])
            y_all = np.concatenate([y, y_es])
        else:
            self.best_iteration = 200
            X_all, y_all = X, y
        pos = max(int(y_all.sum()), 1)
        spw_all = float((len(y_all) - y_all.sum()) / pos)
        self.clf = self._make(seed, self.best_iteration, spw_all)
        self.clf.fit(X_all, y_all)
        return self

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        return self._raw_proba(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        p = self._raw_proba(X)
        if not self.calibrated:
            return p
        z = self.cal_coef * _logit(p) + self.cal_intercept
        return 1.0 / (1.0 + np.exp(-z))

    def explain(self, X: np.ndarray) -> np.ndarray:
        import shap

        sv = shap.TreeExplainer(self.booster()).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        return np.asarray(sv)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "clf": self.clf,
                    "params": self.params,
                    "seed": self.seed,
                    "best_iteration": self.best_iteration,
                    "cal": [self.cal_coef, self.cal_intercept, self.calibrated],
                    "n_features": self.n_features,
                },
                f,
            )
        (path / "meta.json").write_text(
            json.dumps({"best_iteration": self.best_iteration, "calibrated": self.calibrated})
        )

    @classmethod
    def load(cls, path: Path) -> TreeClassifierModel:
        with open(path / "model.pkl", "rb") as f:
            d = pickle.load(f)  # noqa: S301 - artifact produced by this project
        m = cls(**d["params"])
        m.clf, m.seed, m.best_iteration = d["clf"], d["seed"], d["best_iteration"]
        m.cal_coef, m.cal_intercept, m.calibrated = d["cal"]
        m.n_features = d["n_features"]
        return m

    def n_parameters(self) -> int:
        return self.best_iteration * int(
            self.params.get("num_leaves", 2 ** self.params.get("max_depth", 3))
        )


class LightGBMModel(TreeClassifierModel):
    family = "lightgbm"

    def _make(self, seed: int, n_estimators: int, spw: float) -> Any:
        import lightgbm as lgb

        p = self.params
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=int(p["num_leaves"]),
            learning_rate=float(p["learning_rate"]),
            min_child_samples=int(p["min_child_samples"]),
            colsample_bytree=float(p["feature_fraction"]),
            reg_lambda=float(p["reg_lambda"]),
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=1,
            subsample=0.8,
            subsample_freq=1,
            verbose=-1,
        )

    def _fit_es(self, clf: Any, X: Any, y: Any, Xe: Any, ye: Any) -> int:
        import lightgbm as lgb

        clf.fit(
            X,
            y,
            eval_X=Xe,
            eval_y=ye,
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return int(clf.best_iteration_ or self.max_trees)

    def booster(self) -> Any:
        return self.clf.booster_

    def margin(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.clf.predict(X, raw_score=True))


class XGBoostModel(TreeClassifierModel):
    family = "xgboost"

    def _make(self, seed: int, n_estimators: int, spw: float) -> Any:
        import xgboost as xgb

        p = self.params
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            min_child_weight=float(p["min_child_weight"]),
            colsample_bytree=float(p["colsample_bytree"]),
            reg_lambda=float(p["reg_lambda"]),
            scale_pos_weight=spw,
            subsample=0.8,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=None,
        )

    def _fit_es(self, clf: Any, X: Any, y: Any, Xe: Any, ye: Any) -> int:
        import xgboost as xgb

        clf.set_params(early_stopping_rounds=self.early_stopping_rounds)
        clf.fit(X, y, eval_set=[(Xe, ye)], verbose=False)
        best = getattr(clf, "best_iteration", None)
        clf.set_params(early_stopping_rounds=None)
        del xgb
        return int(best + 1) if best is not None else self.max_trees

    def booster(self) -> Any:
        return self.clf.get_booster()

    def margin(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.clf.predict(X, output_margin=True))
