"""Generated README blocks. Every number comes from files in results/ (produced by scripts) or
from the curated metadata files; nothing is typed by hand.

README.md contains markers like <!-- BEGIN:name --> ... <!-- END:name -->; `finanomaly report`
replaces the content between them."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yaml

from finanomaly.config import Config


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def _load(cfg: Config, name: str) -> dict | None:
    p = cfg.path("results") / name
    return json.loads(p.read_text()) if p.exists() else None


def block_headline(cfg: Config) -> str:
    dec = _load(cfg, "promotion_decision.json")
    test = _load(cfg, "test_summary.json")
    q = _load(cfg, "data_quality.json")
    lines = []
    if q:
        lines.append(
            f"* Data: {q['rows']} DAX trading days from {q['date_min']} to {q['date_max']}, "
            "Bundesbank, ECB and Yahoo Finance sources."
        )
    ev = yaml.safe_load(Path(cfg.path("events")).read_text())["events"]
    lines.append(f"* {len(ev)} curated, sourced stress events (frozen before modelling).")
    if dec and dec["champion"]:
        c = dec["champion"]
        lines.append(
            f"* Promoted model: **{c['name']}** version {c['version']} (validation event recall "
            f"{_f(c['recall'])}, {_f(c['fa_rate'], 2)} false alarm episodes per year)."
        )
    if test:
        for e in test["entries"]:
            lines.append(
                f"* Test period, {e['label']}: {e['n_detected']} of {e['n_events']} events "
                f"detected, {_f(e['false_alarm_rate'], 2)} alarm episodes per year outside the "
                "event list."
            )
    return "\n".join(lines)


def block_events(cfg: Config) -> str:
    ev = yaml.safe_load(Path(cfg.path("events")).read_text())["events"]
    rows = ["| Id | Event | Category | Onset | Sources |", "|---|---|---|---|---|"]
    for e in ev:
        srcs = ", ".join(f"[{i + 1}]({u})" for i, u in enumerate(e["sources"]))
        rows.append(f"| {e['id']} | {e['name']} | {e['category']} | {e['onset']} | {srcs} |")
    return "\n".join(rows)


def block_catalog(cfg: Config) -> str:
    cat = pd.read_csv(cfg.path("raw").parent / "series_catalog.csv")
    rows = [
        "| Series | Source | Flow | Key | First | Last | Observations |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cat.itertuples():
        rows.append(
            f"| {r.name} | {r.source} | {r.flow} | `{r.key}` | {r.first_date} | {r.last_date} | "
            f"{int(r.n_obs)} |"
        )
    return "\n".join(rows)


def block_validation(cfg: Config) -> str:
    s = pd.read_csv(cfg.path("results") / "runs_summary.csv")
    s = s[s.seed == 0]
    rows = [
        "| Family | Config | Val recall | Val FA/yr | Val AP | Val ROC AUC | Separation |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s.itertuples():
        rows.append(
            f"| {r.family} | {r.config_idx} | {_f(r.val_event_recall)} | "
            f"{_f(r.val_false_alarm_rate, 2)} | {_f(r.val_average_precision)} | "
            f"{_f(r.val_roc_auc)} | {_f(r.val_score_separation, 2)} |"
        )
    base = _load(cfg, "baselines_validation.json") or {}
    for k, v in base.items():
        r = v["validation"]
        rows.append(
            f"| {k} | rule | {_f(r['event_recall'])} | {_f(r['false_alarm_rate'], 2)} | "
            f"{_f(r['average_precision'])} | {_f(r['roc_auc'])} | "
            f"{_f(r['score_separation'], 2)} |"
        )
    return "\n".join(rows)


def block_test(cfg: Config) -> str:
    t = _load(cfg, "test_summary.json")
    if not t:
        return "Not yet evaluated."
    ev = pd.read_csv(cfg.path("results") / "test_events.csv")
    labels = [e["label"] for e in t["entries"]]
    events = list(dict.fromkeys(ev["event"]))
    head = "| Event | " + " | ".join(labels) + " |"
    rows = [head, "|---|" + "---|" * len(labels)]
    for evn in events:
        cells = []
        for lab in labels:
            r = ev[(ev["entry"] == lab) & (ev["event"] == evn)].iloc[0]
            cells.append(f"day {int(r['delay_days']):+d}" if r["detected"] else "missed")
        rows.append(f"| {evn} | " + " | ".join(cells) + " |")
    rows += [
        "",
        "Delay is in trading days from onset to the first alarm in the detection window.",
        "",
    ]
    rows += [
        "| Entry | Detected | Recall | FA episodes/yr | AP | ROC AUC |",
        "|---|---|---|---|---|---|",
    ]
    for e in t["entries"]:
        rows.append(
            f"| {e['label']} | {e['n_detected']} of {e['n_events']} | {_f(e['event_recall'])} | "
            f"{_f(e['false_alarm_rate'], 2)} | {_f(e['average_precision'])} | "
            f"{_f(e['roc_auc'])} |"
        )
    return "\n".join(rows)


def block_seeds(cfg: Config) -> str:
    s = _load(cfg, "seed_stability.json")
    if not s:
        return "Not yet run."
    rows = [
        "| Family | Val recall mean (std) | Test recall mean (std) | Test FA/yr mean (std) | "
        "Test AP mean (std) |",
        "|---|---|---|---|---|",
    ]
    for fam, v in s.items():
        va = v["validation"]
        rows.append(
            f"| {fam} | {_f(va['recall_mean'])} ({_f(va['recall_std'])}) | "
            f"{_f(v['test_recall_mean'])} ({_f(v['test_recall_std'])}) | "
            f"{_f(v['test_fa_mean'], 2)} ({_f(v['test_fa_std'], 2)}) | "
            f"{_f(v['test_ap_mean'])} ({_f(v['test_ap_std'])}) |"
        )
    r = _load(cfg, "rank_stability.json") or {}
    rows += ["", "Spearman rank correlation of the 7 configurations, validation versus test:", ""]
    rows += ["| Family | AP | Event recall |", "|---|---|---|"]
    for fam, v in r.items():
        rows.append(
            f"| {fam} | {_f(v['spearman_val_ap_vs_test_ap'], 2)} | "
            f"{_f(v['spearman_val_recall_vs_test_recall'], 2)} |"
        )
    return "\n".join(rows)


def block_explain(cfg: Config) -> str:
    e = _load(cfg, "explainability.json")
    if not e:
        return "Not yet run."
    rows = ["| Check | LightGBM | XGBoost |", "|---|---|---|"]
    rows.append(
        f"| Max SHAP additivity gap | {e['lightgbm']['max_additivity_gap']:.2e} | "
        f"{e['xgboost']['max_additivity_gap']:.2e} |"
    )
    rows.append(
        f"| Spearman of mean abs SHAP vs gain importance | "
        f"{_f(e['lightgbm']['spearman_shap_vs_gain'], 2)} | "
        f"{_f(e['xgboost']['spearman_shap_vs_gain'], 2)} |"
    )
    rows.append(
        f"| Top 3 features by SHAP | {', '.join(e['lightgbm']['top_by_shap'][:3])} | "
        f"{', '.join(e['xgboost']['top_by_shap'][:3])} |"
    )
    rows.append(
        f"| Top 3 features by gain | {', '.join(e['lightgbm']['top_by_gain'][:3])} | "
        f"{', '.join(e['xgboost']['top_by_gain'][:3])} |"
    )
    feats = e["isolation_forest"]["features"]
    top = sorted(feats.items(), key=lambda kv: -kv[1]["ap_drop_mean"])[:5]
    rows += [
        "",
        "Isolation forest permutation sensitivity (test AP "
        f"{_f(e['isolation_forest']['base_test_ap'])}), largest AP drops:",
        "",
    ]
    rows += ["| Feature | AP drop mean | std |", "|---|---|---|"]
    for k, v in top:
        rows.append(f"| {k} | {_f(v['ap_drop_mean'])} | {_f(v['ap_drop_std'])} |")
    return "\n".join(rows)


def block_extra(cfg: Config) -> str:
    """Sections produced by the robustness phases (9 to 11), if their result files exist."""
    parts = []
    for fn, title, render in EXTRA_RENDERERS:
        d = _load(cfg, fn)
        if d:
            parts.append(f"### {title}\n\n{render(d)}")
    return "\n\n".join(parts) if parts else "Not yet run."


def _render_loeo(d: dict) -> str:
    rows = ["| Held-out event | Model | Detected | Delay |", "|---|---|---|---|"]
    for r in d["rows"]:
        rows.append(
            f"| {r['event']} | {r['model']} | {'yes' if r['detected'] else 'no'} | {r['delay']} |"
        )
    return "\n".join(rows) + "\n\n" + d.get("note", "")


def _render_table(d: dict) -> str:
    cols = d["columns"]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in d["rows"]:
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(rows) + "\n\n" + d.get("note", "")


def _render_loeo_pivot(d: dict) -> str:
    models = list(dict.fromkeys(r["model"] for r in d["rows"]))
    events = list(dict.fromkeys(r["event"] for r in d["rows"]))
    out = ["| Held-out event | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for e in events:
        cells = []
        for m in models:
            r = next(x for x in d["rows"] if x["event"] == e and x["model"] == m)
            cells.append(r["delay"] if r["detected"] == "yes" else "missed")
        out.append(f"| {e} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n\n" + d.get("note", "")


EXTRA_RENDERERS: list[tuple[str, str, Callable[[dict], str]]] = [
    ("loeo.json", "Leave-one-event-out", _render_loeo_pivot),
    ("sensitivity.json", "Sensitivity to windows, budget and training start", _render_table),
    ("crosscheck.json", "Cross-checks between models", _render_table),
    ("alarm_share.json", "Share of alarm days in the test period", _render_table),
    ("ablation.json", "Feature group ablation", _render_table),
    ("walkforward.json", "Walk-forward refit versus fixed fit", _render_table),
]


def block_findings(cfg: Config) -> str:
    """Plain sentences whose numbers are read from the result files."""
    t = _load(cfg, "test_summary.json")
    a = _load(cfg, "alarm_share.json")
    sd = _load(cfg, "seed_stability.json")
    lo = _load(cfg, "loeo.json")
    sens = _load(cfg, "sensitivity.json")
    wf = _load(cfg, "walkforward.json")
    if not (t and a and sd):
        return "Not yet available."
    share = {r["entry"]: r for r in a["rows"]}
    ent = {e["label"].replace(" (champion)", ""): e for e in t["entries"]}
    out = []
    for name, e in ent.items():
        sh = share.get(name)
        out.append(
            f"* {name}: {e['n_detected']} of {e['n_events']} test events detected, "
            f"{_f(e['false_alarm_rate'], 2)} false alarm episodes per year, "
            f"{sh['share_of_test_days'] if sh else 'n/a'} of test days in alarm "
            f"(mean episode {sh['mean_episode_days'] if sh else 'n/a'} days)."
        )
    budget = cfg.alarm.budget_per_year
    over = [n for n, e in ent.items() if n.startswith("anomaly") and e["false_alarm_rate"] > budget]
    if over:
        out.append(
            f"* The budget used to set thresholds was {budget:g} false alarm episodes per year. "
            f"Exceeded in the test period by: {', '.join(over)}."
        )
    out.append(
        "* The false alarm episode count ignores episodes that overlap an event window, so a "
        "model that alarms on many days can show few episodes; read it together with the share "
        "of alarm days."
    )
    ae = sd.get("autoencoder")
    if ae:
        out.append(
            f"* Autoencoder test recall over 5 seeds: mean {_f(ae['test_recall_mean'])}, spread "
            f"{_f(ae['test_recall_std'])}."
        )
    if lo:
        per: dict[str, int] = {}
        tot: dict[str, int] = {}
        for r in lo["rows"]:
            tot[r["model"]] = tot.get(r["model"], 0) + 1
            per[r["model"]] = per.get(r["model"], 0) + (r["detected"] == "yes")
        out.append(
            "* Leave-one-event-out detections: "
            + ", ".join(f"{m} {per[m]} of {tot[m]}" for m in per)
            + "."
        )
    if sens:
        for r in sens["rows"]:
            if r["setting"].startswith("false alarm budget 1 "):
                out.append(
                    f"* With a budget of 1 episode per year, {r['model']} test recall is "
                    f"{r['test_recall']} (labelled sensitivity analysis, not used to retune)."
                )
    if wf:
        for r in wf["rows"]:
            out.append(
                f"* Walk-forward check, {r['model']}, {r['schedule']}: {r['detected']} events, "
                f"{r['test_fa_per_year']} false alarm episodes per year."
            )
    return "\n".join(out)


def block_modelcard(cfg: Config) -> str:
    dec = _load(cfg, "promotion_decision.json")
    reg = _load(cfg, "registration.json")
    t = _load(cfg, "test_summary.json")
    sd = _load(cfg, "seed_stability.json")
    if not (dec and dec["champion"] and reg):
        return "No promoted model."
    c = dec["champion"]
    info = reg[c["family"]]
    from finanomaly.experiments.runner import load_experiments

    params = load_experiments()[c["family"]][info["config_idx"]]
    man_path = Path(cfg.root) / cfg.serving.model_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    rows = [
        "| Item | Value |",
        "|---|---|",
        f"| Registered name and version | {c['name']} version {c['version']} (alias production) |",
        f"| Family and configuration | {c['family']}, configuration {info['config_idx']}: "
        f"`{params}` |",
        f"| Fit data | all data before {cfg.splits.test_start} (features valid from 2000-12) |",
        f"| Alarm threshold | {_f(man.get('threshold'), 4)} (score scale of the model) |",
        f"| Validation (fit before 2015, 2015 to 2018) | recall {_f(c['recall'])}, "
        f"{_f(c['fa_rate'], 2)} false alarm episodes per year, AP {_f(c['ap'])} |",
    ]
    if sd and c["family"] in sd:
        v = sd[c["family"]]
        rows.append(
            f"| Seed spread (5 seeds), test | recall {_f(v['test_recall_mean'])} "
            f"(std {_f(v['test_recall_std'])}), FA per year {_f(v['test_fa_mean'], 2)} "
            f"(std {_f(v['test_fa_std'], 2)}) |"
        )
    if t:
        e = next(x for x in t["entries"] if x["label"].startswith(c["name"]))
        rows.append(
            f"| Test period (2019 onward) | {e['n_detected']} of {e['n_events']} events, "
            f"{_f(e['false_alarm_rate'], 2)} false alarm episodes per year, AP "
            f"{_f(e['average_precision'])} |"
        )
    if man:
        rows.append(f"| Data hash | `{man['data_hash'][:16]}` |")
        rows.append(f"| Events hash | `{man['events_hash'][:16]}` |")
        rows.append(f"| Git commit at export | `{man['git_commit'][:10]}` |")
    return "\n".join(rows)


BLOCKS: dict[str, Callable[[Config], str]] = {
    "modelcard": block_modelcard,
    "findings": block_findings,
    "headline": block_headline,
    "events": block_events,
    "catalog": block_catalog,
    "validation": block_validation,
    "test": block_test,
    "seeds": block_seeds,
    "explain": block_explain,
    "extra": block_extra,
}


def _replacer(body: str) -> Callable[[re.Match[str]], str]:
    def repl(m: re.Match[str]) -> str:
        return m.group(1) + "\n" + body + "\n" + m.group(3)

    return repl


def update_readme(cfg: Config, path: Path | None = None) -> list[str]:
    path = path or Path(cfg.root) / "README.md"
    text = path.read_text(encoding="utf-8")
    done = []
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.DOTALL)
        if pat.search(text):
            body = fn(cfg)
            text = pat.sub(_replacer(body), text)
            done.append(name)
    path.write_text(text, encoding="utf-8")
    return done
