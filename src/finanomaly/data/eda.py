"""Exploratory figures with event windows shaded."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from finanomaly.config import Config  # noqa: E402
from finanomaly.data.build import load_dataset  # noqa: E402
from finanomaly.events import event_windows, load_events  # noqa: E402


def _shade(ax, idx, wins):
    for w in wins:
        a, b = w.detect
        ax.axvspan(idx[a], idx[min(b, len(idx) - 1)], color="tab:red", alpha=0.18, lw=0)


def run_eda(cfg: Config) -> list[str]:
    t = load_dataset(cfg)
    idx = t.index.tz_localize(None)
    wins = event_windows(cfg, t.index, load_events(cfg))
    out = cfg.path("results") / "figures" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, (col, lab) in zip(
        axes,
        [
            ("dax_close", "DAX close"),
            ("usd_per_eur", "USD per EUR"),
            ("chf_per_eur", "CHF per EUR"),
            ("de_yield_10y", "German 10y yield (%)"),
        ],
        strict=True,
    ):
        ax.plot(idx, t[col].to_numpy(), lw=0.8)
        ax.set_ylabel(lab)
        _shade(ax, idx, wins)
    axes[0].set_title("Series with event detection windows shaded")
    fig.tight_layout()
    f = out / "series_events.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(str(f))

    r = np.log(t["dax_close"]).diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(r, bins=120, density=True, alpha=0.7)
    x = np.linspace(r.min(), r.max(), 400)
    axes[0].plot(x, np.exp(-0.5 * ((x - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi)))
    axes[0].set_yscale("log")
    axes[0].set_title("DAX log returns vs normal (log density)")
    pd.Series(r.to_numpy()).sort_values().plot(ax=axes[1], style=".", ms=1)
    axes[1].set_title(f"Sorted returns, excess kurtosis {r.kurt():.1f}")
    fig.tight_layout()
    f = out / "return_distribution.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(str(f))

    vol = r.rolling(20).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(vol.index.tz_localize(None), vol.to_numpy(), lw=0.8)
    _shade(ax, idx, wins)
    ax.set_title("DAX 20 day realised volatility (annualised) with event windows")
    fig.tight_layout()
    f = out / "rolling_vol.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(str(f))

    rets = pd.DataFrame(
        {
            "dax": np.log(t["dax_close"]).diff(),
            "usd": np.log(t["usd_per_eur"]).diff(),
            "chf": np.log(t["chf_per_eur"]).diff(),
            "y10": t["de_yield_10y"].diff(),
        }
    )
    c1 = rets["dax"].rolling(250).corr(rets["usd"])
    c2 = rets["dax"].rolling(250).corr(rets["y10"])
    c3 = rets["dax"].rolling(250).corr(rets["chf"])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for c, lab in ((c1, "DAX vs USD"), (c2, "DAX vs 10y yield change"), (c3, "DAX vs CHF")):
        ax.plot(c.index.tz_localize(None), c.to_numpy(), lw=0.9, label=lab)
    ax.legend()
    ax.set_title("250 day rolling correlation of daily changes")
    fig.tight_layout()
    f = out / "rolling_correlation.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files.append(str(f))
    return files
