"""Review of the highest-scoring alarm episodes outside the curated event list (champion model).

Scores, dates and episode lengths come from `results/test_false_alarms.csv`. The one-line
descriptions were looked up by hand from the cited sources (retrieved 2026-10-17). Where no source
was found, the entry says so; no cause is asserted without a source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finanomaly.config import Config

NOT_FOUND = "No sourced explanation found."

# peak_date -> (description, source URL or "")
ANNOTATIONS: dict[str, tuple[str, str]] = {
    "2025-03-05": (
        "German fiscal package announcement; 10 year Bund yield rose about 30 bp, the largest "
        "one-day rise since 1998 per the source.",
        "https://russellinvestments.com/kr/blog/2025/03/germany-fiscal-shift-big-deal",
    ),
    "2026-04-08": (
        "Context only, the DAX move on this day was not found: a 2 April 2026 market note cites "
        "US tariff plans, the risk of EU countermeasures and energy costs linked to a Middle East "
        "conflict.",
        "https://capital.com/de-de/market-updates/dax-index-forecast-07-04-2026",
    ),
    "2022-02-04": (
        "Day after the ECB meeting where Lagarde dropped the earlier guidance that a 2022 hike was "
        "very unlikely; German bond yields rose and the DAX traded lower.",
        "https://investing.com/news/stock-market-news/european-stocks-mixed-rising-bond-yields-weigh-on-dax-2756894",
    ),
    "2020-05-19": (
        "Day after the French-German proposal of a 500 billion euro EU recovery fund; European "
        "stocks opened lower on doubts over the fund.",
        "https://investing.com/news/stock-market-news/stocks--europe-edges-higher-2176425",
    ),
    "2022-04-19": (NOT_FOUND, ""),
    "2026-03-03": (
        "Not confirmed. Search results mention a Middle East conflict and sharp DAX declines in "
        "early March 2026 but no dated source for this day was retrieved.",
        "",
    ),
    "2023-12-14": (
        "Day after the Fed held rates and projected cuts for 2024; bond yields fell. The DAX move "
        "itself was not sourced.",
        "https://www.businesstoday.com.my/2023/12/14/dow-hits-a-record-as-wall-street-rallies-after-fed-signals-rate-cuts-in-2024/",
    ),
    "2022-06-13": (
        "Sell-off after the ECB confirmed a July hike and ahead of the Fed meeting; the DAX had "
        "fallen about 1,000 points over the preceding week.",
        "https://www.dailyfx.com/forex/market_alert/2022/06/13/DAX-40-DOW-JONES-FTSE-100-Outlook-Talking-points-Analysis-and-Charts.html",
    ),
    "2026-10-02": (NOT_FOUND, ""),
    "2022-09-23": (
        "UK mini-budget gilt turmoil and euro area flash PMIs signalling contraction; the DAX fell "
        "about 2 percent.",
        "https://investing.com/news/economy/pound-ftse-gilts-all-slump-after-kwartengs-mini-budget-backfires-2897838",
    ),
    "2020-06-11": (
        "Gloomy Fed outlook and second-wave fears; the STOXX 600 fell 4.1 percent, its worst day "
        "in more than 11 weeks.",
        "https://www.cnbc.com/2020/06/11/european-markets-digest-fed-outlook.html",
    ),
    "2021-11-26": (
        "Omicron variant news; the DAX lost more than 3.2 percent between 25 and 26 November.",
        "https://www.aktienwelt360.de/2021/12/03/omikron-crash-oder-aktienboom/",
    ),
    "2026-05-06": (NOT_FOUND, ""),
    "2022-10-04": (
        "Rally day: the DAX gapped higher on 4 October (the model flags unusual moves in either "
        "direction).",
        "https://forex.com/en/news-and-analysis/european-open-dax-to-extend-its-countertrend-bounce-opec-in-focus",
    ),
    "2022-11-11": (
        "Day after a softer than expected US CPI print lifted the DAX by up to 3.2 percent "
        "intraday on 10 November. The 11 November move was not sourced.",
        "https://www.dailyfx.com/news/dax-40-explodes-after-us-inflation-shows-positive-signs-of-easing-20221110.html",
    ),
    "2019-08-02": (
        "US announced a 10 percent tariff on about 300 billion dollars of Chinese goods; the DAX "
        "fell 2.7 percent per the cited report.",
        "https://gulfnews.com/business/trump-tariff-salvo-pushes-stocks-into-sea-of-red-1.65596306",
    ),
    "2022-04-29": (NOT_FOUND, ""),
    "2019-09-13": (
        "Day after the ECB cut the deposit rate and restarted asset purchases; the STOXX 600 "
        "closed 0.4 percent higher.",
        "https://www.cnbc.com/2019/09/13/european-markets-stocks-react-to-ecb-stimulus-trade-war-easing.html",
    ),
    "2019-10-11": (
        "US China partial (phase one) trade deal announcement and positive Brexit news; the DAX "
        "was up about 1.4 percent mid-session.",
        "https://investinglive.com/news/!/equities-extend-gains-in-the-european-morning-20191011/",
    ),
    "2023-02-06": (NOT_FOUND, ""),
}


def write_review(cfg: Config, champion: str = "anomaly-isolation-forest", top: int = 20) -> str:
    d = pd.read_csv(cfg.path("results") / "test_false_alarms.csv")
    c = d[d.model == champion].sort_values("peak_score", ascending=False).head(top)
    lines = [
        "# Alarms outside the curated event list: review",
        "",
        f"Top {top} alarm episodes of the champion ({champion}) in the test period by peak score,"
        " excluding episodes that overlap a detection window or buffer. The event list is "
        "incomplete, so these are not necessarily errors. Descriptions come from the cited "
        "sources only; nothing is asserted without one.",
        "",
        "| Peak date | Episode | Days | Peak score | What happened | Source |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in c.iterrows():
        desc, src = ANNOTATIONS.get(str(r["peak_date"]), (NOT_FOUND, ""))
        link = f"[link]({src})" if src else "none"
        lines.append(
            f"| {r['peak_date']} | {r['start']} to {r['end']} | {int(r['days'])} | "
            f"{r['peak_score']:.3f} | {desc} | {link} |"
        )
    n_found = sum(1 for _, r in c.iterrows() if ANNOTATIONS.get(str(r["peak_date"]), ("", ""))[1])
    lines += [
        "",
        f"A source was found for {n_found} of {len(c)} episodes. Total alarm episodes outside the "
        f"event list for this model: {int((d.model == champion).sum())}.",
    ]
    text = "\n".join(lines) + "\n"
    (Path(cfg.root) / "docs" / "FALSE_ALARM_REVIEW.md").write_text(text, encoding="utf-8")
    return text
