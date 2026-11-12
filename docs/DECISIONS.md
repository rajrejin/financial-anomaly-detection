# Decisions

Each entry: decision, alternatives, reason.

1. **Python 3.13** with a local venv, CPU-only torch 2.14. Alternative 3.11. Reason: it is what the machine has; everything installed.
2. **DAX source**: yfinance `^GDAXI` worked; Stooq fallback unused. The current day is dropped to avoid incomplete bars.
3. **FX and yields** come from the Bundesbank (BBEX3, BBSIS Svensson 2y and 10y). The policy rates come from the ECB FM dataflow (MRO, DFR, MLF). A Bundesbank daily money market series was not searched for; the ECB rates cover short-term policy. €STR (EST dataflow) starts in Oct 2019, too short for 2000 to 2018 training, so it is not used.
4. **Forward fill**: observation series (FX, yields) fill at most 3 calendar days; ECB policy rates are step functions and are carried forward without a limit. No forward fill was actually needed for FX or yields on DAX trading days (see `results/data_quality.json`).
5. **Events**: onset is the first trading day of the market reaction. For the April 2025 tariffs (announced after the European close on 2 Apr) onset is 3 Apr; for SVB it is the closure date 10 Mar. No 2026 stress event could be verified by search (only moderate moves found), so none is included. 15 events, 6 in the test period. The flash crash, ECB hike and August 2024 events have weak daily equity signals (documented in `events.yaml`).
6. **Model inputs** are trailing 252 day median/MAD z-scores of the base features, shared by all families. Where MAD is zero (policy rate change) the window standard deviation is used, then 1.
7. **Threshold for the supervised model**: in-sample probabilities of the refit model cannot be calibrated to a false alarm budget, so the validation-stage out-of-sample threshold is reused (documented in the protocol).
8. **Line endings**: SHA-256 of text files normalises CRLF to LF so the freeze holds across platforms.
9. **Data start 1999** (not 2000): features need 252 days for the drawdown plus 252 days for the trailing z-scores, so with a 2000 start the September 2001 event would have had no features. Splits still start in 2000; features are valid from 2000-12.
10. **Unsupervised training excludes event windows** (detection windows and buffers of events with onset before the cut-off) so the models learn "normal" days. Alternative: train on all days. Reason: stress days are the object of detection.
11. **Alarm day cap**: calibration also limits alarm days to 5 percent of eligible days (`alarm.max_alarm_day_share`). Without it a persistent score (volatility percentile) satisfies an episode budget with one months-long alarm. Written into the protocol before the experiments.
12. **Registry uses aliases** (`staging`, `production`) plus a `lifecycle` tag, because MLflow 3.17 deprecates stages.
13. **Seed study spread criterion** is the population standard deviation of validation event recall over 5 seeds, at most 0.2.
14. **Gate uses the validation-stage fit** (data before 2015) because the deployed refit has no out-of-sample validation. The deployed model is the refit on all pre-2019 data.
15. **Platt calibration** (not isotonic) for the tree models: few positive days in the early-stopping block make isotonic regression unstable.
16. **Leave-one-event-out definition**: the held-out event is treated as ordinary data for unsupervised fitting and threshold calibration; LightGBM leaves its rows out and takes its threshold from out-of-fold probabilities (5 contiguous blocks). It is not chronological.
17. **Alarm day share metric** was added to `evaluate_segment` after the main test evaluation had run, because a degenerate always-alarm model shows zero false alarm episodes. The main test numbers are unchanged; the shares for them are computed from the stored test scores (`results/alarm_share.json`).
18. **Container**: Python 3.13 base to match the build interpreter; family-specific extras (torch, lightgbm, xgboost, shap) are installed only when the model needs them, which cuts the image from 1.89 GB to 1.06 GB.
