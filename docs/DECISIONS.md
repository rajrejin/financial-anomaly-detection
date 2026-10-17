# Decisions

Each entry: decision, alternatives, reason.

1. **Project location**: `D:\Work\Code\financial-anomaly` (the session folder was an unrelated workspace). Alternative: build in place. Reason: keep the repository standalone.
2. **Python 3.13** with a local venv, CPU-only torch 2.14. Alternative 3.11. Reason: it is what the machine has; everything installed.
3. **DAX source**: yfinance `^GDAXI` worked; Stooq fallback unused. The current day is dropped to avoid incomplete bars.
4. **FX and yields** come from the Bundesbank (BBEX3, BBSIS Svensson 2y and 10y). The policy rates come from the ECB FM dataflow (MRO, DFR, MLF). A Bundesbank daily money market series was not searched for; the ECB rates cover short-term policy. €STR (EST dataflow) starts in Oct 2019, too short for 2000 to 2018 training, so it is not used.
5. **Forward fill**: observation series (FX, yields) fill at most 3 calendar days; ECB policy rates are step functions and are carried forward without a limit. No forward fill was actually needed for FX or yields on DAX trading days (see `results/data_quality.json`).
6. **Events**: onset is the first trading day of the market reaction. For the April 2025 tariffs (announced after the European close on 2 Apr) onset is 3 Apr; for SVB it is the closure date 10 Mar. No 2026 stress event could be verified by search (only moderate moves found), so none is included. 15 events, 6 in the test period. The flash crash, ECB hike and August 2024 events have weak daily equity signals (documented in `events.yaml`).
7. **Model inputs** are trailing 252 day median/MAD z-scores of the base features, shared by all families. Where MAD is zero (policy rate change) the window standard deviation is used, then 1.
8. **Threshold for the supervised model**: in-sample probabilities of the refit model cannot be calibrated to a false alarm budget, so the validation-stage out-of-sample threshold is reused (documented in the protocol).
9. **Line endings**: SHA-256 of text files normalises CRLF to LF so the freeze holds across platforms.
