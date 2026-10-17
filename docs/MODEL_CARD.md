# Model card: promoted anomaly detector

<!-- BEGIN:modelcard -->
| Item | Value |
|---|---|
| Registered name and version | anomaly-isolation-forest version 1 (alias production) |
| Family and configuration | isolation_forest, configuration 4: `{'n_estimators': 300, 'max_samples': 1024, 'max_features': 0.7}` |
| Fit data | all data before 2019-01-01 (features valid from 2000-12) |
| Alarm threshold | 0.4854 (score scale of the model) |
| Validation (fit before 2015, 2015 to 2018) | recall 1.000, 2.49 false alarm episodes per year, AP 0.253 |
| Seed spread (5 seeds), test | recall 1.000 (std 0.000), FA per year 4.12 (std 0.35) |
| Test period (2019 onward) | 6 of 6 events, 4.24 false alarm episodes per year, AP 0.217 |
| Data hash | `c747aa62dba8e5fa` |
| Events hash | `aee345cc128dab34` |
| Git commit at export | `6ca68ee51b` |
<!-- END:modelcard -->

## Intended use

Flagging trading days on which German and euro-area daily market data look unusual relative to the
preceding year, as a monitoring aid and a teaching example of a model lifecycle. The score is
produced after the close of each day from that day's data. It detects stress after onset.

## Not intended for

Forecasting, trading or risk limits, investment decisions, or any use that needs timely warning
before a move. This is not financial advice.

## Inputs

Sixteen features computed from the DAX close, four euro reference exchange rates, the German 2 and
10 year Svensson yields and the ECB deposit rate, each expressed as a trailing 252 day median/MAD
z-score (clipped at 10) using only earlier data. See `src/finanomaly/features.py`.

## Output

A score (higher is more unusual), the alarm threshold fixed at training time, an alarm flag, and per
feature contributions (occlusion for the isolation forest; SHAP for tree models; error shares for the
autoencoder). Contributions describe model behaviour, not the cause of a market move.

## Training and evaluation data

Daily data from 1999 (features valid from the end of 2000) to the latest complete day, from the
Bundesbank, the ECB and Yahoo Finance. 15 curated stress events with sources in `data/events.yaml`,
used for threshold exclusion windows, supervised labels (LightGBM only) and evaluation.

## Performance

See the table above and `README.md`. With three validation and six test events the figures are
coarse; per-event outcomes are in `docs/PROMOTION_DECISION.md`, and alarms outside the curated list
are reviewed in `docs/FALSE_ALARM_REVIEW.md`.

## Known limitations

* The threshold was set on 2015 to 2018 data for a budget of 3 false alarm episodes per year; in
  the test period the false alarm rate was higher (see the table above and the README).
* The curated event list is incomplete, so alarms outside it are not necessarily errors.
* One market regime history; no data before 1999; daily resolution only.
* Data from an unofficial source for the DAX; revisions of Bundesbank series are not tracked.
* Explanations are model-behaviour summaries and have not been validated against market causes.
