# Evaluation protocol (pre-registered)

Written and committed before the first experiment run. Deviations are recorded at the end with reasons.

## Data and events
Daily series for 2000 to the latest complete day: DAX close (Yahoo Finance via yfinance), Bundesbank
FX reference rates and Svensson 2 and 10 year yields, ECB key interest rates. All series are aligned
on DAX trading days. The curated event list is `data/events.yaml` (15 events with sources). Its SHA-256
and that of the window and split configuration are stored in `results/events_frozen.json`; a test
fails if either changes.

## Windows (from `configs/default.yaml`)
* Label window: onset day plus the next 9 trading days (10 days labelled 1).
* Detection window: 2 trading days before onset to 20 trading days after onset.
* Buffer: 10 trading days after each detection window, excluded from false alarm counting.

## Splits
* Fit period: 2000 up to 2014-12-31 for hyperparameter tuning and threshold calibration inputs.
* Validation: 2015-01-01 to 2018-12-31 (three events: SNB floor, China sell-off, Brexit).
* Final refit: all pre-2019 data. Test: 2019-01-01 to the end of the data (six events).
* Test results are examined once, after the promotion decision. Anything chosen with test
  information is labelled "oracle".

## Features
Computed from data up to and including day t only (see `features.py` and the leakage test). Model
inputs are trailing 252 day median/MAD z-scores of 18 base features, computed from the 252 days
before t. No full-sample statistics are used.

## Models and configurations
Isolation Forest, a PyTorch autoencoder, LightGBM (supervised on label windows of events with
onset before the training cut-off), and an XGBoost group for comparison, each with 7 fixed
configurations in `configs/experiments.yaml`. Two rule baselines (robust return threshold, volatility
percentile) are evaluated with identical code.

## Alarms and thresholds
Alarm threshold set per model on pre-2019 non-event, non-buffer days so that the false alarm
episode rate does not exceed 3 episodes per year. For tuning, models are fit before 2015 and
thresholds calibrated on 2015 to 2018 non-event days. For the supervised model, the final refit
uses the validation-stage probability threshold (in-sample probabilities are not usable for
calibration); unsupervised refits calibrate on the 2015 to 2018 non-event days of the refit model.
Calibration also caps the share of alarm days among eligible days at 5 percent
(`alarm.max_alarm_day_share`): without it a persistent score such as a volatility percentile meets
an episode budget with one months-long alarm. Alarm episodes: maximal alarm runs, merged when separated by fewer than 3 non-alarm days.

## Metrics
Event recall, per-event detection delay (trading days from onset to the first alarm in the
detection window), false alarm episodes per year, average precision and ROC AUC on daily labels
(coarse), score separation (standardised mean difference), and for LightGBM log loss and
calibration. With so few events no confidence intervals over events are reported. Unlabelled alarms
are "alarms outside the curated event list", not necessarily errors.

## Selection and promotion
Per family, the best configuration is the one with the highest validation event recall at the false
alarm budget; ties are broken by lower validation false alarm rate, then higher validation average
precision, then simpler model (fewer parameters). The best of each family is registered and given the
`staging` alias. The promotion gate promotes a candidate to the `production` alias only if:

1. its validation event recall is at least that of the current production model (or, if none exists,
   of the best rule baseline) and at least that of the best rule baseline;
2. its validation false alarm rate is within the budget (3 per year, plus a tolerance of 0.5);
3. the standard deviation of its validation event recall over 5 seeds is at most 0.2;
4. explanations are available (SHAP additivity check for trees, contribution shares for the
   autoencoder, permutation sensitivity for Isolation Forest);
5. it passes the serving contract test (loads, scores a real recent day, validates the input schema).

Among passing candidates the champion is the one with highest validation recall, then lower false
alarm rate, then higher validation average precision. Test results are not used. They are reported
after the decision in `docs/PROMOTION_DECISION.md`, including if the champion performs poorly.

## Stability and robustness
Five seeds for the best configuration of each family (mean and spread; differences smaller than the
spread are not claimed); rank stability of the 7 configurations between validation and test;
leave-one-event-out table (supplementary); sensitivity to window lengths, to the false alarm budget
and to the training start (labelled analyses, not used to retune); feature group ablation;
walk-forward annual refit versus the fixed schedule.

## Deviations
(none yet)
