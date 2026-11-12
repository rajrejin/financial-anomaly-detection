# Full results

All tables here were produced by the pipeline from the files in `results/`. The README only
shows the headline numbers.

## Data series

| Series | Source | Flow | Key | First | Last | Observations |
|---|---|---|---|---|---|---|
| usd_per_eur | Bundesbank | BBEX3 | `D.USD.EUR.BB.AC.000` | 1999-01-04 | 2026-10-16 | 7116 |
| chf_per_eur | Bundesbank | BBEX3 | `D.CHF.EUR.BB.AC.000` | 1999-01-04 | 2026-10-16 | 7116 |
| gbp_per_eur | Bundesbank | BBEX3 | `D.GBP.EUR.BB.AC.000` | 1999-01-04 | 2026-10-16 | 7116 |
| jpy_per_eur | Bundesbank | BBEX3 | `D.JPY.EUR.BB.AC.000` | 1999-01-04 | 2026-10-16 | 7116 |
| de_yield_2y | Bundesbank | BBSIS | `D.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A` | 1999-01-04 | 2026-10-16 | 7060 |
| de_yield_10y | Bundesbank | BBSIS | `D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A` | 1999-01-04 | 2026-10-16 | 7060 |
| ecb_mro | ECB | FM | `D.U2.EUR.4F.KR.MRR_FR.LEV` | 1999-01-01 | 2026-10-17 | 7121 |
| ecb_dfr | ECB | FM | `D.U2.EUR.4F.KR.DFR.LEV` | 1999-01-01 | 2026-10-17 | 10152 |
| ecb_mlf | ECB | FM | `D.U2.EUR.4F.KR.MLFR.LEV` | 1999-01-01 | 2026-10-17 | 10152 |
| dax_close | Yahoo Finance via yfinance | nan | `^GDAXI` | 1999-01-04 | 2026-10-16 | 7057 |

## Stress events

| Id | Event | Category | Onset | Sources |
|---|---|---|---|---|
| sept11_2001 | September 11 attacks | systemic | 2001-09-11 | [1](https://money.cnn.com/2001/09/14/europe/markets_europe/), [2](https://money.cnn.com/2001/09/17/europe/markets_europe/) |
| worldcom_2002 | WorldCom bankruptcy and the July 2002 sell-off | equity | 2002-07-22 | [1](https://www.upi.com/Business_News/2002/07/22/Stocks-fall-sharply/99731027354751/), [2](https://www.bis.org/publ/qtrpdf/r_qt0209ger_a.pdf) |
| lehman_2008 | Lehman Brothers bankruptcy | systemic | 2008-09-15 | [1](https://www.nber.org/papers/w22410.pdf), [2](https://www.history.com/this-day-in-history/lehman-brothers-collapses) |
| crash_oct2008 | October 2008 global equity crash | equity | 2008-10-06 | [1](https://www.thelocal.de/20081006/14719), [2](https://www.t-online.de/finanzen/ratgeber/geldanlage/boerse/id_100670460/dax-stuerzt-ab-dunkelste-tage-des-deutschen-leitindex-im-rueckblick.html) |
| flash_crash_2010 | Flash crash and May 2010 euro area stress | equity | 2010-05-06 | [1](https://www.csmonitor.com/Money/2010/0519/SEC-unveils-flash-crash-fixes-but-Dow-plunge-still-mysterious), [2](https://www.planadviser.com/regulators-release-flash-crash-probe-report/) |
| aug2011_selloff | August 2011 US downgrade and euro crisis sell-off | systemic | 2011-08-08 | [1](https://en.wikipedia.org/wiki/August_2011_stock_markets_fall), [2](https://www.wbur.org/news/2011/08/08/stocks-tumble-downgrade) |
| snb_floor_2015 | SNB removes the EUR/CHF floor | fx | 2015-01-15 | [1](https://www.bis.org/review/r150116a.htm), [2](https://www.ft.lk/Financial-Services/switzerland-ends-euro-minimum-exchange-rate-cuts-interest-rate/42-382312) |
| china_aug2015 | China devaluation sell-off | equity | 2015-08-24 | [1](https://www.cityam.com/ftse-edges-towards-6000-after-china-shock/), [2](https://fortune.com/2015/08/24/dow-plunges-1000-points-stocks-markets) |
| brexit_2016 | Brexit referendum result | systemic | 2016-06-24 | [1](https://www.nbcnews.com/storyline/brexit-referendum/brexit-prospect-sends-pound-35-year-low-pummeling-markets-n598096), [2](https://www.ii.co.uk/analysis-commentary/stockmarkets-and-pound-plunge-black-friday-britain-votes-brexit-ii331341) |
| covid_crash_2020 | COVID-19 crash | systemic | 2020-02-24 | [1](https://www.cnbc.com/2020/02/24/european-markets-focus-on-coronavirus-spread-in-italy.html), [2](https://fox6now.com/2020/03/12/global-shares-sink-on-pandemic-news-waning-hopes-for-fix/) |
| ukraine_invasion_2022 | Russian invasion of Ukraine | systemic | 2022-02-24 | [1](https://www.business-standard.com/amp/article/international/european-stocks-in-correction-territory-after-russia-attacks-ukraine-122022400743_1.html), [2](https://gna.org.gh/2022/02/world-markets-dive-oil-soars-as-russia-invades-ukraine/) |
| ecb_hike_2022 | First ECB rate hike of the 2022 cycle | rates | 2022-07-21 | [1](https://www.fxstreet.com/amp/news/breaking-ecb-hikes-key-rates-by-50-basis-points-in-july-202207211216), [2](https://schulung.table.media/en/europe/news/ecb-raises-interest-rates-surprisingly-sharply) |
| svb_credit_suisse_2023 | Silicon Valley Bank failure and Credit Suisse turmoil | systemic | 2023-03-10 | [1](https://www.euronews.com/2023/03/15/european-banks-plunge-in-the-stock-market-dragged-down-by-credit-suisse), [2](https://www.kitco.com/news/off-the-wire/2023-03-15/credit-suisse-leads-europe-bank-stock-rout-after-short-lived-svb) |
| carry_unwind_2024 | August 2024 yen carry trade unwind sell-off | fx | 2024-08-05 | [1](https://www.nbcnewyork.com/news/business/money-report/investors-are-unwinding-the-biggest-carry-trade-the-world-has-ever-seen-socgen-strategist-says/5671777/), [2](https://www.atalayar.com/en/articulo/economy-and-business/uncertainty-over-the-us-economy-and-the-yen-to-blame-for-global-stock-markets-black-monday/20240806111804203876.html) |
| tariff_shock_2025 | April 2025 tariff shock | systemic | 2025-04-03 | [1](https://stock3.com/news/liberation-day-handelsaengste-belasten-den-dax-16258502), [2](https://www.pennmutualam.com/market-insights/blogs/monday-morning-perspectives/2025-04-07-markets-likely-to-continue-to-reel-from-liberation-day-tariffs) |

## Validation, all configurations (fit before 2015, scored on 2015 to 2018)

| Family | Config | Val recall | Val FA/yr | Val AP | Val ROC AUC | Separation |
|---|---|---|---|---|---|---|
| autoencoder | 0 | 0.667 | 2.76 | 0.175 | 0.775 | 0.66 |
| autoencoder | 1 | 0.667 | 2.76 | 0.223 | 0.773 | 0.73 |
| autoencoder | 2 | 0.667 | 2.76 | 0.205 | 0.861 | 1.06 |
| autoencoder | 3 | 0.667 | 2.76 | 0.212 | 0.874 | 1.06 |
| autoencoder | 4 | 0.667 | 2.76 | 0.218 | 0.842 | 0.83 |
| autoencoder | 5 | 0.667 | 2.76 | 0.254 | 0.853 | 0.94 |
| autoencoder | 6 | 0.667 | 2.76 | 0.104 | 0.707 | 0.48 |
| isolation_forest | 0 | 1.000 | 2.76 | 0.221 | 0.872 | 1.41 |
| isolation_forest | 1 | 1.000 | 2.76 | 0.256 | 0.880 | 1.48 |
| isolation_forest | 2 | 1.000 | 2.76 | 0.305 | 0.885 | 1.55 |
| isolation_forest | 3 | 1.000 | 2.76 | 0.228 | 0.865 | 1.35 |
| isolation_forest | 4 | 1.000 | 2.49 | 0.253 | 0.865 | 1.30 |
| isolation_forest | 5 | 1.000 | 2.76 | 0.238 | 0.868 | 1.44 |
| isolation_forest | 6 | 1.000 | 2.76 | 0.257 | 0.862 | 1.27 |
| lightgbm | 0 | 1.000 | 2.76 | 0.224 | 0.837 | 0.97 |
| lightgbm | 1 | 1.000 | 2.76 | 0.320 | 0.745 | 0.93 |
| lightgbm | 2 | 1.000 | 2.49 | 0.209 | 0.898 | 1.04 |
| lightgbm | 3 | 0.667 | 2.49 | 0.317 | 0.811 | 0.99 |
| lightgbm | 4 | 0.667 | 2.49 | 0.300 | 0.830 | 1.07 |
| lightgbm | 5 | 1.000 | 2.76 | 0.368 | 0.810 | 0.98 |
| lightgbm | 6 | 1.000 | 2.49 | 0.181 | 0.831 | 1.00 |
| xgboost | 0 | 1.000 | 2.76 | 0.381 | 0.862 | 1.07 |
| xgboost | 1 | 1.000 | 2.49 | 0.374 | 0.889 | 0.95 |
| xgboost | 2 | 1.000 | 2.76 | 0.397 | 0.880 | 1.16 |
| xgboost | 3 | 1.000 | 2.76 | 0.368 | 0.864 | 0.93 |
| xgboost | 4 | 1.000 | 2.76 | 0.425 | 0.926 | 0.99 |
| xgboost | 5 | 1.000 | 2.76 | 0.407 | 0.920 | 0.92 |
| xgboost | 6 | 1.000 | 2.76 | 0.367 | 0.909 | 1.21 |
| rule_return | rule | 0.667 | 2.76 | 0.118 | 0.676 | 0.63 |
| rule_vol | rule | 1.000 | 1.66 | 0.464 | 0.960 | 2.43 |

## Test period (2019 onward)

| Event | rule_return | rule_vol | anomaly-isolation-forest (champion) | anomaly-autoencoder | anomaly-lightgbm |
|---|---|---|---|---|---|
| covid_crash_2020 | day +0 | day +1 | day +0 | day +10 | day +0 |
| ukraine_invasion_2022 | day +0 | missed | day +0 | day +0 | day +0 |
| ecb_hike_2022 | missed | missed | day +9 | day +4 | missed |
| svb_credit_suisse_2023 | missed | missed | day +1 | day +1 | day +1 |
| carry_unwind_2024 | day -2 | missed | day -1 | missed | day -2 |
| tariff_shock_2025 | day +0 | day +1 | day +0 | day +2 | day -2 |

Delay is in trading days from onset to the first alarm in the detection window.

| Entry | Detected | Recall | FA episodes/yr | AP | ROC AUC |
|---|---|---|---|---|---|
| rule_return | 4 of 6 | 0.667 | 2.83 | 0.149 | 0.674 |
| rule_vol | 2 of 6 | 0.333 | 0.42 | 0.149 | 0.802 |
| anomaly-isolation-forest (champion) | 6 of 6 | 1.000 | 4.24 | 0.217 | 0.878 |
| anomaly-autoencoder | 5 of 6 | 0.833 | 4.10 | 0.115 | 0.828 |
| anomaly-lightgbm | 5 of 6 | 0.833 | 1.41 | 0.233 | 0.855 |

## Seed spread and rank stability

| Family | Val recall mean (std) | Test recall mean (std) | Test FA/yr mean (std) | Test AP mean (std) |
|---|---|---|---|---|
| isolation_forest | 1.000 (0.000) | 1.000 (0.000) | 4.12 (0.35) | 0.224 (0.005) |
| autoencoder | 0.667 (0.000) | 0.900 (0.082) | 4.89 (0.76) | 0.118 (0.016) |
| lightgbm | 0.933 (0.133) | 0.967 (0.067) | 2.49 (1.08) | 0.267 (0.039) |

Spearman rank correlation of the 7 configurations, validation versus test:

| Family | AP | Event recall |
|---|---|---|
| autoencoder | 0.43 | n/a |
| isolation_forest | -0.14 | n/a |
| lightgbm | 0.86 | 0.49 |

## Robustness checks

### Leave-one-event-out

| Held-out event | rule_return | rule_vol | isolation_forest | autoencoder | lightgbm |
|---|---|---|---|---|---|
| sept11_2001 | +0 | +0 | +0 | +0 | +0 |
| worldcom_2002 | +5 | -2 | +1 | +1 | -2 |
| lehman_2008 | +4 | +4 | +4 | +15 | +0 |
| crash_oct2008 | +0 | -2 | +0 | +2 | -2 |
| flash_crash_2010 | +2 | missed | +1 | missed | +2 |
| aug2011_selloff | -2 | missed | -2 | missed | -1 |
| snb_floor_2015 | missed | -2 | +6 | +4 | -1 |
| china_aug2015 | +0 | +0 | +0 | missed | +1 |
| brexit_2016 | +0 | +0 | +0 | +0 | +0 |
| covid_crash_2020 | +0 | +0 | +4 | +11 | +0 |
| ukraine_invasion_2022 | +0 | +9 | +0 | +3 | +1 |
| ecb_hike_2022 | missed | missed | +4 | +4 | missed |
| svb_credit_suisse_2023 | missed | missed | +1 | missed | +6 |
| carry_unwind_2024 | -2 | missed | -1 | +0 | -2 |
| tariff_shock_2025 | +0 | +1 | +0 | +1 | +0 |

Each event is held out in turn: unsupervised models and rule thresholds are fit and calibrated with the event treated as ordinary data (budget 3 per year over the whole sample); LightGBM is trained on the other events' labels without the held-out rows, with its threshold from out-of-fold probabilities (5 time blocks). Not chronological, so later events inform earlier ones. Detected events per model: autoencoder 11 of 15, isolation_forest 15 of 15, lightgbm 14 of 15, rule_return 12 of 15, rule_vol 10 of 15.

### Sensitivity to windows, budget and training start

| setting | model | val_recall | test_recall | test_fa_per_year | test_ap | test_alarm_days |
|---|---|---|---|---|---|---|
| base | isolation forest | 1.000 | 1.000 | 4.24 | 0.217 | 8.3% |
| base | lightgbm | 1.000 | 0.833 | 1.41 | 0.233 | 14.3% |
| label window 5 days | isolation forest | 1.000 | 1.000 | 4.24 | 0.115 | 8.3% |
| label window 5 days | lightgbm | 1.000 | 0.833 | 3.96 | 0.139 | 5.9% |
| label window 20 days | isolation forest | 1.000 | 1.000 | 4.24 | 0.419 | 8.3% |
| label window 20 days | lightgbm | 1.000 | 0.667 | 1.13 | 0.428 | 15.2% |
| detection window after onset 10 days | isolation forest | 1.000 | 1.000 | 4.78 | 0.211 | 9.2% |
| detection window after onset 10 days | lightgbm | 1.000 | 0.833 | 1.50 | 0.233 | 14.3% |
| detection window after onset 40 days | isolation forest | 1.000 | 1.000 | 3.79 | 0.221 | 8.9% |
| detection window after onset 40 days | lightgbm | 1.000 | 0.833 | 1.51 | 0.233 | 14.3% |
| false alarm budget 1 per year | isolation forest | 0.667 | 0.667 | 0.85 | 0.217 | 2.5% |
| false alarm budget 1 per year | lightgbm | 0.000 | 0.000 | 0.00 | 0.233 | 0.0% |
| false alarm budget 6 per year | isolation forest | 1.000 | 1.000 | 5.51 | 0.217 | 12.3% |
| false alarm budget 6 per year | lightgbm | 1.000 | 1.000 | 4.24 | 0.233 | 19.5% |
| training start 2010 | isolation forest | 1.000 | 1.000 | 4.80 | 0.225 | 8.6% |
| training start 2010 | lightgbm | 0.667 | 0.833 | 1.55 | 0.197 | 15.7% |

Labelled sensitivity analysis on the registered configurations (seed 0); nothing here was used to retune. Window changes alter labels, detection windows and buffers together. Test AP is on daily labels under the changed label window, so it is not comparable across label window settings. Only 3 validation and 6 test events.

### Cross-checks between models

| pair | spearman_test_days | episodes_a | episodes_b | a_overlapping_b | b_overlapping_a |
|---|---|---|---|---|---|
| rule_return vs rule_vol | 0.27 | 26 | 5 | 6 | 3 |
| rule_return vs anomaly-isolation-forest | 0.35 | 26 | 40 | 17 | 12 |
| rule_return vs anomaly-autoencoder | 0.36 | 26 | 42 | 5 | 6 |
| rule_return vs anomaly-lightgbm | 0.29 | 26 | 18 | 17 | 6 |
| rule_vol vs anomaly-isolation-forest | 0.37 | 5 | 40 | 4 | 6 |
| rule_vol vs anomaly-autoencoder | 0.39 | 5 | 42 | 4 | 8 |
| rule_vol vs anomaly-lightgbm | 0.43 | 5 | 18 | 5 | 4 |
| anomaly-isolation-forest vs anomaly-autoencoder | 0.90 | 40 | 42 | 29 | 36 |
| anomaly-isolation-forest vs anomaly-lightgbm | 0.49 | 40 | 18 | 17 | 12 |
| anomaly-autoencoder vs anomaly-lightgbm | 0.39 | 42 | 18 | 18 | 9 |

Rank correlation of daily scores in the test period and overlap of alarm episodes (episodes of A that overlap at least one episode of B), all alarms including those inside event windows.

### Share of alarm days in the test period

| entry | alarm_days | share_of_test_days | episodes | mean_episode_days |
|---|---|---|---|---|
| rule_return | 50 | 2.5% | 26 | 2.8 |
| rule_vol | 97 | 4.9% | 5 | 19.8 |
| anomaly-isolation-forest | 165 | 8.3% | 40 | 5.1 |
| anomaly-autoencoder | 86 | 4.3% | 42 | 2.5 |
| anomaly-lightgbm | 283 | 14.3% | 18 | 16.3 |

Alarm days in the test period at each entry's fixed threshold (read from the stored scores, no refit). The false alarm episode metric ignores episodes that overlap an event window, so the share of alarm days is reported alongside it.

### Feature group ablation

| features | n_features | model | val_recall | val_ap | test_recall | test_fa_per_year | test_ap | test_alarm_days |
|---|---|---|---|---|---|---|---|---|
| equity only | 6 | isolation forest | 0.667 | 0.277 | 0.833 | 2.26 | 0.155 | 10.2% |
| equity only | 6 | autoencoder | 0.333 | 0.045 | 0.833 | 1.98 | 0.040 | 9.4% |
| equity only | 6 | lightgbm | 1.000 | 0.204 | 0.667 | 1.84 | 0.204 | 7.4% |
| equity + fx | 11 | isolation forest | 1.000 | 0.374 | 0.833 | 2.40 | 0.209 | 7.0% |
| equity + fx | 11 | autoencoder | 1.000 | 0.379 | 0.667 | 5.09 | 0.123 | 5.7% |
| equity + fx | 11 | lightgbm | 1.000 | 0.221 | 0.667 | 1.98 | 0.191 | 7.8% |
| equity + rates | 11 | isolation forest | 1.000 | 0.192 | 1.000 | 4.66 | 0.205 | 10.4% |
| equity + rates | 11 | autoencoder | 1.000 | 0.118 | 0.833 | 6.22 | 0.103 | 10.4% |
| equity + rates | 11 | lightgbm | 0.667 | 0.280 | 0.667 | 3.11 | 0.187 | 10.4% |
| fx + rates (no equity) | 10 | isolation forest | 0.667 | 0.151 | 1.000 | 3.25 | 0.206 | 3.4% |
| fx + rates (no equity) | 10 | autoencoder | 0.667 | 0.159 | 0.833 | 5.09 | 0.104 | 4.0% |
| fx + rates (no equity) | 10 | lightgbm | 0.333 | 0.049 | 1.000 | 0.00 | 0.022 | 100.0% |
| all features | 16 | isolation forest | 1.000 | 0.253 | 1.000 | 4.24 | 0.217 | 8.3% |
| all features | 16 | autoencoder | 0.667 | 0.254 | 0.833 | 4.10 | 0.115 | 4.3% |
| all features | 16 | lightgbm | 1.000 | 0.209 | 0.833 | 1.41 | 0.233 | 14.3% |

Feature group ablation with the registered configurations (seed 0), same protocol as the main pipeline. Differences are small relative to the seed spread and the 3 and 6 event counts and are not claimed as effects. The share of alarm days matters: a model that alarms on most days has a low false alarm episode count because one long episode overlaps an event window.

### Walk-forward refit versus fixed fit

| model | schedule | detected | test_fa_per_year | test_ap |
|---|---|---|---|---|
| isolation forest | fixed (fit before 2019) | 6 of 6 | 4.24 | 0.217 |
| isolation forest | walk-forward, annual refit | 6 of 6 | 4.52 | 0.196 |
| autoencoder | fixed (fit before 2019) | 5 of 6 | 4.10 | 0.115 |
| autoencoder | walk-forward, annual refit | 3 of 6 | 4.10 | 0.089 |

Walk-forward: before each test year the model is refit on all earlier data and its threshold recalibrated on the previous four years (non-event days, same budget). The fixed schedule used for promotion is shown for comparison; it was not changed.

## Explainability checks

| Check | LightGBM | XGBoost |
|---|---|---|
| Max SHAP additivity gap | 1.78e-15 | 3.34e-06 |
| Spearman of mean abs SHAP vs gain importance | 0.82 | 0.44 |
| Top 3 features by SHAP | vol20, r_jpy, vol_ratio | vol20, vol_ratio, corr60 |
| Top 3 features by gain | vol20, r_jpy, d_spread | vol20, vol5, r_jpy |

Isolation forest permutation sensitivity (test AP 0.217), largest AP drops:

| Feature | AP drop mean | std |
|---|---|---|
| vol_ratio | 0.014 | 0.002 |
| vol5 | 0.012 | 0.007 |
| policy_rate | 0.010 | 0.002 |
| r_dax | 0.009 | 0.004 |
| corr60 | 0.007 | 0.002 |

## Is the recall better than chance?

Event recall counts an event as detected if any alarm falls in its 23 trading day detection window,
so a model that alarms often hits windows by chance. `scripts/audit_chance_level.py` measures, on
test days outside all event windows, how often a random window of that length contains an alarm,
and writes `results/audit_chance_level.json`.

| Entry | Alarm days | Detected (of 6) | Expected by chance | Detected within 2 days of onset | Expected by chance |
|---|---|---|---|---|---|
| rule_return | 2.5% | 4 | 1.14 | 4 | 0.35 |
| rule_vol | 4.9% | 2 | 0.34 | 2 | 0.19 |
| isolation forest (champion) | 8.3% | 6 | 1.88 | 5 | 0.78 |
| autoencoder | 4.3% | 5 | 1.59 | 3 | 0.57 |
| LightGBM | 14.3% | 5 | 1.15 | 5 | 0.76 |

All entries detect more than chance (binomial tail p-values are in the JSON file; they assume
independent events, which holds only approximately).

## Matched alarm budget (oracle analysis)

Thresholds set on the test scores so that every entry alarms on the same share of test days. This
uses test information and is for interpretation only; it changes no decision.

| Alarm-day share | rule_return | isolation forest | autoencoder | LightGBM | rule_vol |
|---|---|---|---|---|---|
| 2.5% (detected, fast) | 4, 4 | 4, 3 | 4, 0 | 3, 3 | 2, 1 |
| 5.0% | 5, 5 | 6, 4 | 5, 3 | 4, 4 | 2, 2 |
| 8.3% | 6, 6 | 6, 5 | 6, 5 | 4, 4 | 3, 2 |
