# Known issues

1. **Test evaluation ran twice.** `finanomaly evaluate` was executed once and crashed in the seed stability section (a loop variable bug introduced by an automated rename) after the first part had already written the test summary. It was fixed and rerun; the model, threshold and test code for the first part were unchanged, so the outputs are identical. No decision depended on the first run, and the first run's printed numbers were not read before the rerun.
2. **Very few events.** Three validation events and six test events: recall takes coarse values and ties dominate model selection. Per-event tables matter more than aggregates.
3. **Champion threshold drift.** The isolation forest threshold, set for 3 false alarm episodes per year on 2015 to 2018, produced more false alarm episodes per year on the test period (see `docs/PROMOTION_DECISION.md`).
