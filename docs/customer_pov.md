# Customer proof-of-value plan

## What this demo shows

The current run uses Kaggle's simulated `card_transdata.csv` benchmark to demonstrate fraud ranking and the trade-off between fraud capture and legitimate transactions flagged for review. Its stratified random split is useful for a basic model demonstration, but it does not show future-in-time performance, customer value, or the incremental value of identity signals.

## A real customer evaluation

1. **Agree on the decision.** Define fraud captured, false positive, review, and approval in the customer's workflow, including what action each score band triggers.
2. **Confirm the data.** Use authorized representative data. Check transaction keys, event timestamps, label maturity, missingness, and whether fields are available before the payment decision.
3. **Set a baseline.** Agree on a simple baseline and a chronological train, validation, and test design that reflects the intended use.
4. **Test added signals.** If identity signals are available and approved for evaluation, compare transaction-only features against transaction plus identity features on the same partitions.
5. **Review operating trade-offs.** Compare fraud capture, false-positive rate, legitimate approval rate, precision, and review volume. Select thresholds with the customer's actual costs and capacity.
6. **Check integration.** Validate schema, key matching, missing signals, latency, event-time handling, and fallback behavior.
7. **Report limits clearly.** Describe the population, time period, exclusions, join coverage, uncertainty, and next experiment.

## Metrics

- **Average precision (PR-AUC):** summarizes ranking quality and is useful under class imbalance; it depends on fraud prevalence.
- **Fraud capture (recall):** share of labeled fraud transactions flagged at a threshold.
- **False-positive rate:** share of legitimate transactions flagged.
- **Legitimate approval rate:** share of legitimate transactions below the threshold.
- **Precision:** share of flagged transactions that are labeled fraud.
- **Review-or-decline share:** share of all transactions at or above the threshold; the workflow determines what happens to them.

## Limits and responsible use

The current benchmark is simulated and does not represent a specific processor or merchant. Its fields and score behavior may differ from a real payment workflow. Never use this demo to make actual payment decisions. A real evaluation needs authorized representative data, label and field-availability checks, agreed fraud and review costs, privacy and security review, fairness review where applicable, independent validation, monitoring, latency testing, and an agreed fallback for missing signals.
