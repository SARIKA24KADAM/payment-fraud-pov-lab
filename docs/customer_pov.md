# Customer proof-of-value plan

## Business question

For a hypothetical online payments workflow, do identity-related signals add useful fraud-ranking information beyond transaction attributes, and what fraud-capture versus legitimate-customer-friction trade-offs appear at candidate review thresholds?

This project can support a technical discussion. It cannot prove customer value because the benchmark is not the customer's traffic, fraud labels, economics, or operating environment.

## Proposed POV workflow

1. **Clarify the decision.** Agree what “fraud captured,” “false positive,” “review,” and “approved” mean in the customer's workflow. Identify the action triggered by each score band.
2. **Verify data and labels.** Confirm transaction keys, timestamps, fraud-label maturity, join coverage, missingness, and whether the identity record is available before the payment decision.
3. **Establish a baseline.** Score transaction-only data with a regularized logistic-regression model.
4. **Test incremental signal.** Repeat the same time split and modeling approach with identity attributes added. Compare ranking and operating points.
5. **Review operating trade-offs.** Compare fraud capture, false-positive rate, legitimate approval rate, precision, and the share flagged for review at several thresholds. Select thresholds with the customer using their costs and capacity; this repository does not optimize an assumed cost function.
6. **QA integration.** Validate request/response schema, key matching, missing and unavailable signals, latency, and consistent event-time interpretation.
7. **Present findings.** State the tested population and period, exclusions, join coverage, uncertainty, recommended next experiment, and limitations.

## Metrics

- **Average precision (PR-AUC):** useful for comparing ranking under class imbalance; depends on prevalence.
- **Fraud capture rate (recall):** fraction of labeled fraud transactions flagged at a selected threshold.
- **False-positive rate:** fraction of legitimate transactions flagged.
- **Legitimate approval rate:** fraction of legitimate transactions below the threshold.
- **Precision:** fraction of flagged transactions labeled fraud.
- **Review-or-decline share:** fraction of all transactions at or above the threshold. Whether those are reviewed or declined is a workflow decision.

Metrics are reported on a later time period than training. The pipeline's threshold table is an exploratory operating curve; thresholds are not selected on a separate validation set, so use a three-way train/validation/test design before making a final threshold recommendation.

## Caveats and responsible use

The IEEE-CIS dataset is an anonymized competition benchmark. It may contain engineered or obfuscated features and does not guarantee that a field is available at authorization time. Temporal ordering is based on the provided relative time field. Results are not evidence of performance for Mastercard or any merchant.

Before any real use, require representative authorized data, label maturity checks, fraud-loss and review-cost definitions, privacy and security review, jurisdiction-specific compliance review, fairness analysis where applicable, independent validation, production monitoring, latency testing, and an agreed fallback for missing signals. Never use this demonstration to make actual payment decisions.
