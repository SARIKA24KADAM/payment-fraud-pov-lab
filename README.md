# Payment Fraud Proof-of-Value Lab

A reproducible demonstration of how transaction and identity signals can affect payment fraud decisions. It compares models trained on transaction-only features with models trained on transaction plus identity features, then summarizes the fraud capture and legitimate-payment friction trade-offs at configurable review thresholds.

> **Scope:** This is a portfolio project using the public IEEE-CIS Fraud Detection benchmark. It does not use Mastercard data, represent a real client deployment, or establish production performance. The dataset is available through the [Kaggle competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and requires accepting its terms.

## What it demonstrates

- A chronological holdout split to reduce leakage from random splitting.
- Paired feature-set experiments: transaction-only versus transaction + identity.
- An interpretable logistic-regression baseline with class weighting.
- Precision-recall metrics and operating-point tables focused on fraud capture, false positives, and approval rate.
- A mock identity-signal API contract and integration QA checks.
- A customer-style proof-of-value brief with explicit assumptions and limitations.

## Quick start

Python 3.10+ is recommended. Download the competition files and place these in `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

The project never downloads data automatically. Kaggle access and data terms are your responsibility.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m payment_fraud_pov.pipeline --data-dir data/raw --output-dir reports
```

The pipeline writes summary metrics and threshold scenarios to `reports/`. It does not write transaction-level predictions by default.

## Results to review

- `reports/model_comparison.csv`: holdout PR-AUC and ROC-AUC for both feature sets.
- `reports/operating_points.csv`: precision, recall (fraud capture), false-positive rate, and approval rate at each threshold.

Do not claim lift until you run the analysis and calculate a difference on your downloaded dataset. Results depend on the chronological split, preprocessing, and selected thresholds. The data are anonymized competition data, not representative of a specific payment processor or customer.

## Repository map

- `src/payment_fraud_pov/pipeline.py`: data preparation, temporal split, paired models, metrics, operating-point outputs.
- `src/payment_fraud_pov/api_contract.py`: mock identity API schema and response validation.
- `docs/customer_pov.md`: customer-style POV plan and interpretation guide.
- `docs/api_contract.md`: example request/response and integration checks.

## Suggested extension

Add a validation period before the final holdout to select thresholds, then compare the selected operating points on the untouched test period. For a real deployment, evaluation would also require an agreed fraud-cost matrix, representative customer data, operational latency and availability targets, privacy review, monitoring, and independent model validation.
