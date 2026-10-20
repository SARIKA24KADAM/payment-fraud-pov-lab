# Payment Fraud Proof-of-Value Lab

This project is a small, reproducible study of whether identity-related information can help rank potentially fraudulent payments. It compares a model using transaction information with one using transaction and identity information, then shows how different score thresholds change fraud capture and legitimate-payment approvals.

## Data used

The analysis is designed for the [IEEE-CIS Fraud Detection dataset on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data). Kaggle requires an account and acceptance of the competition terms. The dataset is not included in this repository, and no dataset files or analysis results were present when this README was written. You need to download the files yourself:

| File | What it contains | Required columns |
| --- | --- | --- |
| `train_transaction.csv` | Transaction attributes, event time, and fraud label | `TransactionID`, `TransactionDT`, `isFraud` |
| `train_identity.csv` | Optional identity attributes for some transactions | `TransactionID` |

Put both files in `data/raw/`. The transaction file is required. The identity file is optional, but without it the comparison cannot show the contribution of identity data. The pipeline joins identity records to transactions using `TransactionID`, and stops if the identity file contains duplicate keys.

These are anonymized competition data, not Mastercard data and not data from a real customer. Do not put private customer information in this repository. The downloaded competition files and generated reports are ignored by Git.

## How the analysis works

The pipeline sorts transactions by `TransactionDT` and uses the earliest 80% for training and the latest 20% for holdout, keeping equal timestamps on the same side of the boundary. `TransactionID`, `TransactionDT`, and `isFraud` are not model features. The transaction-only model also leaves out columns beginning with `id_`; the second model includes those identity columns when available.

Both feature sets use the same class-weighted logistic regression approach. Numeric values are median-imputed and scaled; categorical values are imputed and one-hot encoded. The holdout comparison reports average precision (PR-AUC) and ROC-AUC. It also calculates precision, fraud capture (recall), false-positive rate, legitimate approval rate, and the share of transactions flagged at thresholds of 1%, 2%, 5%, 10%, 20%, and 50% by default.

Thresholds are examples for exploration, not recommendations. They are not selected using a separate validation set. Results depend on the downloaded data, time split, preprocessing, and thresholds. Do not claim an improvement until you run the project and compare the results from your data.

## Set up and run

Python 3.10 or newer is recommended. Download the Kaggle files and place them in `data/raw/`, then run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m payment_fraud_pov.pipeline --data-dir data/raw --output-dir reports
```

The project does not download the data automatically. You can provide a different output folder or thresholds, for example:

```bash
python -m payment_fraud_pov.pipeline --data-dir data/raw --output-dir reports --thresholds 0.03 0.08 0.15
```

## Files produced

After a successful run, `reports/` contains:

| File | Contents |
| --- | --- |
| `model_comparison.csv` | Holdout metrics, row counts, fraud rate, feature counts, and identity availability. |
| `operating_points.csv` | Metrics at each model and score threshold. |
| `model_comparison.png` | A chart comparing holdout ROC-AUC and PR-AUC. |
| `operating_points.png` | A chart of fraud capture against false-positive rate, with thresholds labeled. |
| `run_metadata.json` | Dataset name, split information, time boundary, thresholds, and project status. |

The charts are generated from your run's measured results; they are not sample or fabricated results. The pipeline does not save per-transaction predictions.

## Other project materials

- `src/payment_fraud_pov/pipeline.py` contains the data preparation, time split, models, metrics, and chart generation.
- `src/payment_fraud_pov/api_contract.py` validates the shape of an illustrative identity-signal API response.
- `docs/api_contract.md` provides synthetic example request and response data plus integration checks. It is not a real API specification and does not contact a service.
- `docs/customer_pov.md` describes a possible evaluation workflow and its limitations.
- `docs/data_and_artifacts.md` provides additional detail on data provenance and output files.

## Important limitations

This is a portfolio demonstration, not a production fraud system or evidence of customer value. The benchmark may not represent any payment processor, and some fields may not be available at decision time. Before applying this kind of analysis in a real setting, use authorized representative data, confirm label quality and field availability, agree on fraud and review costs, and conduct appropriate privacy, security, fairness, and independent model reviews. Never use this demonstration to make real payment decisions.
