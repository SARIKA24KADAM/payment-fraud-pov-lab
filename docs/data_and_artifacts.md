# Data and generated artifacts

## Dataset provenance

The analysis is designed for the IEEE-CIS Fraud Detection competition dataset, distributed through [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data). It is not bundled here: access requires a Kaggle account and acceptance of the competition's terms. Download `train_transaction.csv` and `train_identity.csv` yourself and place them in `data/raw/`. The identity file is optional; without it, the identity-enriched model has no identity columns and should not be interpreted as an identity lift comparison.

Do not commit the downloaded competition files or any private/customer data. The `.gitignore` excludes files under `data/raw/` and generated files under `reports/` by default. This repository contains the code, analysis documentation, API example, and reproducibility instructions, but no benchmark records or run results because no dataset was present when this documentation was prepared.

## Expected source files

| File | Use | Required fields |
| --- | --- | --- |
| `data/raw/train_transaction.csv` | Transaction features, binary label, and relative event time | `TransactionID`, `TransactionDT`, `isFraud` |
| `data/raw/train_identity.csv` | Optional identity attributes joined one-to-one to transactions | `TransactionID` |

The pipeline holds out the latest 20% by `TransactionDT` boundary. It excludes `TransactionID`, `TransactionDT`, and `isFraud` from model features. The transaction-only model excludes columns beginning `id_`; the enriched model includes them when available.

## Generated files

Running the documented command writes these artifacts to `reports/`:

- `model_comparison.csv`: row counts, prevalence, feature counts, PR-AUC, ROC-AUC, and identity availability.
- `operating_points.csv`: precision, fraud capture, false-positive rate, legitimate approval rate, and flagged share per model and threshold.
- `model_comparison.png`: scatter plot comparing holdout ROC-AUC and average precision.
- `operating_points.png`: fraud capture versus false-positive rate, with score thresholds labeled.
- `run_metadata.json`: dataset identifier, split description and time boundary, thresholds, and non-production status.

CSV/JSON outputs and PNGs are reproducible from the same source files, dependency versions, code, and command-line thresholds. The generated PNGs are not checked in as static sample results; run the pipeline against your authorized local copy to generate them. The pipeline deliberately does not persist per-transaction predictions.

## Other project inputs

`docs/api_contract.md` contains illustrative synthetic request/response JSON for interface discussion; it is not dataset input and does not call an API. `docs/customer_pov.md` describes the proposed evaluation and interpretation. No customer data, credentials, or external identity API responses are included.
