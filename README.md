# Payment Fraud Proof-of-Value Lab

This project uses transaction behavior data to show how a fraud scoring model performs at different review thresholds. It trains a class-weighted logistic-regression model and reports how many fraudulent and legitimate transactions would be flagged at each threshold.

## Dataset used

The current run uses Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/dhanushnarayanan/credit-card-fraud), file `card_transdata.csv`. Kaggle credentials were not available in this workspace, so the file was downloaded from this [public Hugging Face mirror](https://huggingface.co/datasets/arjund1999/CCFraud/blob/main/card_transdata.csv). The downloaded file is about 73 MB and lives in `data/raw/`. It is ignored by Git and is not uploaded to the project repository. See the Kaggle page for dataset terms and attribution.

The file contains seven feature columns and one target:

| Column | Use |
| --- | --- |
| `distance_from_home` | Model feature |
| `distance_from_last_transaction` | Model feature |
| `ratio_to_median_purchase_price` | Model feature |
| `repeat_retailer` | Model feature |
| `used_chip` | Model feature |
| `used_pin_number` | Model feature |
| `online_order` | Model feature |
| `fraud` | Fraud label (the pipeline renames this internally to `isFraud`) |

This dataset has no timestamp or separate identity table. The pipeline therefore uses a stratified random 80/20 train/holdout split with `random_state=42` and trains one transaction-feature model. The results are not a future-in-time evaluation and do not measure identity-signal lift. The pipeline also continues to support IEEE-CIS files if they are placed in `data/raw/` instead.

## Run the project

Python 3.10 or newer is recommended. The dataset is already in `data/raw/card_transdata.csv` in this workspace. Set up the environment and run:

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

Thresholds default to 1%, 2%, 5%, 10%, 20%, and 50%. You can supply others:

```bash
python -m payment_fraud_pov.pipeline --data-dir data/raw --output-dir reports --thresholds 0.03 0.08 0.15
```

## Generated files

The run writes these files to `reports/`. They are kept in the repository so the charts and summary results are visible without rerunning the analysis:

| File | What it contains |
| --- | --- |
| `model_comparison.csv` | Holdout row counts, fraud rate, feature count, average precision (PR-AUC), and ROC-AUC. |
| `operating_points.csv` | Precision, fraud capture, false-positive rate, legitimate approval rate, and flagged share at each threshold. |
| `model_comparison.png` | A chart showing holdout ROC-AUC and PR-AUC. |
| `operating_points.png` | A chart of fraud capture versus false-positive rate, with thresholds labeled. |
| `run_metadata.json` | Dataset name, split method, thresholds, and non-production status. |

The PNGs are generated from the actual holdout results. Thresholds are exploratory examples, not recommendations; there is no separate validation set used to choose them. The pipeline does not save per-transaction predictions.

## Other project materials

- `src/payment_fraud_pov/pipeline.py` contains data loading, splitting, model training, metrics, and chart generation.
- `src/payment_fraud_pov/api_contract.py` validates an illustrative identity-signal API response. It is not used by this dataset or model.
- `docs/api_contract.md` gives synthetic example request and response data for interface discussion.
- `docs/customer_pov.md` describes a possible evaluation workflow and its limitations.
- `docs/data_and_artifacts.md` records data provenance and generated file details.

This is a portfolio demonstration, not a production fraud system or evidence of customer value. The dataset is simulated and may not represent any payment processor or customer. Never use this demonstration to make actual payment decisions.
