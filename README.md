# Payment Fraud Proof-of-Value Lab

This project uses transaction behavior data to show how a fraud scoring model performs at different review thresholds. It trains a class-weighted logistic-regression model and reports how many fraudulent and legitimate transactions would be flagged at each threshold.

## Dataset used

The current run uses Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/dhanushnarayanan/credit-card-fraud), file `card_transdata.csv`. Kaggle credentials were not available in this workspace, so the file was downloaded from a [pinned public Hugging Face mirror revision](https://huggingface.co/datasets/arjund1999/CCFraud/blob/4beebe528117ddd470bd1b3a7286f91cd305ccc2/card_transdata.csv). It is about 73 MB. The raw file is ignored by Git and is not uploaded to the project repository. See the Kaggle page for dataset terms and attribution.

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

This dataset has no timestamp or separate identity table. The pipeline uses a stratified random 60/20/20 split for training, validation, and test with fixed random seeds, then trains one transaction-feature model. The validation set selects a threshold; the test set provides the final evaluation. The results are not a future-in-time evaluation and do not measure identity-signal lift. The pipeline also supports IEEE-CIS files if those are placed in `data/raw/` instead; when both formats are present, it uses the IEEE-CIS transaction file and creates chronological 60/20/20 partitions.

For this CSV, the seven behavior fields are median-imputed if needed, scaled, and passed to a class-weighted logistic-regression model. For IEEE-CIS input, categorical fields are one-hot encoded and the transaction-only and identity-enriched models are compared only when usable identity columns are present.

## Run the project

Python 3.10 or newer is recommended. If you cloned the repository and do not have the CSV, download it from the public mirror first:

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/arjund1999/CCFraud/resolve/4beebe528117ddd470bd1b3a7286f91cd305ccc2/card_transdata.csv' -OutFile 'data/raw/card_transdata.csv'
```

```bash
# macOS/Linux
mkdir -p data/raw
curl -L --fail -o data/raw/card_transdata.csv 'https://huggingface.co/datasets/arjund1999/CCFraud/resolve/4beebe528117ddd470bd1b3a7286f91cd305ccc2/card_transdata.csv'
```

Then set up the environment and run:

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

The candidate score thresholds default to 1%, 2%, 5%, 10%, 20%, and 50%. The pipeline selects the candidate with the lowest validation cost, then reports its cost on the untouched test split. Default illustrative costs are 250 units per missed fraud, 1 unit per reviewed transaction, and 10 additional units per legitimate transaction flagged. Cost is calculated as missed-fraud loss plus review cost for every flagged transaction plus false-decline cost for flagged legitimate transactions. These generic units are examples, not business estimates. Set assumptions and candidates explicitly if you have better estimates:

```bash
python -m payment_fraud_pov.pipeline --data-dir data/raw --output-dir reports --thresholds 0.03 0.08 0.15 --missed-fraud-cost 300 --review-cost 2 --false-decline-cost 12
```

## Generated files

The run writes these files to `reports/`. They are kept in the repository so the charts and summary results are visible without rerunning the analysis:

| File | What it contains |
| --- | --- |
| `model_comparison.csv` | Validation and test ranking metrics, selected threshold, cost estimates, and row counts. |
| `operating_points.csv` | Test precision, fraud capture, false-positive rate, approval rate, and cost at each threshold. Marks the validation-selected threshold. |
| `cost_analysis.csv` | Validation and test estimated cost per 1,000 transactions for each candidate threshold. |
| `model_comparison.png` | A chart showing test ROC-AUC and PR-AUC. |
| `operating_points.png` | A chart of test fraud capture versus false-positive rate, with thresholds labeled. |
| `cost_impact.png` | Validation and test cost curves, the selected threshold, and a no-review baseline. |
| `class_balance.png` | Counts of legitimate and fraud rows in the test partition. |
| `run_metadata.json` | Dataset name, input file hashes, split details, cost assumptions, package versions, and non-production status. |

In the checked-in run, validation selected a score threshold of 0.50. On the untouched test set, PR-AUC was 0.7575, ROC-AUC was 0.9795, fraud capture at that threshold was 94.8%, and estimated cost was 1,889.85 generic units per 1,000 transactions. The no-review baseline was 21,851.25 units under the same assumptions. These cost figures are illustrative and are not currency or expected customer savings. The model is fit on training data only; its class-weighted scores are not calibrated probabilities. The simulated benchmark and random split do not establish real-world performance. Results can change if you use different cost assumptions or candidate thresholds. The pipeline does not save per-transaction predictions.

## Other project materials

- `src/payment_fraud_pov/pipeline.py` contains data loading, splitting, model training, metrics, and chart generation.
- `src/payment_fraud_pov/api_contract.py` validates an illustrative identity-signal API response. It is not used by this dataset or model.
- `docs/api_contract.md` gives synthetic example request and response data for interface discussion.
- `docs/customer_pov.md` describes a possible evaluation workflow and its limitations.
- `docs/data_and_artifacts.md` records data provenance and generated file details.

This is a portfolio demonstration, not a production fraud system or evidence of customer value. The dataset is simulated and may not represent any payment processor or customer. Never use this demonstration to make actual payment decisions.
