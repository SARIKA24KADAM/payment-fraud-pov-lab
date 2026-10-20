# Data and generated artifacts

## Dataset in this run

This run uses Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/dhanushnarayanan/credit-card-fraud), specifically `card_transdata.csv`. Because Kaggle credentials were not configured in this workspace, the file was obtained from a [public Hugging Face mirror at a pinned revision](https://huggingface.co/datasets/arjund1999/CCFraud/blob/4beebe528117ddd470bd1b3a7286f91cd305ccc2/card_transdata.csv). The file is 76,277,977 bytes (about 73 MB), with SHA-256 `7013c329bae9ef0ef32d65dbeb095694f0c7cd6c00ff74b2d0087fa1c67b8717`. Refer to the Kaggle page for the current dataset terms and attribution requirements.

The CSV columns are `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`, `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order`, and `fraud`. The seven behavior columns are model features. `fraud` is the binary target; the pipeline renames it to `isFraud` internally.

There is no event timestamp or separate identity table in this source. The pipeline uses a stratified random 80/20 train/holdout split with `random_state=42` and trains one transaction-feature model. The checked-in run has 800,000 training rows, 200,000 holdout rows, 8.7405% holdout fraud prevalence, 0.7573 average precision, and 0.9795 ROC-AUC. These results are not a temporal evaluation and do not establish the value of identity signals.

The downloaded file is stored at `data/raw/card_transdata.csv`. Raw data is ignored by Git and has not been uploaded to the repository. This avoids redistributing it without confirming the Kaggle dataset's terms. Generated CSV/JSON summaries and PNG charts in `reports/` are tracked with the project.

## Generated artifacts

Running the command in the main README writes these files to `reports/`:

- `model_comparison.csv`: holdout row counts, fraud rate, feature count, average precision (PR-AUC), and ROC-AUC.
- `operating_points.csv`: precision, fraud capture, false-positive rate, legitimate approval rate, and flagged share at each threshold.
- `model_comparison.png`: holdout ROC-AUC and PR-AUC visualization.
- `operating_points.png`: fraud capture against false-positive rate, with thresholds labeled.
- `class_balance.png`: legitimate and fraud counts in the holdout set.
- `run_metadata.json`: dataset, source file SHA-256, split method, sample counts, holdout fraud rate, configured thresholds, software versions, and explicit simulated-data/customer-data/production flags.

The PNGs are generated from the downloaded dataset's measured holdout results. The pipeline does not save per-transaction predictions.

## Other project material

`docs/api_contract.md` contains synthetic request and response examples for discussing a hypothetical identity API. Those examples are not part of the dataset or model inputs. `docs/customer_pov.md` outlines an evaluation workflow and its limitations.
