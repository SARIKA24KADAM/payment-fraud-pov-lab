# Data and generated artifacts

## Dataset in this run

This run uses Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/dhanushnarayanan/credit-card-fraud), specifically `card_transdata.csv`. Because Kaggle credentials were not configured in this workspace, the file was obtained from a [public Hugging Face mirror at a pinned revision](https://huggingface.co/datasets/arjund1999/CCFraud/blob/4beebe528117ddd470bd1b3a7286f91cd305ccc2/card_transdata.csv). The file is 76,277,977 bytes (about 73 MB), with SHA-256 `7013c329bae9ef0ef32d65dbeb095694f0c7cd6c00ff74b2d0087fa1c67b8717`. Refer to the Kaggle page for the current dataset terms and attribution requirements.

The CSV columns are `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`, `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order`, and `fraud`. The seven behavior columns are model features. `fraud` is the binary target; the pipeline renames it to `isFraud` internally.

There is no event timestamp or separate identity table in this source. The pipeline uses a reproducible stratified random 60/20/20 training/validation/test split. The model is fit on training rows. Validation data chooses a score threshold from the candidate list by minimizing the estimated cost; the test partition is used for final evaluation. These results are not a temporal evaluation and do not establish the value of identity signals.

With the checked-in default configuration, validation selected threshold `0.50`. Test PR-AUC was `0.7575`, test ROC-AUC was `0.9795`, test fraud capture at the selected threshold was `94.8%`, and estimated cost was `1,889.85` generic units per 1,000 transactions. The test no-review baseline was `21,851.25` units. These depend on the illustrative cost assumptions below and should not be read as currency or expected savings.

The configurable cost formula is: missed-fraud count × missed-fraud cost + flagged count × review cost + false-positive count × additional false-decline cost. Defaults are 250, 1, and 10 generic cost units respectively. They are examples only and should be replaced with customer-approved estimates for any real evaluation.

The downloaded file is stored at `data/raw/card_transdata.csv`. Raw data is ignored by Git and has not been uploaded to the repository. This avoids redistributing it without confirming the Kaggle dataset's terms. Generated CSV/JSON summaries and PNG charts in `reports/` are tracked with the project.

## Generated artifacts

Running the command in the main README writes these files to `reports/`:

- `model_comparison.csv`: validation and test ranking metrics, row counts, selected threshold, and cost estimates.
- `operating_points.csv`: test precision, fraud capture, false-positive rate, legitimate approval rate, and estimated cost at each candidate threshold; identifies the validation-selected threshold.
- `cost_analysis.csv`: validation and test costs per 1,000 transactions across candidate thresholds.
- `model_comparison.png`: test ROC-AUC and PR-AUC visualization.
- `operating_points.png`: test fraud capture against false-positive rate, with thresholds labeled.
- `cost_impact.png`: validation/test cost curves, chosen threshold, and no-review test baseline.
- `class_balance.png`: legitimate and fraud counts in the test set.
- `run_metadata.json`: dataset, source file SHA-256, split method, sample counts and fraud rates, cost assumptions, software versions, and explicit simulated-data/customer-data/production flags.

The charts are generated from the downloaded dataset's measured validation and test results. The pipeline does not save per-transaction predictions.

## Other project material

`docs/api_contract.md` contains synthetic request and response examples for discussing a hypothetical identity API. Those examples are not part of the dataset or model inputs. `docs/customer_pov.md` outlines an evaluation workflow and its limitations.
