"""Temporal fraud POV comparing transaction-only and identity-enriched signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "isFraud"
JOIN_KEY = "TransactionID"
TIME_KEY = "TransactionDT"
DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load competition training files and join identity attributes when present."""
    transactions_path = data_dir / "train_transaction.csv"
    if not transactions_path.exists():
        raise FileNotFoundError(
            f"Missing {transactions_path}. Download the Kaggle training files "
            "and place them in the selected data directory."
        )

    transactions = pd.read_csv(transactions_path)
    if TARGET not in transactions or TIME_KEY not in transactions:
        raise ValueError(f"Transaction file must contain {TARGET!r} and {TIME_KEY!r}.")

    identity_path = data_dir / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        if JOIN_KEY not in identity:
            raise ValueError(f"Identity file must contain {JOIN_KEY!r}.")
        if identity[JOIN_KEY].duplicated().any():
            raise ValueError("Identity file contains duplicate transaction keys.")
        transactions = transactions.merge(
            identity, on=JOIN_KEY, how="left", validate="one_to_one"
        )
    transactions["_identity_available"] = transactions.filter(
        regex=r"^id_"
    ).notna().any(axis=1)
    return transactions


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    """Split by transaction time; use the latest fraction as an untouched holdout."""
    ordered = frame.sort_values(TIME_KEY, kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    if cut < 1 or cut >= len(ordered):
        raise ValueError("Dataset is too small for the requested temporal split.")
    train, test = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    if train[TIME_KEY].max() >= test[TIME_KEY].min():
        raise ValueError("Temporal split is not strictly ordered.")
    return train, test


def feature_columns(train: pd.DataFrame, include_identity: bool) -> list[str]:
    excluded = {TARGET, JOIN_KEY, TIME_KEY}
    columns = [column for column in train.columns if column not in excluded]
    if not include_identity:
        columns = [
            column for column in columns
            if not column.startswith("id_") and column != "_identity_available"
        ]
    # Remove all-empty columns based on training data to keep imputation defined.
    return [column for column in columns if not train[column].isna().all()]


def make_model(train_x: pd.DataFrame) -> Pipeline:
    categorical = [
        column for column in train_x.columns
        if (pd.api.types.is_object_dtype(train_x[column])
            or pd.api.types.is_string_dtype(train_x[column])
            or isinstance(train_x[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(train_x[column]))
    ]
    numeric = [column for column in train_x.columns if column not in categorical]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler(with_mean=False)),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore", min_frequency=50, max_categories=50
            )),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))

    preprocess = ColumnTransformer(transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=500, solver="liblinear", random_state=42
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def score_operating_points(
    y_true: np.ndarray, scores: np.ndarray, thresholds: tuple[float, ...], model_name: str
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        predicted_fraud = scores >= threshold
        positives = y_true == 1
        negatives = ~positives
        tp = int(np.sum(predicted_fraud & positives))
        fp = int(np.sum(predicted_fraud & negatives))
        fn = int(np.sum(~predicted_fraud & positives))
        tn = int(np.sum(~predicted_fraud & negatives))
        rows.append({
            "model": model_name,
            "threshold": threshold,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "fraud_capture_rate": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "legitimate_approval_rate": tn / (fp + tn) if fp + tn else 0.0,
            "review_or_decline_share": float(predicted_fraud.mean()),
        })
    return rows


def run(data_dir: Path, output_dir: Path, thresholds: tuple[float, ...]) -> None:
    frame = load_data(data_dir)
    train, holdout = chronological_split(frame)
    y_train = train[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()
    if np.unique(y_train).size < 2 or np.unique(y_holdout).size < 2:
        raise ValueError("Both temporal partitions must contain fraud and non-fraud cases.")

    model_rows, operating_rows = [], []
    for name, use_identity in (
        ("transaction_only", False),
        ("transaction_plus_identity", True),
    ):
        columns = feature_columns(train, include_identity=use_identity)
        if not columns:
            raise ValueError(f"No usable features found for {name}.")
        train_x = train[columns].replace([np.inf, -np.inf], np.nan)
        holdout_x = holdout[columns].replace([np.inf, -np.inf], np.nan)
        model = make_model(train_x)
        model.fit(train_x, y_train)
        scores = model.predict_proba(holdout_x)[:, 1]
        model_rows.append({
            "model": name,
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "holdout_fraud_rate": float(y_holdout.mean()),
            "average_precision_pr_auc": float(average_precision_score(y_holdout, scores)),
            "roc_auc": float(roc_auc_score(y_holdout, scores)),
            "feature_count": len(columns),
            "identity_available_holdout_share": (
                float(holdout["_identity_available"].mean()) if use_identity else None
            ),
        })
        operating_rows.extend(
            score_operating_points(y_holdout, scores, thresholds, name)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(output_dir / "operating_points.csv", index=False)
    metadata = {
        "dataset": "IEEE-CIS Fraud Detection (Kaggle)",
        "split": "earliest 80% train; latest 20% holdout by TransactionDT",
        "train_time_max": float(train[TIME_KEY].max()),
        "holdout_time_min": float(holdout[TIME_KEY].min()),
        "thresholds": list(thresholds),
        "synthetic_or_customer_data": False,
        "production_ready": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        help="Score thresholds for the operating-point comparison.",
    )
    args = parser.parse_args()
    if any(value < 0 or value > 1 for value in args.thresholds):
        parser.error("Thresholds must be between 0 and 1.")
    run(args.data_dir, args.output_dir, tuple(args.thresholds))


if __name__ == "__main__":
    main()
