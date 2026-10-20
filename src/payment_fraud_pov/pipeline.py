"""Train and evaluate a fraud scoring baseline on supported public datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "isFraud"
JOIN_KEY = "TransactionID"
TIME_KEY = "TransactionDT"
DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
DEFAULT_COSTS = {"missed_fraud": 250.0, "review": 1.0, "false_decline": 10.0}


def file_sha256(path: Path) -> str:
    """Hash a source file without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load either IEEE-CIS files or the Kaggle card_transdata benchmark."""
    transactions_path = data_dir / "train_transaction.csv"
    card_path = data_dir / "card_transdata.csv"
    if not transactions_path.exists():
        if not card_path.exists():
            raise FileNotFoundError(
                f"Missing {card_path}. Place the Kaggle card_transdata.csv file "
                "in the selected data directory."
            )
        frame = pd.read_csv(card_path)
        if "fraud" not in frame:
            raise ValueError("card_transdata.csv must contain a 'fraud' column.")
        frame = frame.rename(columns={"fraud": TARGET})
        frame.attrs["dataset"] = "Kaggle Credit Card Fraud Detection (card_transdata)"
        frame.attrs["split"] = (
            "stratified random 60/20/20 train/validation/test "
            "(random_state=42, validation_state=43)"
        )
        frame.attrs["source_files"] = [card_path]
        return frame

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
    transactions.attrs["dataset"] = "IEEE-CIS Fraud Detection (Kaggle)"
    transactions.attrs["split"] = (
        "earliest 60% train, next 20% validation, latest 20% test "
        "by TransactionDT boundaries"
    )
    transactions.attrs["source_files"] = [
        transactions_path,
        *([identity_path] if identity_path.exists() else []),
    ]
    return transactions


def chronological_split(frame: pd.DataFrame):
    """Create 60/20/20 chronological partitions without splitting tied times."""
    ordered = frame.sort_values(TIME_KEY, kind="mergesort").reset_index(drop=True)
    first_cut, second_cut = int(len(ordered) * 0.6), int(len(ordered) * 0.8)
    if first_cut < 1 or second_cut <= first_cut or second_cut >= len(ordered):
        raise ValueError("Dataset is too small for a temporal 60/20/20 split.")
    train_boundary = ordered.loc[first_cut, TIME_KEY]
    test_boundary = ordered.loc[second_cut, TIME_KEY]
    train = ordered.loc[ordered[TIME_KEY] < train_boundary].copy()
    validation = ordered.loc[
        (ordered[TIME_KEY] >= train_boundary)
        & (ordered[TIME_KEY] < test_boundary)
    ].copy()
    test = ordered.loc[ordered[TIME_KEY] >= test_boundary].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal boundaries produced an empty partition.")
    if not (train[TIME_KEY].max() < validation[TIME_KEY].min()
            <= validation[TIME_KEY].max() < test[TIME_KEY].min()):
        raise ValueError("Temporal partitions are not strictly ordered.")
    return train, validation, test


def stratified_split(frame: pd.DataFrame):
    """Create a reproducible stratified 60/20/20 random split."""
    try:
        train_validation, test = train_test_split(
            frame, test_size=0.2, random_state=42, stratify=frame[TARGET]
        )
        train, validation = train_test_split(
            train_validation, test_size=0.25, random_state=43,
            stratify=train_validation[TARGET],
        )
    except ValueError as exc:
        raise ValueError(
            "A stratified 60/20/20 split needs enough rows in both label classes."
        ) from exc
    return train, validation, test


def feature_columns(train: pd.DataFrame, include_identity: bool) -> list[str]:
    excluded = {TARGET, JOIN_KEY, TIME_KEY}
    columns = [column for column in train.columns if column not in excluded]
    if not include_identity:
        columns = [
            column for column in columns
            if not column.startswith("id_") and column != "_identity_available"
        ]
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
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: tuple[float, ...],
    model_name: str,
    costs: dict[str, float],
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
        total_cost = (
            fn * costs["missed_fraud"]
            + (tp + fp) * costs["review"]
            + fp * costs["false_decline"]
        )
        rows.append({
            "model": model_name,
            "threshold": threshold,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "fraud_capture_rate": tp / (tp + fn) if tp + fn else 0.0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "legitimate_approval_rate": tn / (fp + tn) if fp + tn else 0.0,
            "review_or_decline_share": float(predicted_fraud.mean()),
            "estimated_total_cost": total_cost,
            "estimated_cost_per_1000": total_cost / len(y_true) * 1000,
        })
    return rows


def run(
    data_dir: Path,
    output_dir: Path,
    thresholds: tuple[float, ...],
    costs: dict[str, float] | None = None,
) -> None:
    costs = dict(DEFAULT_COSTS if costs is None else costs)
    if not thresholds or any(
        not np.isfinite(value) or value < 0 or value > 1 for value in thresholds
    ):
        raise ValueError("At least one finite score threshold from 0 to 1 is required.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Candidate score thresholds must be unique.")
    if set(costs) != set(DEFAULT_COSTS) or any(
        not np.isfinite(value) or value < 0 for value in costs.values()
    ):
        raise ValueError("Cost assumptions must be finite, non-negative values.")
    frame = load_data(data_dir)
    dataset_name = frame.attrs.get("dataset", "unknown")
    split_description = frame.attrs.get("split", "")
    labels = pd.to_numeric(frame[TARGET], errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET!r} must contain only non-missing binary 0/1 labels.")
    frame[TARGET] = labels.astype(int)
    if frame.empty:
        raise ValueError("The selected dataset has no transaction rows.")
    if "TransactionDT" in frame:
        train, validation, test = chronological_split(frame)
    else:
        train, validation, test = stratified_split(frame)
    y_train = train[TARGET].astype(int).to_numpy()
    y_validation = validation[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()
    for partition_name, labels_in_partition in (
        ("training", y_train), ("validation", y_validation), ("test", y_test)
    ):
        if np.unique(labels_in_partition).size < 2:
            raise ValueError(
                f"The {partition_name} partition must contain fraud and non-fraud rows."
            )

    model_rows, operating_rows, cost_rows = [], [], []
    identity_features = [
        column for column in train.columns
        if column.startswith("id_") and not train[column].isna().all()
    ]
    if "TransactionDT" in frame and identity_features:
        model_specs = (("transaction_only", False),
                       ("transaction_plus_identity", True))
    elif "TransactionDT" in frame:
        model_specs = (("transaction_only", False),)
    else:
        model_specs = (("transaction_features", False),)
    for name, use_identity in model_specs:
        columns = feature_columns(train, include_identity=use_identity)
        if not columns:
            raise ValueError(f"No usable features found for {name}.")
        train_x = train[columns].replace([np.inf, -np.inf], np.nan)
        validation_x = validation[columns].replace([np.inf, -np.inf], np.nan)
        test_x = test[columns].replace([np.inf, -np.inf], np.nan)
        model = make_model(train_x)
        model.fit(train_x, y_train)
        validation_scores = model.predict_proba(validation_x)[:, 1]
        test_scores = model.predict_proba(test_x)[:, 1]
        validation_rows = score_operating_points(
            y_validation, validation_scores, thresholds, name, costs
        )
        selected = min(
            validation_rows,
            key=lambda row: (row["estimated_cost_per_1000"], row["threshold"]),
        )
        test_rows = score_operating_points(y_test, test_scores, thresholds, name, costs)
        for row in test_rows:
            row["selected_on_validation"] = row["threshold"] == selected["threshold"]
        operating_rows.extend(test_rows)
        validation_by_threshold = {
            row["threshold"]: row["estimated_cost_per_1000"]
            for row in validation_rows
        }
        for row in test_rows:
            cost_rows.append({
                "model": name,
                "threshold": row["threshold"],
                "validation_cost_per_1000": validation_by_threshold[row["threshold"]],
                "test_cost_per_1000": row["estimated_cost_per_1000"],
                "selected_on_validation": row["selected_on_validation"],
            })
        selected_test = next(
            row for row in test_rows if row["threshold"] == selected["threshold"]
        )
        no_review_validation_cost = (
            int(y_validation.sum()) * costs["missed_fraud"]
            / len(y_validation) * 1000
        )
        no_review_test_cost = (
            int(y_test.sum()) * costs["missed_fraud"] / len(y_test) * 1000
        )
        model_rows.append({
            "model": name,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "test_fraud_rate": float(y_test.mean()),
            "validation_average_precision_pr_auc": float(
                average_precision_score(y_validation, validation_scores)
            ),
            "validation_roc_auc": float(roc_auc_score(y_validation, validation_scores)),
            "average_precision_pr_auc": float(average_precision_score(y_test, test_scores)),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "feature_count": len(columns),
            "selected_threshold": selected["threshold"],
            "validation_cost_per_1000_at_selected_threshold": selected[
                "estimated_cost_per_1000"
            ],
            "test_cost_per_1000_at_selected_threshold": selected_test[
                "estimated_cost_per_1000"
            ],
            "validation_cost_per_1000_no_review": no_review_validation_cost,
            "test_cost_per_1000_no_review": no_review_test_cost,
            "test_precision_at_selected_threshold": selected_test["precision"],
            "test_fraud_capture_at_selected_threshold": selected_test[
                "fraud_capture_rate"
            ],
            "test_false_positive_rate_at_selected_threshold": selected_test[
                "false_positive_rate"
            ],
            "identity_available_test_share": (
                float(test["_identity_available"].mean()) if use_identity else None
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(output_dir / "operating_points.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output_dir / "cost_analysis.csv", index=False)
    write_plots(
        pd.DataFrame(model_rows), pd.DataFrame(operating_rows),
        pd.DataFrame(cost_rows), output_dir,
    )
    source_hashes = {
        source.name: file_sha256(source)
        for source in frame.attrs.get("source_files", [])
    }
    metadata = {
        "dataset": dataset_name,
        "source_files_sha256": source_hashes,
        "split": split_description,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_fraud_rate": float(y_validation.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "cost_assumptions": costs,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "train_time_max": float(train[TIME_KEY].max()) if TIME_KEY in train else None,
        "validation_time_min": float(validation[TIME_KEY].min()) if TIME_KEY in validation else None,
        "validation_time_max": float(validation[TIME_KEY].max()) if TIME_KEY in validation else None,
        "test_time_min": float(test[TIME_KEY].min()) if TIME_KEY in test else None,
        "thresholds": list(thresholds),
        "data_is_simulated": "card_transdata" in dataset_name.lower(),
        "customer_data_used": False,
        "production_ready": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def write_plots(
    model_results: pd.DataFrame,
    operating_results: pd.DataFrame,
    cost_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write PNG visualizations from this run's measured test results."""
    colors = {"transaction_only": "#2563eb",
              "transaction_plus_identity": "#e4572e",
              "transaction_features": "#2563eb"}
    display_names = {
        "transaction_only": "Transaction only",
        "transaction_plus_identity": "Transaction + identity",
        "transaction_features": "Transaction features",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(model_results) == 1:
        row = model_results.iloc[0]
        values = [row["roc_auc"], row["average_precision_pr_auc"]]
        bars = ax.bar(["ROC-AUC", "Average precision\n(PR-AUC)"], values,
                      color=colors.get(row["model"], "#2563eb"), width=0.55)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title("Test performance - transaction features")
    else:
        for _, row in model_results.iterrows():
            ax.scatter(row["roc_auc"], row["average_precision_pr_auc"], s=110,
                       color=colors.get(row["model"], "#666666"),
                       label=display_names.get(row["model"], row["model"]))
            ax.annotate(row["model"].replace("_", " "),
                        (row["roc_auc"], row["average_precision_pr_auc"]),
                        xytext=(7, 7), textcoords="offset points", fontsize=8)
        ax.set(xlabel="ROC-AUC", ylabel="Average precision (PR-AUC)",
               title="Test model comparison")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in operating_results.groupby("model", sort=False):
        group = group.sort_values("threshold")
        ax.plot(group["false_positive_rate"], group["fraud_capture_rate"],
                marker="o", color=colors.get(name, "#666666"),
                label=display_names.get(name, name))
        for _, row in group.iterrows():
            ax.annotate(f"{row['threshold']:.0%}",
                        (row["false_positive_rate"], row["fraud_capture_rate"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlabel="False-positive rate", ylabel="Fraud capture rate",
           title="Test operating points by score threshold")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "operating_points.png", dpi=160)
    plt.close(fig)

    row = model_results.iloc[0]
    fraud_count = round(row["test_rows"] * row["test_fraud_rate"])
    counts = [row["test_rows"] - fraud_count, fraud_count]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Legitimate", "Fraud"], counts,
                  color=["#2563eb", "#e4572e"], width=0.55)
    ax.bar_label(bars, labels=[f"{int(n):,}" for n in counts], padding=4)
    ax.set_ylabel("Transactions")
    ax.set_title("Test partition class balance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in cost_results.groupby("model", sort=False):
        group = group.sort_values("threshold")
        color = colors.get(name, "#2563eb")
        test_color = "#e4572e" if cost_results["model"].nunique() == 1 else color
        ax.plot(group["threshold"], group["validation_cost_per_1000"],
                marker="o", color=color,
                label=f"{display_names.get(name, name)}: validation")
        ax.plot(group["threshold"], group["test_cost_per_1000"],
                marker="s", linestyle="--", color=test_color,
                label=f"{display_names.get(name, name)}: test")
        selected = group.loc[group["selected_on_validation"]]
        if not selected.empty:
            ax.axvline(selected.iloc[0]["threshold"], color=color,
                       linestyle=":", alpha=0.8)
    for index, (_, row) in enumerate(model_results.iterrows()):
        ax.axhline(row["test_cost_per_1000_no_review"], color="#666666",
                   linestyle="-.", alpha=0.45,
                   label="Test: no-review baseline" if index == 0 else "_nolegend_")
    ax.set(xlabel="Fraud-score threshold", ylabel="Estimated cost units per 1,000 transactions",
           title="Validation threshold choice and test cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "cost_impact.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
        help="Candidate score thresholds; validation data selects the lowest-cost one.",
    )
    parser.add_argument("--missed-fraud-cost", type=float,
                        default=DEFAULT_COSTS["missed_fraud"],
                        help="Cost units for each missed fraudulent transaction.")
    parser.add_argument("--review-cost", type=float, default=DEFAULT_COSTS["review"],
                        help="Cost units for reviewing each flagged transaction.")
    parser.add_argument("--false-decline-cost", type=float,
                        default=DEFAULT_COSTS["false_decline"],
                        help="Additional cost units when a legitimate payment is flagged.")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value < 0 or value > 1
           for value in args.thresholds):
        parser.error("Thresholds must be finite values between 0 and 1.")
    if len(set(args.thresholds)) != len(args.thresholds):
        parser.error("Thresholds must not contain duplicates.")
    costs = {
        "missed_fraud": args.missed_fraud_cost,
        "review": args.review_cost,
        "false_decline": args.false_decline_cost,
    }
    if any(not np.isfinite(value) or value < 0 for value in costs.values()):
        parser.error("Cost assumptions must be finite, non-negative values.")
    run(args.data_dir, args.output_dir, tuple(args.thresholds), costs)


if __name__ == "__main__":
    main()
