"""
Census Income Prediction Pipeline
==================================
Self-contained script that reproduces every step in the companion notebook
(Census_Income_Prediction.ipynb) end-to-end.

Sections
--------
  0. Imports & constants
  1. Data loading
  2. Data exploration
  3. Data cleaning
  4. Exploratory data analysis (EDA)
  5. Preprocessing
  6. Train / test split
  7. Model training & hyperparameter sensitivity
     7.1  Logistic Regression
     7.2  Random Forest
     7.3  XGBoost
  8. Model comparison

Run
---
    python census_income_pipeline.py

Authors : Emad Kalantari Khalilabad, Yousef Rezaei Mirghaed
Course  : Knowledge Discovery and Data Mining
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0. Imports & constants
# ══════════════════════════════════════════════════════════════════════════════

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

# ── Global settings ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.30

NUMERICAL   = ["age", "fnlwgt", "education-num",
               "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status",
               "occupation", "relationship", "race", "sex", "native-country"]

plt.rcParams.update({"figure.dpi": 120,
                     "axes.spines.top": False,
                     "axes.spines.right": False})


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Fetch the UCI Census Income dataset and return a combined DataFrame."""
    print("Fetching dataset from UCI ML Repository …")
    census_income = fetch_ucirepo(id=20)
    df = pd.concat(
        [census_income.data.features, census_income.data.targets],
        axis=1,
    )
    print(f"  Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target classes: {df['income'].unique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data exploration
# ══════════════════════════════════════════════════════════════════════════════

def explore_data(df: pd.DataFrame) -> None:
    """Print column types, missing-value counts, and class distribution."""
    print("\n── Column types ──────────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing values per column ─────────────────────────────────────")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(report[report["Missing Count"] > 0].to_string())

    print("\n── Target class distribution (raw) ──────────────────────────────")
    print(df["income"].value_counts(dropna=False).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 3. Data cleaning
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalise target labels  (strip whitespace and trailing '.')
    - Drop rows that contain any NaN
    """
    df = df.copy()

    df["income"] = df["income"].str.strip().str.rstrip(".")
    print(f"\n  Target labels after normalisation: {df['income'].unique()}")

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"  Rows dropped (NaN): {n_dropped:,}  ({n_dropped/n_before*100:.1f}%)")
    print(f"  Rows remaining    : {len(df):,}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Exploratory data analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame) -> None:
    """Print descriptive stats and save three EDA plots."""
    print("\n── Numerical feature statistics ──────────────────────────────────")
    print(df[NUMERICAL].describe().round(2).to_string())

    _plot_income_by_gender(df)
    _plot_correlation_matrix(df)
    _plot_age_distribution(df)
    print("  EDA plots saved.")


def _plot_income_by_gender(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    contingency = pd.crosstab(df["sex"], df["income"])
    contingency.plot(kind="bar", stacked=True, ax=ax,
                     color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.6)
    ax.set_title("Annual Income Distribution by Gender",
                 fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Gender", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.legend(title="Income", fontsize=10)
    plt.tight_layout()
    plt.savefig("eda_income_by_gender.png", dpi=150)
    plt.close()


def _plot_correlation_matrix(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[NUMERICAL].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix — Numerical Features",
                 fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eda_correlation_matrix.png", dpi=150)
    plt.close()


def _plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="age", hue="income", kde=True,
                 palette={"<=50K": "#4C72B0", ">50K": "#DD8452"},
                 bins=35, alpha=0.7, ax=ax)
    ax.set_title("Age Distribution by Income Class",
                 fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("eda_age_distribution.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    1. MinMaxScaler on numerical features
    2. One-hot encoding on categorical features  (drop_first=True)
    3. LabelEncoder on binary target

    Returns X (pd.DataFrame), Y (np.ndarray)
    """
    df_proc = df.copy()

    scaler = MinMaxScaler()
    df_proc[NUMERICAL] = scaler.fit_transform(df_proc[NUMERICAL])

    df_proc = pd.get_dummies(df_proc, columns=CATEGORICAL, drop_first=True)

    le = LabelEncoder()
    y_raw = df_proc.pop("income")
    Y = le.fit_transform(y_raw)        # <=50K → 0,  >50K → 1
    X = df_proc

    print(f"\n  Feature matrix shape : {X.shape}")
    print(f"  Class distribution   : 0 (≤50K) = {(Y==0).sum():,}  "
          f"|  1 (>50K) = {(Y==1).sum():,}")
    print(f"  Class imbalance ratio: {(Y==0).sum()/(Y==1).sum():.2f}:1")

    return X, Y


# ══════════════════════════════════════════════════════════════════════════════
# 6. Train / test split
# ══════════════════════════════════════════════════════════════════════════════

def split_data(X, Y):
    """Stratified 70 / 30 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y,
    )
    print(f"\n  Training set : {len(X_train):,} samples")
    print(f"  Test set     : {len(X_test):,} samples")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 7. Shared evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model_name: str, y_true, y_pred) -> dict:
    """Print all metrics and save the confusion-matrix plot."""
    acc  = accuracy_score(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {acc:.5f}")
    print(f"  MSE       : {mse:.5f}")
    print(f"  RMSE      : {rmse:.5f}")
    print(f"  R² Score  : {r2:.5f}")
    print(f"  Precision : {prec:.5f}")
    print(f"  Recall    : {rec:.5f}")
    print(f"  F1 Score  : {f1:.5f}")

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["≤50K", ">50K"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=12, pad=10, fontweight="bold")
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(f"cm_{safe}.png", dpi=150)
    plt.close()

    return dict(Model=model_name,
                Accuracy=round(acc, 5), MSE=round(mse, 5),
                RMSE=round(rmse, 5),    R2=round(r2, 5),
                Precision=round(prec, 5), Recall=round(rec, 5),
                F1=round(f1, 5))


# ══════════════════════════════════════════════════════════════════════════════
# 7.1 Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

def run_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    """Sweep regularisation C, plot sensitivity, train final model."""
    print("\n══ 7.1  Logistic Regression ══════════════════════════════════════")

    # ── Hyperparameter sensitivity ─────────────────────────────────────────────
    C_values     = [0.01, 0.1, 1, 10, 100]
    lr_accuracies = []

    for C in C_values:
        lr = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_train, y_train)
        lr_accuracies.append(accuracy_score(y_test, lr.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(C_values, lr_accuracies, marker="o", linewidth=2, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_title("Logistic Regression — Accuracy vs Regularisation Parameter C",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("C (log scale)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    for C, acc in zip(C_values, lr_accuracies):
        ax.annotate(f"{acc:.4f}", (C, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("hp_lr_c.png", dpi=150)
    plt.close()

    best_C = C_values[lr_accuracies.index(max(lr_accuracies))]
    print(f"  Best C = {best_C}  →  Accuracy = {max(lr_accuracies):.5f}")

    # ── Final model ────────────────────────────────────────────────────────────
    lr_model = LogisticRegression(C=100, max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    return evaluate("Logistic Regression", y_test, lr_model.predict(X_test))


# ══════════════════════════════════════════════════════════════════════════════
# 7.2 Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def run_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """Sweep n_estimators and max_depth, plot sensitivity, train final model."""
    print("\n══ 7.2  Random Forest ════════════════════════════════════════════")

    # ── n_estimators sweep ────────────────────────────────────────────────────
    n_est_values = [10, 50, 100, 200, 300]
    rf_acc_est   = []

    for n in n_est_values:
        rf = RandomForestClassifier(n_estimators=n, max_depth=20,
                                    random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        rf_acc_est.append(accuracy_score(y_test, rf.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_est_values, rf_acc_est, marker="o", linewidth=2, color="#55A868")
    ax.set_title("Random Forest — Accuracy vs Number of Estimators",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("n_estimators", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    for n, acc in zip(n_est_values, rf_acc_est):
        ax.annotate(f"{acc:.4f}", (n, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("hp_rf_nestimators.png", dpi=150)
    plt.close()

    # ── max_depth sweep ───────────────────────────────────────────────────────
    depth_values = [5, 10, 15, 20, 30, None]
    rf_acc_depth = []

    for d in depth_values:
        rf = RandomForestClassifier(n_estimators=200, max_depth=d,
                                    random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        rf_acc_depth.append(accuracy_score(y_test, rf.predict(X_test)))

    labels = [str(d) if d is not None else "None" for d in depth_values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(labels)), rf_acc_depth, marker="o",
            linewidth=2, color="#55A868")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Random Forest — Accuracy vs Max Depth",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("max_depth", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    for i, acc in enumerate(rf_acc_depth):
        ax.annotate(f"{acc:.4f}", (i, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("hp_rf_depth.png", dpi=150)
    plt.close()

    # ── Final model ────────────────────────────────────────────────────────────
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                      random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    return evaluate("Random Forest", y_test, rf_model.predict(X_test))


# ══════════════════════════════════════════════════════════════════════════════
# 7.3 XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def run_xgboost(X_train, X_test, y_train, y_test) -> dict:
    """Sweep learning_rate and n_estimators, plot sensitivity, train final model."""
    print("\n══ 7.3  XGBoost ══════════════════════════════════════════════════")

    # ── Learning rate sweep ───────────────────────────────────────────────────
    lr_values   = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    xgb_acc_lr  = []

    for lr in lr_values:
        xgb = XGBClassifier(learning_rate=lr, n_estimators=100,
                            eval_metric="logloss", random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        xgb_acc_lr.append(accuracy_score(y_test, xgb.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lr_values, xgb_acc_lr, marker="o", linewidth=2, color="#C44E52")
    ax.set_title("XGBoost — Accuracy vs Learning Rate",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("Learning Rate", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    for lr, acc in zip(lr_values, xgb_acc_lr):
        ax.annotate(f"{acc:.4f}", (lr, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("hp_xgb_lr.png", dpi=150)
    plt.close()

    # ── n_estimators sweep ────────────────────────────────────────────────────
    tree_values    = [50, 100, 200, 300, 400]
    xgb_acc_trees  = []

    for n in tree_values:
        xgb = XGBClassifier(learning_rate=0.30, n_estimators=n,
                            eval_metric="logloss", random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        xgb_acc_trees.append(accuracy_score(y_test, xgb.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tree_values, xgb_acc_trees, marker="o", linewidth=2, color="#C44E52")
    ax.set_title("XGBoost — Accuracy vs Number of Trees",
                 fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("n_estimators", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    for n, acc in zip(tree_values, xgb_acc_trees):
        ax.annotate(f"{acc:.4f}", (n, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("hp_xgb_nestimators.png", dpi=150)
    plt.close()

    # ── Final model ────────────────────────────────────────────────────────────
    xgb_model = XGBClassifier(learning_rate=0.30, n_estimators=100,
                               eval_metric="logloss", random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    return evaluate("XGBoost", y_test, xgb_model.predict(X_test))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Model comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(results: list[dict]) -> None:
    """Print a summary table and save a grouped bar chart."""
    df_res = pd.DataFrame(results).set_index("Model")

    print("\n" + "═" * 75)
    print("  MODEL COMPARISON — TEST SET")
    print("═" * 75)
    print(df_res.to_string())
    print("═" * 75)

    best = df_res["Accuracy"].idxmax()
    print(f"\n  ✔  Best model : {best}  "
          f"(Accuracy {df_res.loc[best,'Accuracy']:.5f}  |  "
          f"F1 {df_res.loc[best,'F1']:.5f})\n")

    # ── Grouped bar chart ──────────────────────────────────────────────────────
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x       = np.arange(len(metrics))
    width   = 0.25
    colors  = ["#4C72B0", "#55A868", "#C44E52"]
    names   = df_res.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [df_res.loc[name, m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0.70, 0.95)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison",
                 fontsize=14, pad=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()
    print("  Comparison chart saved to model_comparison.png")

    df_res.to_csv("model_results.csv")
    print("  Results saved to model_results.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 55)
    print("  Census Income Prediction — Full Pipeline")
    print("═" * 55)

    df_raw   = load_data()
    explore_data(df_raw)

    df_clean = clean_data(df_raw)

    print("\n══ 4. EDA ════════════════════════════════════════════════════════")
    run_eda(df_clean)

    print("\n══ 5. Preprocessing ══════════════════════════════════════════════")
    X, Y = preprocess(df_clean)

    print("\n══ 6. Train / Test Split ════════════════════════════════════════")
    X_train, X_test, y_train, y_test = split_data(X, Y)

    print("\n══ 7. Model Training & Evaluation ═══════════════════════════════")
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test))
    results.append(run_random_forest(X_train, X_test, y_train, y_test))
    results.append(run_xgboost(X_train, X_test, y_train, y_test))

    print("\n══ 8. Model Comparison ═══════════════════════════════════════════")
    compare_models(results)


if __name__ == "__main__":
    main()
