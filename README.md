# Census Income Prediction using Machine Learning

> **Binary classification** of individual income (≤$50K vs >$50K) using the UCI Adult Census dataset — with a full KDD pipeline: data cleaning, EDA, preprocessing, hyperparameter tuning, and comparative model evaluation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Business Impact](#business-impact)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Visualisations](#visualisations)
- [Quickstart](#quickstart)
- [Authors](#authors)

---

## Overview

This project applies the **Knowledge Discovery in Databases (KDD)** process to the [UCI Adult Census Income dataset](https://archive.ics.uci.edu/dataset/20/census+income) to predict whether an individual earns more than **$50,000 per year**. The goal is to understand which socioeconomic and demographic factors are the strongest predictors of high income, and to identify the most accurate classification model from a shortlist of three candidates.

---

## Business Impact

Understanding income determinants has real-world value:

- **Policy-making:** Governments can use income-prediction models to target social support programmes more effectively.
- **Credit risk:** Financial institutions can supplement traditional credit scoring with demographic risk signals.
- **Labour-market research:** Researchers can quantify the contribution of education, occupation, and working hours to income inequality.

Achieving **~87% accuracy** with an interpretable, reproducible pipeline means this work can be directly used or extended for such applications.

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository — Adult](https://archive.ics.uci.edu/dataset/20/census+income) |
| Instances | 48,842 (47,621 after cleaning) |
| Features | 14 (6 numerical + 8 categorical) |
| Target | `income`: `<=50K` / `>50K` |
| Missing values | ~3.7% of rows (workclass, occupation, native-country) |

**Numerical features:** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

**Categorical features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`

---

## Repository Structure

```
Census-Income-Prediction-using-Machine-Learning/
│
├── Census_Income_Data_Mining_Project.ipynb   # Full interactive notebook
├── census_income_pipeline.py                 # Clean, modular Python script
├── requirements.txt                          # Reproducible dependency list
├── .gitignore
└── README.md
```

> **Note on data:** The dataset is fetched automatically at runtime via the `ucimlrepo` library — no manual download required.

---

## Pipeline

```
Raw Data  ──►  Data Cleaning  ──►  EDA  ──►  Preprocessing  ──►  Model Training  ──►  Evaluation
```

### 1 · Data Cleaning
- Strip inconsistent formatting from the target label (`<=50K.` → `<=50K`)
- Drop rows with missing values (1,221 rows removed, ~2.5%)

### 2 · Exploratory Data Analysis
- Descriptive statistics for all numerical features
- **Stacked bar chart:** income distribution by gender
- **Correlation heatmap:** relationships among numerical features
- **Age histogram with KDE:** age distribution coloured by income class

### 3 · Preprocessing
- `MinMaxScaler` applied to 6 numerical features
- `pd.get_dummies` (one-hot encoding, `drop_first=True`) for 8 categorical features
- `LabelEncoder` applied to binary target (`<=50K` → 0, `>50K` → 1)
- Stratified 70/30 train-test split

### 4 · Hyperparameter Sensitivity
Accuracy was measured across a grid of key hyperparameters to select optimal values without overfitting:

| Model | Parameter explored | Chosen value |
|---|---|---|
| Logistic Regression | Regularisation `C` ∈ {0.01, 0.1, 1, 10, 100} | `C = 100` |
| Random Forest | `n_estimators` ∈ {10 … 300}, `max_depth` ∈ {5 … None} | 200 trees, depth 20 |
| XGBoost | `learning_rate` ∈ {0.01 … 0.5}, `n_estimators` ∈ {50 … 200} | lr = 0.30, 100 trees |

### 5 · Evaluation Metrics
Accuracy, MSE, RMSE, R², Precision, Recall, F1 (all macro-averaged), and confusion matrices.

---

## Results

| Model | Accuracy | MSE | RMSE | R² | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.84790 | 0.15209 | 0.38999 | 0.18611 | 0.80996 | 0.76044 | 0.77978 |
| Random Forest | 0.85980 | 0.14019 | 0.37442 | 0.24978 | 0.83600 | 0.76741 | 0.79251 |
| **XGBoost** | **0.86946** | **0.13053** | **0.36130** | **0.30147** | **0.83916** | **0.79624** | **0.81407** |

**XGBoost achieves the best performance across all metrics**, delivering 86.95% accuracy and an F1 score of 0.814 — a +2.2 pp accuracy gain over the Logistic Regression baseline.

---

## Visualisations

| File | Description |
|---|---|
| `eda_income_by_gender.png` | Stacked bar — income class by sex |
| `eda_correlation_matrix.png` | Heatmap of numerical feature correlations |
| `eda_age_distribution.png` | Age histogram with KDE, coloured by income |
| `hp_lr_c.png` | LR accuracy vs regularisation C |
| `hp_rf_nestimators.png` | RF accuracy vs number of trees |
| `hp_rf_depth.png` | RF accuracy vs max depth |
| `hp_xgb_lr.png` | XGBoost accuracy vs learning rate |
| `cm_logistic_regression.png` | Confusion matrix — Logistic Regression |
| `cm_random_forest.png` | Confusion matrix — Random Forest |
| `cm_xgboost.png` | Confusion matrix — XGBoost |

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/emaadkalantarii/Census-Income-Prediction-using-Machine-Learning.git
cd Census-Income-Prediction-using-Machine-Learning

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Run the clean Python script (all outputs saved to current directory)
python census_income_pipeline.py

# 4b. Or open the interactive notebook
jupyter notebook Census_Income_Data_Mining_Project.ipynb
```

---

## Authors

**Emad Kalantari Khalilabad** — [GitHub](https://github.com/emaadkalantarii)  
**Yousef Rezaei Mirghaed**  

*Knowledge Discovery and Data Mining — Master's Programme*

---

## License

This project is released under the [MIT License](LICENSE).  
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income) and is publicly available for research use.
