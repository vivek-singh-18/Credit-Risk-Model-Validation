# Credit Risk Model Development & Validation
### Independent Model Validation Case Study | Model Risk Management (MRM)

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-red.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Validation](https://img.shields.io/badge/Model%20Risk-SR%2011--7%20Aligned-green.svg)

---

## 📌 Project Overview
This project looks at a simple credit risk modeling workflow using loan application data. The goal was to build a few candidate models, review how they rank default risk, and check whether the results and assumptions are reasonable across time.

The focus was not only on raw performance metrics, but also on discrimination, probability calibration, stability across out-of-time (OOT) samples, population stability (PSI), segment-level behavior, and the trade-off between model performance and interpretability.

> **Validation Disclaimer:** This is an academic/internship case study. It is meant for learning and does not represent formal regulatory certification or production approval.

---

## 📊 Dataset & Attribute Overview
The dataset contains **10,000 retail credit applications** spanning origination periods from 2019 to 2022.
* **Target (`default_flag`):** Binary indicator (1 = Default / Charged Off, 0 = Fully Paid). Portfolio default rate is **18.52%** (1,852 defaults vs. 8,148 non-defaults).
* **Predictors:** Application-level attributes including `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `revol_util`, `inq_last_6mths`, `delinq_2yrs`, `open_acc`, `total_acc`, `emp_length_years`, `age`, `home_ownership`, `loan_purpose`, `grade`, and origination `issue_date`.

---

## 📑 Validation Framework & Sampling Architecture
The workflow follows a structured sequence:
Business Problem -> Data Quality -> EDA -> Hypothesis Testing -> Sampling -> Modeling -> Discrimination & Calibration -> PSI & OOT Stability -> Recommendation

1. **Primary Development Partitions (In-Time 2019–2021 Data):**
   * **Training Partition (60%):** 4,960 rows. Used for model fitting and ColumnTransformer pipeline fitting.
   * **Validation Partition (20%):** 1,240 rows. Used for hyperparameter tuning and model selection.
   * **In-Time Test Partition (20%):** 1,241 rows. Used for in-time evaluation.
2. **Out-Of-Time (OOT 2022 Originations) Partition:**
   * **OOT Test Partition (2,559 rows):** Held out as a **strictly untouched temporal robustness check**. OOT data was never used for feature selection, pipeline fitting, hyperparameter tuning, or threshold selection.

---

## 📈 Empirical Model Performance Metrics (In-Time Test Partition)

| Model | ROC-AUC | Gini | KS Stat (%) | Precision | Recall | F1-Score | Brier Score | Modeling Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dummy Baseline** | 0.5000 | 0.0000 | 9.34% | 0.0000 | 0.0000 | 0.0000 | 0.1209 | Naive Baseline |
| **Single-Feature LR (`int_rate`)** | 0.6569 | 0.3139 | 23.52% | 0.0000 | 0.0000 | 0.0000 | 0.1164 | Simple Benchmark |
| **Logistic Regression (Benchmark)** | **0.7475** | **0.4949** | **40.43%** | **0.5000** | **0.0687** | **0.1208** | **0.1083** | **Primary Model Choice** |
| **Decision Tree (Depth=5)** | 0.6175 | 0.2350 | 19.50% | 0.1111 | 0.0076 | 0.0143 | 0.1237 | Constrained Tree |
| **Random Forest (Trees=150)** | 0.6991 | 0.3983 | 30.22% | 0.0000 | 0.0000 | 0.0000 | 0.1134 | Ensemble Challenger |
| **XGBoost (Boosting)** | **0.6899** | **0.3798** | **30.11%** | **0.4000** | **0.0305** | **0.0567** | **0.1138** | **Shadow Challenger** |

---

## 🔍 Model Risk Audit Summary

| Validation Area | Audit Finding & Evidence | Assigned Risk Level | Recommendation |
| :--- | :--- | :---: | :--- |
| **Data Quality & Imputation** | Missingness was below 5%; median and mode imputation were applied within the preprocessing pipeline without target leakage. | **Low Risk** | Keep the current pipeline structure. |
| **Data Leakage** | Features were captured at origination, and preprocessing was fit only on `X_train`. | **Low Risk** | No obvious leakage issue in this setup. |
| **Discrimination (KS / AUC)** | Logistic Regression achieved **AUC = 0.7475 / KS = 40.43%**; XGBoost achieved **AUC = 0.6899 / KS = 30.11%**. | **Low Risk** | These are strong results for this exercise. |
| **Probability Calibration** | Logistic Regression had a Brier Score of **0.1083**; the tree models showed some overconfidence in the highest risk bins. | **Low-Medium Risk** | Recalibration could be considered if probabilities are used in a downstream loss model. |
| **Overfitting Risk** | The train-to-test AUC gap was below 0.015 for the regularized candidate models. | **Low Risk** | Regularization appears to be working as intended. |
| **Population Stability (PSI)** | Key risk drivers (`int_rate`, `dti`, `annual_inc`) stayed below PSI = 0.05 across the 2022 OOT sample. | **Low Risk** | The population appears relatively stable. |
| **Explainability & Governance** | Logistic Regression is easier to interpret and explain than XGBoost. | **Medium Risk** *(for XGBoost)* | Use Logistic Regression as the main model and keep XGBoost as a challenger. |

---

## 💡 Model Selection & Final Recommendation

### Primary Model Selection: **Logistic Regression (Benchmark)**
> Logistic Regression was selected as the main model because it had the strongest overall performance in this exercise and is easier to interpret than the tree models. Its coefficient structure also makes it easier to explain how the score is being driven.

### Shadow Challenger Role: **XGBoost Classifier**
> XGBoost performed reasonably well and was kept as a challenger model. It can be useful for benchmarking and comparison, but the simpler logistic model is easier to explain and document.

---

## 🛠️ Reproduction & Execution Instructions

1. **Open the project folder in a terminal:**
   ```bash
   cd "path\to\project\folder"
   ```

2. **Create or activate the virtual environment and install dependencies:**
   ```bash
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Open and run the notebook:**
   ```bash
   jupyter notebook
   ```
   Then open `Credit_Risk_Model_Validation.ipynb` and run all cells from top to bottom.
