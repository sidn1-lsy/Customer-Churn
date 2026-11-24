# Customer Churn Analysis & Predictive Modeling

An end-to-end exploratory data analysis (EDA) and machine learning workflow in `analysis.ipynb` for predicting customer churn (`Churn` / `Churn_Binary`) using a 1,000-row subscription dataset.

## Dataset Overview

The dataset contains **1,000 rows** and **10 columns** (`CustomerID`, `Age`, `Gender`, `Tenure`, `MonthlyCharges`, `ContractType`, `InternetService`, `TotalCharges`, `TechSupport`, and `Churn`) with no duplicate rows or duplicate `CustomerID`s.

- **Target balance:** `Churn == 'Yes'` accounts for **883 rows (88.3%)** and `Churn == 'No'` accounts for **117 rows (11.7%)**.
- **Missing data:** `InternetService` has **297 missing values** (`Fiber Optic`: 395, `DSL`: 308); all other columns have 0 missing values.
- **Data characteristics & caveats:**
  - `TotalCharges` equals `Tenure * MonthlyCharges` across all 1,000 records.
  - Several subgroups in this dataset have a 100% churn rate (`Month-to-Month`: 511/511, `TechSupport == 'No'`: 494/494, and missing `InternetService`: 297/297), and features use telecom-style labels (`ContractType`, `InternetService`, `TechSupport`).
  - Extreme values include `Age` at `12` and `83`, and 20 records with `Tenure` above `72` months (up to `122`).

## Workflow & Modeling Summary

1. **Data Cleaning & Preprocessing:** Imputes missing `InternetService` values with `'None'` and maps `Churn` to `Churn_Binary` (`1` for `'Yes'`, `0` for `'No'`).
2. **Exploratory Data Analysis:** Generates visualizations comparing churn across `ContractType`, `MonthlyCharges` distributions by churn status, and `Tenure` vs. `MonthlyCharges` colored by churn.
3. **Train/Test Split & Pipelines:** Uses an `80/20` stratified split (`800` training rows, `200` test rows, `random_state=42`) with 8 input features (`Age`, `Gender`, `Tenure`, `MonthlyCharges`, `ContractType`, `InternetService`, `TotalCharges`, `TechSupport`). A `ColumnTransformer` applies `StandardScaler` to numeric features and `OneHotEncoder(drop='first', handle_unknown='ignore')` to categorical features inside scikit-learn pipelines with `class_weight='balanced'`.
4. **Model Evaluation (Test Set, n = 200):**
   - **Logistic Regression:** `0.91` accuracy, `0.9914` ROC-AUC (class `0` precision `0.56`, recall `1.00`, f1 `0.72`; class `1` precision `1.00`, recall `0.90`, f1 `0.95`).
   - **Random Forest:** `1.00` accuracy, `1.0000` ROC-AUC (precision, recall, and f1 of `1.00` for both classes).

> Because the dataset exhibits strong deterministic separation across features, these evaluation scores should be interpreted alongside the dataset's structure rather than assumed to generalize to noisy real-world data without further validation.

## Repository Contents

- `analysis.ipynb` — Main Jupyter notebook covering cleaning, EDA, pipelines, training, and evaluation.
- Churn CSV dataset (`customer_churn_data_2.csv` / `customer_churn_data.csv` — ensure the filename in the load cell matches the CSV in your working directory).
- Generated chart files (`churn_by_contract.png`, `monthly_charges_distribution.png`, `tenure_vs_charges.png`, `roc_curve_comparison.png`).
