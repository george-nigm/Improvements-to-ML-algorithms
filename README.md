# Improvements-to-ML-algorithms
This is repository of project: "Methods of detecting anomalies, imputation for improving the quality of machine learning algorithms."

Methods of detecting anomalies: 
1) Elliptic Envelope, One-Class SVM, Isolation Forest, LocalOutlierFactor.

Methods of imputation of data: 
1) Discard Data: Complete-case analysis — CCA, available case analysis — ACA, Dropping Variables 
2) Retain All Data: Mean, Median, Mode, midpoint, Random Sampling Imputation , Arbitrary Value Imputation, variable to capture NA  and assign sample with missings , variable to capture NA of each feature 
3) Multiple Imputation: Multiple Imputation (MI), Multiple Imputation by Chained Equations (MICE)
4) Predictive/Statistical models that impute the missing data: Linear Regression, Random Forest, k-NN (k Nearest Neighbour), Maximum likelihood, Expectation-Maximization, Sensitivity analysis

Evaluation metrics were calculated on:
- sberbank-russian-housing-market. Missing value: 261026 / 8806119, (2.964%)
- house-prices-advanced-regression-techniques. Missing values: 6965 / 115340, (6.039%)
- CaliforniaHousing



| Dataset | Default data | Complete-case analysis | available case analysis — ACA |
| --- | --- | --- | --- | 
| sberbank-russian-housing-market | 2631539.242 (216386.749) | (rows: 30471 -> 6042) 3545089.937 (324188.444) | --- |
| house-prices-advanced-regression-techniques | 27582.549 (5511.916) | (rows: 1460 -> 0)  | --- |
