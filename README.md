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
- CaliforniaHousing. Missing values: 0 / 165120, (0.0%)
- santander-value-prediction-challenge. Missing values: 0 / 22254869, (0.0%)
- allstate-claims-severity. Missing values: 0 / 24481340, (0.0%)




| Dataset | Default data | Complete-case analysis | available case analysis — ACA |
| --- | --- | --- | --- | 
| sberbank-russian-housing-market | 2631539.24 (216386.74) | --- | --- |
| house-prices-advanced-regression-techniques | 27582.54 (5511.91) | ---  | --- |
| CaliforniaHousing | ---  | ---  | --- |
| santander-value-prediction-challenge | ---  | ---  | --- |
| allstate-claims-severity | ---  | ---  | --- |
