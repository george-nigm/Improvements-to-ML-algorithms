# improvements-to-ml-algorithms

This is repository of project: "Methods of imputation, detecting anomalies for improving the quality of machine learning algorithms."

Check the current tasks on [tasks.md](https://github.com/georgii-nigm/Improvements-to-ML-algorithms/blob/master/tasks.md)

* * *

## Performance evaluating & description of datasets

The performance of the methods is considered by the effect on the final RMSE when solving the regression problem by LightGBM. The RMSE mean and standard deviation of 10-fold Cross-Validation are used.

Evaluation metrics were calculated on:

- sberbank-russian-housing-market. Missing value: 261026 / 8806119, (2.964%)
- house-prices-advanced-regression-techniques. Missing values: 6965 / 115340, (6.039%)

* * *

## Methods of imputation of data

1\. discard_data: Complete-case analysis — CCA (default data), available case analysis — ACA.

|                   **Dataset**                   | **Available case analysis — ACA (Default data)** |             **Complete-case analysis**             |
|:-------------------------------------------:|:--------------------------------------------:|:----------------------------------------------:|
|       sberbank-russian-housing-market       |           **2631539.242 (216386.749)**           | (rows: 30471 -> 6042) 3545089.937 (324188.444) |
| house-prices-advanced-regression-techniques |             **27582.549 (5511.916)**             |     (rows: 1460 -> 0) No data     |

* * *

2\. univariate_imputation: mean, median, mode (most\_frequent) for numeric features - for categorical: the most\_frequent value is always imputed. (What about Random Sampling Imputation?)

| **Dataset**                                     | **without imputation**       | **mean imputation**          | **median imputation**        | **mode   imputation**        |
|---------------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| sberbank-russian-housing-market             | 2631539.242 (216386.749) | **2628162.876 (210446.647)** | 2634163.850 (208134.141) | 2631626.241 (210974.957) |
| house-prices-advanced-regression-techniques | 27582.549 (5511.916)     | 27447.227 (5597.852)     | **27434.728 (5447.184)**     | 27594.386 (5419.882)     |

* * *

1.  multiple_imputation: Multiple Imputation (MI), Multiple Imputation by Chained Equations (MICE)
2.  Predictive/Statistical models that impute the missing data: Linear Regression, Random Forest, k-NN (k Nearest Neighbour), Maximum likelihood, Expectation-Maximization, Sensitivity analysis

## Methods of detecting anomalies:

1.  Elliptic Envelope,
2.  One-Class SVM,
3.  Isolation Forest,
4.  LocalOutlierFactor.
