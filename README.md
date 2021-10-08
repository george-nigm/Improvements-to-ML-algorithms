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

## Encoding Categorical Data
Due to the inability of some imputation methods and all anomaly detection methods to process categorical data, these 
categorical data encoding methods are used:
1. Dummy Encoding
2. Hash Encoding
3. Target Encoding

This is mentioned when used. (in K-NN imputation)

* * *

## Methods of imputation of data

1. discard_data: Complete-case analysis — CCA (default data), available case analysis — ACA.

| Dataset | Available case analysis — ACA (Default data) | Complete-case analysis |
| --- | --- | --- |
| sberbank-russian-housing-market | <ins>2631539.242 (216386.749)</ins> | (rows: 30471 -> 6042) 3545089.937 (324188.444) |
| house-prices-advanced-regression-techniques | <ins>27582.549 (5511.916)</ins> | (rows: 1460 -> 0) - |

* * *

2. univariate_imputation: mean, median, mode (most_frequent) for numeric features - for categorical: the most_frequent value is always imputed. (What about Random Sampling Imputation?)

| Dataset                                     | without imputation       | mean imputation    | median imputation        | mode   imputation        |
|---------------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| sberbank-russian-housing-market             | 2631539.242 (216386.749) | 2628162.876 (210446.647) | 2634163.850 (208134.141) | 2631626.241 (210974.957) |
| house-prices-advanced-regression-techniques | 27582.549 (5511.916)     | 27447.227 (5597.852)     | 27434.728 (5447.184)     | 27594.386 (5419.882)     |
* * *

3. Multiple Imputation by Chained Equations (MICE)

| Dataset                                     | without imputation       | MICE  (1 iterations)     | MICE (2 iterations)      | MICE (3 iterations)      | MICE (4 iterations)      | MICE (5 iterations)      |
|---------------------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| sberbank-russian-housing-market             | 2631539.242 (216386.749) | 2703630.036 (202271.468) | 2696990.977 (213461.259) | 2698336.675 (221695.244) | 2715312.284 (220577.961) | 2710557.522 (220654.923) |
| house-prices-advanced-regression-techniques | 27582.549 (5511.916)     | 27615.901 (5456.934)     | 27493.623 (5590.724)     | 27391.502 (5490.600)     | 27382.827 (5638.731)     | 27564.188 (5472.096)     |
* * *

4. K-NN (k Nearest Neighbour)

| Dataset                                     | without imputation       | K-NN with Dummy Encoding | K-NN with Target Encoding | K-NN with Hash Encoding  |
|---------------------------------------------|--------------------------|--------------------------|---------------------------|--------------------------|
| sberbank-russian-housing-market             | 2631539.242 (216386.749) | 2635246.656 (196394.226) | 2644254.147 (208836.211)  | 2635998.259 (199751.412) |
| house-prices-advanced-regression-techniques | 27582.549 (5511.916)     | 28891.819 (6044.575)     | 27562.018 (5448.057)      | 29576.808 (6096.318)     |
* * *


## Methods of detecting anomalies:

1.  Elliptic Envelope,
2.  One-Class SVM,
3.  Isolation Forest,
4.  LocalOutlierFactor.
