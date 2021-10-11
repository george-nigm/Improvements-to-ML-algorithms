# improvements-to-ml-algorithms

This is repository of project: "Methods of detecting anomalies for improving the quality of machine learning algorithms."

Due to the inability of anomaly detection algorithms to work with categorical values, NaNs: encoding and imputation models have been presented. The usage of these methods is necessary for the process of anomaly detection algorithms - the study of the effect of these methods is the purpose of this research.

* * *

### Pipeline

<p align="center">
  <img src="https://user-images.githubusercontent.com/48650320/136817591-49022a1c-5c42-45e0-bd41-46ea469b9d6f.png" />
</p>

Total number of combinations of encoding, imputation and anomaly detection models: 111.
* * *

### Performance evaluating & description of datasets

The performance of the methods is considered by the effect on the final RMSE when solving the regression problem by LightGBM. The RMSE mean and standard deviation of 10-fold Cross-Validation are used.

Evaluation metrics were calculated on: "house-prices-advanced-regression-techniques". Missing values: 6965 / 115340, (6.039%). The following ratio of "NaN" data are used:

1.  6% of missed data (Initial dataset)
2.  30% of missed data
3.  50% of missed data

(In dataset "house-prices-advanced-regression-techniques" there are missing data in each row. Complete-Case Analysis is not available.)

The results of the experiments are available in the file: [house-prices-scheme-experiments.xlsx](https://github.com/georgii-nigm/Improvements-to-ML-algorithms/blob/master/house-prices-scheme-experiments.xlsx)

* * *

### Expirements

Best combinations for anomalies detection algorithms:

| RMSE                                                       | The importance of the anomaly indicator | The importance of the anomaly indicator |
|------------------------------------------------------------|:---------------------------------------:|:---------------------------------------:|
| 6% of missing data (Initial)                               |                                         |                                         |
| 1. Default model                                           |           27582.549 (5511.916)          |                    -                    |
| 2. Target Encoding + Median imputation                     |           27535.075 (5445.082)          |                    -                    |
| 3. Elliptic Envelope (Target Encoding + Median imputation) |           27517.042 (5544.918)          |          0.0000518 (0.0000541)          |
| 4. One-Class SVM (Target Encoding + Median imputation)     |           27467.854 (5394.370)          |          -0.0000723 (0.0001001)         |
| 5. Isolation Forest (Target Encoding + Mode imputation)    |           27451.540 (5365.327)          |          -0.0016849 (0.0059135)         |
| 6. LocalOutlierFactor( Target Encoding + K-NN)             |           27537.623 (5437.554)          |          0.0000096 (0.0000337)          |
| 7. (Best) MICE                                             |           27391.502 (5490.600)          |                    -                    |
|                                                            |                                         |                                         |
| 30% of missing data                                        |                                         |                                         |
| 1. Default model                                           |           33165.202 (4617.468)          |                    -                    |
| 2. Target Encoding + MICE                                  |           30188.023 (5813.186)          |                    -                    |
| 3. Elliptic Envelope (Target Encoding + MICE)              |           30094.619 (5741.747)          |          -0.0003155 (0.0000924)         |
| 4. One-Class SVM(Target Encoding + MICE)                   |           30142.952 (5766.397)          |          -0.0004070 (0.0003780)         |
| 5. (Best) Isolation Forest (Target Encoding + MICE)        |           30037.433 (5642.734)          |          -0.0095051 (0.0035107)         |
| 6. LocalOutlierFactor (Target Encoding + MICE)             |           30108.693 (5764.039)          |          -0.0000030 (0.0000933)         |
|                                                            |                                         |                                         |
|                                                            |                                         |                                         |
| 50% of missing data                                        |                                         |                                         |
| 1. Default model                                           |           40571.481 (5936.180)          |                    -                    |
| 2. Target Encoding + MICE                                  |           35892.779 (5715.790)          |                    -                    |
| 3. Elliptic Envelope (Target Encoding + MICE)              |           35725.265 (5618.712)          |          0.0001082 (0.0001855)          |
| 4. (Best) One-Class SVM(Target Encoding + MICE)            |           35629.558 (5657.682)          |          -0.0003512 (0.0002492)         |
| 5. Isolation Forest (Target Encoding + MICE)               |           35720.589 (5642.171)          |          -0.0181765 (0.0067545)         |
| 6. LocalOutlierFactor (Target Encoding + MICE)             |           35775.763 (5683.544)          |          0.0001991 (0.0001781)          |


* * *

### Results

1. Using the pipeline described above is beneficial because the RMSE values are improving in all cases.
2. Best preprocessing combination: Target Encoding + MICE
3. Anomaly detection methods reduce RMSE compared to cases with default data / encoding & imputation only.
4. [Difference in previous & current ways to process categorical features](https://github.com/georgii-nigm/Improvements-to-ML-algorithms/blob/master/additional_result.md)
***
(Unclear) With the data being dropped, there is no increase in the std of RMSE. Does it mean that the dataset has enough data and well-described distributions?
