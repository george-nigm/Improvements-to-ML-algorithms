# improvements-to-ml-algorithms

This is repository of project: "Methods of detecting anomalies for improving the quality of machine learning algorithms."

Due to the inability of anomaly detection algorithms to work with categorical values, NaNs: encoding and imputation models have been presented. The usage of these methods is necessary for the process of anomaly detection algorithms - the study of the effect of these methods is the purpose of this research.

* * *

### Pipeline

![](https://user-images.githubusercontent.com/48650320/136623212-9575eb61-1244-4398-8510-6e16bbbb06cc.png)

* * *

### Performance evaluating & description of datasets

The performance of the methods is considered by the effect on the final RMSE when solving the regression problem by LightGBM. The RMSE mean and standard deviation of 10-fold Cross-Validation are used.

Evaluation metrics were calculated on: "house-prices-advanced-regression-techniques. Missing values: 6965 / 115340, (6.039%)". The following ratio of "NaN" data are used:

1.  6% of missed data (Initial dataset)
2.  30% of missed data
3.  50% of missed data

The results of the experiments are available in the file:

* * *

### Results
