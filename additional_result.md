# Additional study

Comparing last realization (from 20-sep-21) and current (from 11-oct-21), there is difference in processing categorical data. 

1) Previously pd.get_dummies() encoding was used. After using it, the detection algorithms worked fine and did not cause an error that Nan was received at the input. Most likely, separate columns of *_isNan were created, and the missed values were lost.

2) In the current version, the *_isnan columns are deleted, the nan values are restored to the table. This results in imputation becoming mandatory before detecting anomalies.

The following table shows comparisons of the two methods.

| Anomalies   detection method |                             Concept                             |          RMSE          |        Importance        | Count |
|:----------------------------:|:---------------------------------------------------------------:|:----------------------:|:------------------------:|:-----:|
|       Elliptic Envelope      | (Old)   One-Hot Encoding (nan has become a feature of *_nan)    | 27626.411   (5486.364) |  0.0000406   (0.0000190) |  113  |
|                              | (New) Target Encoding + Median imputation +   Elliptic Envelope | 27517.042   (5544.918) |  0.0000518   (0.0000541) |  146  |
|         One-Class SVM        | (Old)   One-Hot Encoding (nan has become a feature of *_nan)    | 27678.004   (5533.849) |  0.0000345   (0.0000809) |  560  |
|                              | (New) Target Encoding + Median imputation +   Elliptic Envelope |  27467.854 (5394.370)  | -0.0000723   (0.0001001) |  730  |
|       Isolation Forest       | (Old)   One-Hot Encoding (nan has become a feature of *_nan)    | 27173.646   (5636.475) |  0.0025560   (0.0025115) |  112  |
|                              | (New) Target Encoding + Median imputation +   Elliptic Envelope |  27451.540 (5365.327)  | -0.0016849   (0.0059135) |  146  |
|      LocalOutlierFactor      | (Old)   One-Hot Encoding (nan has become a feature of *_nan)    | 27733.654   (5553.494) | -0.0002322   (0.0005277) |   30  |
|                              | (New) Target Encoding + Median imputation +   Elliptic Envelope |  27552.646 (5413.188)  | -0.0001870   (0.0003610) |   73  |

The current concept (2), due to the fact that it fills in the missing data more accurately, allows you to get better results. Possibly, because *_NaN it's not feature - objects with == 1 can be a different in ground truth.
