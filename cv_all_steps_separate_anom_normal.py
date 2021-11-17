import pandas as pd
from preprocessing.data_preprocessing import load_and_split_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
from encoding_models.encoding_categorical import encoding_categorical_choose
from imputation_models.imputation import imputation_choose
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def get_all_metrics_list(y_true, y_pred):
    result = [len(y_pred),
              mean_absolute_percentage_error(y_pred, y_true),
              mean_squared_error(y_pred, y_true, squared=False),
              mean_absolute_error(y_pred, y_true),
              r2_score(y_pred, y_true)]
    result = [round(x,3) for x in result]
    return result

# dataset_digit = input('Choose a dataset (enter a digit).\n'
#                       '1: "sberbank-russian-housing-market"\n'
#                       '2: "house-prices-advanced-regression-techniques"\n'
#                       '3: "CaliforniaHousing"\n'
#                       '4: "santander-value-prediction-challenge"\n'
#                       '5: "allstate-claims-severity"\n')

if __name__ == '__main__':
    X, y, dataset_name = load_and_split_data(4)
    y = y.values
    print('\nloading ended\n')

    X = encoding_categorical_choose(X, y)
    print('\nencoding ended\n')

    # X, y = imputation_choose(X, y)
    print('\nimputation ended\n')


    X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)

    # model_anoamalie = EllipticEnvelope()
    # model_anoamalie = OneClassSVM()]
    # model_anoamalie = IsolationForest()
    # model_anoamalie = LocalOutlierFactor(novelty = True)


    # models_list_1_2_3 =[EllipticEnvelope(), OneClassSVM(), IsolationForest(), LocalOutlierFactor(novelty = True)]
    models_list_4 = [OneClassSVM(), IsolationForest(), LocalOutlierFactor(novelty = True)]
    # models_list_5 = [EllipticEnvelope(), IsolationForest(), LocalOutlierFactor(novelty=True)]

    for model_anoamalie in models_list_4:

        model_regression = lgb.LGBMRegressor(random_state=0)

        k = 10

        kf = KFold(n_splits=k, random_state=None)

        metrics_columns = ['support', 'mape', 'rmse', 'mae', 'r2']

        train_metrics = pd.DataFrame(columns = metrics_columns) # K values
        cv_metrics = pd.DataFrame(columns = metrics_columns) # K values
        unobserved_metrics = pd.DataFrame(columns = metrics_columns) # K values

        final_metrics = pd.DataFrame(columns = metrics_columns) # 3 values: avg train, cv, unobserved metrics

        split_no = 0
        for train_index, cv_index in kf.split(X):
            split_no = split_no + 1
            print(split_no )

            X_train, X_cv = X.iloc[train_index, :], X.iloc[cv_index, :]
            y_train, y_cv = y[train_index], y[cv_index]
            X_unobserved, y_unobserved = X_test.copy(), y_test.copy()



            # COMMENT IF DEFAULT REGRESSION MODEL. IF ACTIVE THAN SOFT ANOMALIES DETECTION
            # # Fit anomalie detector and add column-indicator
            model_anoamalie.fit(X_train)
            X_train = X_train.assign(anomalie=model_anoamalie.decision_function(X_train))
            X_cv = X_cv.assign(anomalie=model_anoamalie.decision_function(X_cv))
            X_unobserved = X_unobserved.assign(anomalie=model_anoamalie.decision_function(X_unobserved))


            # COMMENT IF DEFAULT REGRESSION MODEL. IF ACTIVE THAN HARD ANOMALIES DETECTION
            # # Fit anomalie detector and add column-indicator
            # model_anoamalie.fit(X_train)
            # X_train = X_train.assign(anomalie = model_anoamalie.predict(X_train))
            # X_cv = X_cv.assign(anomalie = model_anoamalie.predict(X_cv))
            # X_unobserved = X_unobserved.assign(anomalie = model_anoamalie.predict(X_unobserved))


            print('\nhere is X_train.anomalie starts\n')
            print(X_train.anomalie)
            print('\nhere is X_train.anomalie ends\n')


            ## COMMENT IF NOT TRAINING ONLY ON ANOMALIES DATA
            # 1 if train on normal data, -1 if train on anomalies
            # train_on_anomalie = 1
            # X_train = X_train.reset_index(drop=True)
            # X_train = X_train[X_train.index.isin(X_train[X_train.anomalie == train_on_anomalie].index)]
            # y_train = y_train[X_train.index]


            # COMMENT IF IN TEST DEFAULT DATA CONFIGURATION
            # 1 if train on normal data, -1 if train on anomalies
            test_on_anomalie = -1
            expirement_title = 'testing_on_' + str(test_on_anomalie)
            X_cv = X_cv.reset_index(drop=True)
            X_unobserved = X_unobserved.reset_index(drop = True)

            # "<" if test on anomalies, ">" if test on normal data
            X_cv = X_cv[X_cv.index.isin(X_cv[X_cv.anomalie < 0].index)]

            X_unobserved = X_unobserved[X_unobserved.index.isin(X_unobserved[X_unobserved.anomalie == test_on_anomalie].index)]
            y_cv = y_cv[X_cv.index]
            y_unobserved = y_unobserved[X_unobserved.index]

            if (X_cv.shape[0] == 0) or (X_unobserved.shape[0] == 0):
                continue

            model_regression.fit(X_train, y_train)

            y_pred_train = model_regression.predict(X_train)
            y_pred_cv = model_regression.predict(X_cv)
            y_pred_unobserved = model_regression.predict(X_unobserved)

            train_metrics.loc[len(train_metrics)] = get_all_metrics_list(y_pred_train, y_train)
            cv_metrics.loc[len(cv_metrics)] = get_all_metrics_list(y_pred_cv, y_cv)
            unobserved_metrics.loc[len(unobserved_metrics)] = get_all_metrics_list(y_pred_unobserved, y_unobserved)

        final_metrics.loc['train'] = train_metrics.sum() / k
        final_metrics.loc['cv'] = cv_metrics.sum() / k
        final_metrics.loc['unobserved'] = unobserved_metrics.sum() / k

        print(final_metrics)

        # if training / testing saving files command
        final_metrics.to_csv(f"results_expirements/soft_experiments/soft_{expirement_title[:-3]}/{expirement_title}-{dataset_name}-{model_anoamalie}.csv")

        # if use soft value of abnormality
        # final_metrics.to_csv(f"results_expirements/soft_experiments/soft-{dataset_name}-{model_anoamalie}.csv")

