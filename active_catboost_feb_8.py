import pandas as pd
from preprocessing.data_preprocessing import load_and_split_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor
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
# datasets = {1: ['data/sberbank-russian-housing-market/train.csv', 'price_doc', ['price_doc', 'id', 'timestamp']],
#             2: ['data/house-prices-advanced-regression-techniques/train.csv', 'SalePrice', ['Id', 'SalePrice']],
#             3: ['data/CaliforniaHousing/cal_housing.csv', 'medianHouseValue', 'medianHouseValue'],
#             6: ['parkinsons', 'PPE', ['PPE','subject#']],
#             7: ['bike', 'cnt', ['cnt', 'instant', 'dteday']],
#             8: ['concrete', 'strength', ['strength']],
#             9: ['diamond', 'Price', ['Price']],
#             10: ['traffic', 'traffic_volume', ['traffic_volume']],
#             11: ['insurance', 'charges', ['charges']],
#             12: ['forest', 'area', ['area']],
#             13: ['energy', 'Cooling Load', ['Cooling Load']],
#             #13: ['energy', 'Heating Load', ['Heating Load']],
#             15: ['boston', 'medv' , ['medv']],
#             }

# datasets_nubers = [1,2,3,6,7,8,9,10,11,12,13,15]
datasets_nubers = [13]

if __name__ == '__main__':
    for number in datasets_nubers:
        X, y, dataset_name = load_and_split_data(number)
        y = y.values
        print('\nloading ended\n')

        X = encoding_categorical_choose(X, y)
        print('\nencoding ended\n')

        try:
            X, y = imputation_choose(X, y)
            print('\nimputation ended\n')
        except:
            print("Imputation is not required")

        X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)

        models_list =[EllipticEnvelope(), OneClassSVM(), IsolationForest(), LocalOutlierFactor(novelty = True)]
        # models_list_4 = [OneClassSVM(), IsolationForest(), LocalOutlierFactor(novelty = True)]
        # models_list_5 = [EllipticEnvelope(), IsolationForest(), LocalOutlierFactor(novelty=True)]

        for model_anoamalie in models_list:

            model_regression = CatBoostRegressor(random_state=0)

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
                print(split_no)
                X_train, X_cv = X.iloc[train_index, :], X.iloc[cv_index, :]
                y_train, y_cv = y[train_index], y[cv_index]
                X_unobserved, y_unobserved = X_test.copy(), y_test.copy()

                # COMMENT IF DEFAULT REGRESSION MODEL. IF ACTIVE THAN SOFT ANOMALIES DETECTION
                # # Fit anomalie detector and add column-indicator
                model_anoamalie.fit(X_train)
                X_train = X_train.assign(anomalie=model_anoamalie.decision_function(X_train))
                # X_cv = X_cv.assign(anomalie=model_anoamalie.decision_function(X_cv))
                # X_unobserved = X_unobserved.assign(anomalie=model_anoamalie.decision_function(X_unobserved))

                # COMMENT IF DEFAULT REGRESSION MODEL. IF ACTIVE THAN HARD ANOMALIES DETECTION
                # # Fit anomalie detector and add column-indicator
                # model_anoamalie.fit(X_train)
                # X_train = X_train.assign(anomalie = model_anoamalie.predict(X_train))
                # X_cv = X_cv.assign(anomalie = model_anoamalie.predict(X_cv))
                # X_unobserved = X_unobserved.assign(anomalie = model_anoamalie.predict(X_unobserved))

                print('\nhere is X_train.anomalie starts\n')
                print(X_train.columns)
                print(X_train.anomalie)
                print('\nhere is X_train.anomalie ends\n')


                ## COMMENT IF NOT TRAINING ONLY ON ANOMALIES DATA
                # 1 if train on normal data, -1 if train on anomalies
                anomalies_ratio = 0.20
                X_train = X_train.reset_index(drop=True)
                X_train = X_train.sort_values(by='anomalie', ascending=False)
                anomalies_index = int(anomalies_ratio * X_train.shape[0])
                X_train = X_train[anomalies_index:]
                X_train = X_train.iloc[:,:-1] # drop anomalies column. Lost active if anomalies detector is used

                y_train = y_train[X_train.index]

                # COMMENT IF IN TEST DEFAULT DATA CONFIGURATION
                # -1 if test on anomalies, 1 if test on normal data
                # test_on_anomalie = 1
                # expirement_title = 'testing_on_' + str(test_on_anomalie)
                # X_cv = X_cv.reset_index(drop=True)
                # X_unobserved = X_unobserved.reset_index(drop=True)
                # X_cv = X_cv[X_cv.index.isin(X_cv[X_cv.anomalie * test_on_anomalie > 0].index)]
                # X_unobserved = X_unobserved[
                #     X_unobserved.index.isin(X_unobserved[X_unobserved.anomalie * test_on_anomalie > 0].index)]
                # y_cv = y_cv[X_cv.index]
                # y_unobserved = y_unobserved[X_unobserved.index]

                if (X_cv.shape[0] == 0) or (X_unobserved.shape[0] == 0):
                    continue

                model_regression.fit(X_train, y_train)
                y_pred_train = model_regression.predict(X_train)
                y_pred_cv = model_regression.predict(X_cv)
                y_pred_unobserved = model_regression.predict(X_unobserved)
                train_metrics.loc[len(train_metrics)] = get_all_metrics_list(y_pred_train, y_train)
                cv_metrics.loc[len(cv_metrics)] = get_all_metrics_list(y_pred_cv, y_cv)
                unobserved_metrics.loc[len(unobserved_metrics)] = get_all_metrics_list(y_pred_unobserved, y_unobserved)

            print(final_metrics)

            final_metrics.loc['train'] = train_metrics.sum() / k
            final_metrics.loc['cv'] = cv_metrics.sum() / k
            final_metrics.loc['unobserved'] = unobserved_metrics.sum() / k
            print(model_anoamalie)
            print(final_metrics)

            # final_metrics.to_csv(f"results_expirements/hard_experiments/hard_training_normal/{dataset_name}-{model_anoamalie}.csv")
            final_metrics.to_csv(f"results_expirements\hard_experiments/{dataset_name}-{model_anoamalie}.csv")





            # final_metrics.to_csv(f"results_expirements/soft_experiments/soft_{expirement_title[:10]}/{expirement_title}-{dataset_name}-{model_anoamalie}.csv")
            # if use soft value of abnormality
            # final_metrics.to_csv(f"results_expirements/soft_experiments/soft-{dataset_name}-{model_anoamalie}.csv")

