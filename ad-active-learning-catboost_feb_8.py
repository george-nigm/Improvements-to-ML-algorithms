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

datasets_nubers = [2,13]

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

        for model_anoamalie in models_list:
            for anomalies_ratio in [0.00, 0.025, 0.05, 0.075, 0.1, 0.15]:

                k = 10
                kf = KFold(n_splits=k, random_state=None)


                metrics_columns = ['support', 'mape', 'rmse', 'mae', 'r2']
                train_metrics = pd.DataFrame(columns = metrics_columns) # K values
                cv_metrics = pd.DataFrame(columns = metrics_columns) # K values
                unobserved_metrics = pd.DataFrame(columns = metrics_columns) # K values
                final_metrics = pd.DataFrame(columns = metrics_columns) # 3 values: avg train, cv, unobserved metrics

                al_fold_cv_metrics = pd.DataFrame(columns=['split', 'queries_ratio'] + metrics_columns)

                split_no = 0
                for train_index, cv_index in kf.split(X):
                    split_no = split_no + 1
                    print(f'AL-{dataset_name}-{model_anoamalie}-{anomalies_ratio} / split_no: {split_no}\n')
                    X_train, X_cv = X.iloc[train_index, :], X.iloc[cv_index, :]
                    y_train, y_cv = y[train_index], y[cv_index]
                    X_unobserved, y_unobserved = X_test.copy(), y_test.copy()

                    print(f'BEFORE AD. X_train shape: {X_train.shape}, y_train: {y_train.shape}')

                    # COMMENT IF DEFAULT REGRESSION MODEL. IF ACTIVE THAN SOFT ANOMALIES DETECTION
                    # # Fit anomalie detector and add column-indicator
                    model_anoamalie.fit(X_train)
                    X_train = X_train.assign(anomalie=model_anoamalie.decision_function(X_train))
                    X_train = X_train.reset_index(drop=True)

                    if anomalies_ratio != 0.00:
                        anomalies_index = int(anomalies_ratio * X_train.shape[0])
                        X_train = X_train.sort_values(by='anomalie', ascending=False)
                        X_train = X_train[anomalies_index:]
                    X_train = X_train.drop('anomalie', axis=1)
                    y_train = y_train[X_train.index]
                    X_train = X_train.reset_index(drop=True)



                    # НУЖНО ВЫКИНУТЬ ПОРЦИЮ АНОМАЛИЙ, ДА!
                    # НО И НУЖНО ВЕРНУТЬ ПОРЯДОК СЭМПЛОВ, CATBOOST К ЭТОМУ ЧУВСТВИТЕЛЕН

                    if (X_cv.shape[0] == 0) or (X_unobserved.shape[0] == 0):
                        continue

                    print(f'AFTER AD. X_train shape: {X_train.shape}, y_train: {y_train.shape}\n')

                    # randomly pick ...% of initial dataset and train shallow CatBoostRegressor with Uncertainty
                    train_random_portion = 0.10
                    step = 0.05
                    queries_ratio = train_random_portion

                    X_train_quere = X_train.sample(frac=queries_ratio)
                    y_train_quere = y_train[X_train_quere.index]
                    X_train_not_quered = X_train.loc[~X_train.index.isin(X_train_quere.index)]
                    y_train_not_quered = y_train[X_train_not_quered.index]

                    queries_index = int(step * X_train_not_quered.shape[0])

                    while queries_ratio <= 1.01:

                        print(f'queries_ratio: {queries_ratio}. X_train_quere shape: {X_train_quere.shape}, Y_train_quere: {y_train_quere.shape}')
                        print(f'not_queried_ratio: {1-queries_ratio}. X_train_not_quered shape: {X_train_not_quered.shape}, y_train_not_quered: {y_train_not_quered.shape}')

                        # Recieve uncertainty for remaining samples of initial dataset and sort according to uncertainty
                        model_regression = CatBoostRegressor(loss_function='RMSEWithUncertainty', posterior_sampling=True, verbose=False, random_seed=0)  # random_seed=0
                        model_regression.fit(X_train_quere, y_train_quere)

                        y_pred_cv = model_regression.virtual_ensembles_predict(X_cv, prediction_type='TotalUncertainty')[:,0]

                        al_fold_cv_metrics.loc[len(al_fold_cv_metrics)] = [split_no , queries_ratio] + get_all_metrics_list(y_pred_cv, y_cv)
                        print([split_no, queries_ratio] + get_all_metrics_list(y_pred_cv, y_cv))
                        print('\n')

                        # Use if Active learning with Catboost uncertainty
                        data_uncertainty = model_regression.virtual_ensembles_predict(X_train_not_quered, prediction_type='TotalUncertainty')[:, 1]
                        knowldege_uncertainty = model_regression.virtual_ensembles_predict(X_train_not_quered, prediction_type='TotalUncertainty')[:, 2]
                        total = data_uncertainty + knowldege_uncertainty
                        X_train_not_quered['uncertainty_total'] = total
                        X_train_not_quered = X_train_not_quered.sort_values(by='uncertainty_total', ascending=False)
                        X_train_not_quered = X_train_not_quered.drop('uncertainty_total', axis=1)

                        # Use if Active learning randomly

                        # X_train_not_quered = X_train_not_quered.sample(frac=1, random_state=0)


                        X_train_quere = pd.concat([X_train_quere , X_train_not_quered[:queries_index]])
                        y_train_quere = y_train[X_train_quere.index]
                        X_train_not_quered = X_train_not_quered[queries_index:]
                        y_train_not_quered = y_train[X_train_not_quered.index]

                        queries_ratio += step

                al_fold_cv_metrics.to_csv(f"results_expirements\hard_experiments/AL-{dataset_name}-{model_anoamalie}-{anomalies_ratio}.csv")

