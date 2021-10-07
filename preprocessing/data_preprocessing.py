import pandas as pd
from pandas import read_csv

# a dictionary with structure {'key': [file_path, target, [columns to drop from X]]
datasets = {'1': ['data/sberbank-russian-housing-market/train.csv', 'price_doc', ['price_doc', 'id', 'timestamp']],
            '2': ['data/house-prices-advanced-regression-techniques/train.csv', 'SalePrice', ['Id', 'SalePrice']],
            '3': ['data/CaliforniaHousing/cal_housing.csv', 'medianHouseValue', 'medianHouseValue'],
            '4': ['data/santander-value-prediction-challenge/train.csv', 'target', ['ID', 'target']],
            '5': ['data/allstate-claims-severity/train.csv', 'loss', ['id', 'loss']]}


def not_numeric_to_category(X):
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype == 'str' or X[c].dtype == 'bool' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category')
    return X

def load_and_split_data():
    dataset_digit = input('Choose a dataset (enter a digit).\n'
                          '1: "sberbank-russian-housing-market"\n'
                          '2: "house-prices-advanced-regression-techniques"\n'
                          '3: "CaliforniaHousing"\n'
                          '4: "santander-value-prediction-challenge"\n'
                          '5: "allstate-claims-severity"\n')

    data = pd.read_csv(datasets[dataset_digit][0], low_memory=False)
    data_y = data[datasets[dataset_digit][1]]
    data_X = data.drop(datasets[dataset_digit][2], axis=1)

    not_numeric_to_category(data_X)

    return data_X, data_y


def num_cat_columns(X):

    num_col = X._get_numeric_data().columns
    cat_col = list(set(X.columns) - set(num_col))

    return num_col, cat_col