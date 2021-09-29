import pandas as pd
from pandas import read_csv

# a dictionary with structure {'key': [file_path, target, [columns to drop from X]]
datasets = {'1': ['data/sberbank-russian-housing-market/train.csv', 'price_doc', ['price_doc', 'id', 'timestamp']],
            '2': ['data/house-prices-advanced-regression-techniques/train.csv', 'SalePrice', ['Id', 'SalePrice']],
            '3': ['data/CaliforniaHousing/cal_housing.csv', 'medianHouseValue', 'medianHouseValue']}


def load_and_split_data():
    dataset_digit = input('Choose a dataset (enter a digit).\n'
                          '1: "sberbank-russian-housing-market"\n'
                          '2: "house-prices-advanced-regression-techniques"\n'
                          '3: "CaliforniaHousing"\n')
    data = pd.read_csv(datasets[dataset_digit][0])
    data_y = data[datasets[dataset_digit][1]]
    data_X = data.drop(datasets[dataset_digit][2], axis=1)

    for c in data_X.columns:
        if data_X[c].dtype == 'object' or data_X[c].dtype == 'bool' or data_X[c].dtype.name == 'category':
            data_X[c] = data_X[c].astype('category')

    return data_X, data_y