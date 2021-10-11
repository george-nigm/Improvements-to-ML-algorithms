from preprocessing.missing_data_analysis import miss_percent
from pandas import read_csv
import numpy as np

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


def rate_of_missed_value(X):
    miss_rate_digit = input(f'Choose a rate of missed value (enter a digit).\n'
                          f'0: "Original rate: {round(100 * X.isna().sum().sum() / (X.shape[0] * X.shape[1]),5)} "\n'
                          '1: "10%"\n'
                          '2: "30%"\n'
                          '3: "50%"\n')

    # miss_rate_digit = '3'

    np.random.seed(100)
    if miss_rate_digit == '1':
        X = X.mask(np.random.choice([True, False], size=X.shape, p=[0.1,0.9]))

    if miss_rate_digit == '2':
        X = X.mask(np.random.choice([True, False], size=X.shape, p=[0.3,0.7]))

    if miss_rate_digit == '3':
        X = X.mask(np.random.choice([True, False], size=X.shape, p=[0.5,0.5]))

    return X


def load_and_split_data():
    dataset_digit = input('Choose a dataset (enter a digit).\n'
                          '1: "sberbank-russian-housing-market"\n'
                          '2: "house-prices-advanced-regression-techniques"\n'
                          '3: "CaliforniaHousing"\n'
                          '4: "santander-value-prediction-challenge"\n'
                          '5: "allstate-claims-severity"\n')

    # dataset_digit = '2'

    data = read_csv(datasets[dataset_digit][0], low_memory=False)
    data_y = data[datasets[dataset_digit][1]]
    data_X = data.drop(datasets[dataset_digit][2], axis=1)

    data_X = rate_of_missed_value(data_X)

    print(data_X)
    print(data_y)

    print("\nColumn type counts:")
    print(data_X.dtypes.value_counts())

    # Calculate initial rate of missed data
    print()
    miss_percent(data_X)
    miss_percent(data_y)

    print("\nTop 10 columns with missed data:")
    print((data_X.isna().sum()/len(data_X)).sort_values(ascending = False).head(10))

    not_numeric_to_category(data_X)

    return data_X, data_y


def num_cat_columns(X):

    num_col = X._get_numeric_data().columns
    cat_col = list(set(X.columns) - set(num_col))

    return num_col, cat_col

