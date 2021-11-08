import numpy as np
import pandas as pd
import category_encoders as ce
from preprocessing.missing_data_analysis import miss_percent

def encoding_categorical_choose(X, y):
    # encoding_diget = input('\nChoose the encoding_models categorical data method (enter a digit).\n'
    #                          '0: No Encoding\n'
    #                          '1: Dummy Encoding\n'
    #                          '2: Target Encoding\n'
    #                          '3: Hash Encoding\n')

    encoding_diget = '2'

    if encoding_diget == '0':
        return X

    if encoding_diget == '1':
        X = pd.get_dummies(data=X, dummy_na=True)

        # Use created columns by dummy_na=True for inserting NaN
        # in base diummies columns

        for column_with_nan in X.filter(regex='_nan$', axis=1).columns:
            prefix_columns = X.filter(regex='^' + column_with_nan[: -4], axis=1).columns

            X.loc[X.index[X[prefix_columns[-1]] == 1].tolist(),
                  prefix_columns[:-1]] = np.nan

            del X[prefix_columns[-1]]
        print()

    if encoding_diget == '2':
        encoder = ce.TargetEncoder(handle_missing = 'return_nan')
        X = encoder.fit_transform(X, y)


    if encoding_diget == '3':
        encoder = ce.HashingEncoder()
        X = encoder.fit_transform(X)

    print(X)
    miss_percent(X)

    return X