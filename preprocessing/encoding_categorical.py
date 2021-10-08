import pandas as pd
import category_encoders as ce
from preprocessing.missing_data_analysis import miss_percent

def encoding_categorical_choose(X):
    encoding_diget = input('\nChoose the encoding categorical data method (enter a digit).\n'
                             '0: No Encoding\n'
                             '1: Dummy Encoding\n'
                             '2: Hash Encoding\n'
                             '3: Target Encoding\n')

    if encoding_diget == '0':
        return X

    if encoding_diget == '1':
        X = pd.get_dummies(data=X, drop_first=True, dummy_na=True)

    if encoding_diget == '2':
        encoder = ce.HashingEncoder()
        X = encoder.fit_transform(X)

    if encoding_diget == '3':
        encoder = ce.TargetEncoder()
        X = encoder.fit_transform(X)

    print(X)
    miss_percent(X)

    return X