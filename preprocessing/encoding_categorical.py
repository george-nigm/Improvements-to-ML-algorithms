import pandas as pd
import category_encoders as ce

def encoding_categorical_choose(X):
    encoding_diget = input('\nChoose the encoding categorical data method (enter a digit).\n'
                             '1: Dummy Encoding\n'
                             '2: Hash Encoding\n'
                             '3: Target Encoding\n')

    if encoding_diget == '1':
        X = pd.get_dummies(data=X, drop_first=True)

    if encoding_diget == '2':
        encoder = ce.HashingEncoder()
        X = encoder.fit_transform(X)

    if encoding_diget == '3':
        encoder = ce.TargetEncoder()
        X = encoder.fit_transform(X)
