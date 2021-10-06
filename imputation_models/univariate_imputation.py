from sklearn.impute import SimpleImputer
from preprocessing.data_preprocessing import num_cat_columns


def univariate_method_choose(X):
    univariate_diget = input('Choose the imputation numeric univariate method (enter a digit).\n'
                             '1: "Mean"\n'
                             '2: "Median"\n'
                             '3: "Most frequent (Mode)"\n'
                             '(For categorical features, most frequent is always used)\n')

    num, cat = num_cat_columns(X)

    imp = SimpleImputer(strategy='most_frequent').fit(X[cat])
    X[cat] = imp.transform(X[cat])


    if univariate_diget == '1':
        imp = SimpleImputer(strategy='mean').fit(X[num])
        X[num] = imp.transform(X[num])

    if univariate_diget == '2':
        imp = SimpleImputer(strategy='median').fit(X[num])
        X[num] = imp.transform(X[num])

    if univariate_diget == '3':
        imp = SimpleImputer(strategy='most_frequent').fit(X[num])
        X[num] = imp.transform(X[num])

    return X