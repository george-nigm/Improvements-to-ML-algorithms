from sklearn.impute import SimpleImputer
from preprocessing.data_preprocessing import num_cat_columns, dtype_to_category
from preprocessing.missing_data_analysis import miss_percent

def univariate_method_choose(X):
    univariate_diget = input('\nChoose the imputation numeric univariate method (enter a digit).\n'
                             '1: Mean\n'
                             '2: Median\n'
                             '3: Mode (Most frequent)\n'
                             '(for categorical features the most_frequent value is always imputed)\n')

    num, cat = num_cat_columns(X)

    imp = SimpleImputer(strategy='most_frequent').fit(X[cat])
    X[cat] = imp.transform(X[cat])
    dtype_to_category(X)

    if univariate_diget == '1':
        imp = SimpleImputer(strategy='mean').fit(X[num])
        X[num] = imp.transform(X[num])

    if univariate_diget == '2':
        imp = SimpleImputer(strategy='median').fit(X[num])
        X[num] = imp.transform(X[num])

    if univariate_diget == '3':
        imp = SimpleImputer(strategy='most_frequent').fit(X[num])
        X[num] = imp.transform(X[num])

    print('\nRate of missing value after imputation')
    miss_percent(X)

    return X