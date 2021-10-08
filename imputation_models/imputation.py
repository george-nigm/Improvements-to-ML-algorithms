from preprocessing.missing_data_analysis import miss_percent
from imputation_models.discard_data import complete_case
from imputation_models.univariate_imputation import univariate_method_choose
from imputation_models.multivariate_imputation import multivariate_method_choose
from imputation_models.k_nearest_neighbors import knn

def imputation_choose(X, y):
    imputation_digit = input('Choose the imputation_models method (enter a digit).\n'
                             '0: No imputation\n'
                             '1: Discard data\n'
                             '2: Univariate imputation_models\n'
                             '3: Multivariate imputation_models\n'
                             '4: K-Nearest Nieghbours\n'
                             '5: Model-Based methods\n')

    if imputation_digit == '0':
        pass

    if imputation_digit == '1':
        X, y = complete_case(X, y)

    if imputation_digit == '2':
        X = univariate_method_choose(X)

    if imputation_digit == '3':
        X = multivariate_method_choose(X)

    if imputation_digit == '4':
        X = knn(X)

    print('\nRate of missing value after imputation')
    miss_percent(X)

    return X, y
