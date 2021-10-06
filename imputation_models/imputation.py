from imputation_models.discard_data import complete_case
from imputation_models.univariate_imputation import univariate_method_choose

def imputation_choose(X, y):
    imputation_digit = input('Choose the imputation_models method (enter a digit).\n'
                             '0: No imputation\n'
                             '1: Discard data\n'
                             '2: Univariate imputation_models\n'
                             '3: Multivariate imputation_models\n'
                             '4: Imputation by predictive/statistical models\n')

    if imputation_digit == '0':
        pass

    if imputation_digit == '1':
        X, y = complete_case(X, y)

    if imputation_digit == '2':
        X = univariate_method_choose(X)

    return X, y
