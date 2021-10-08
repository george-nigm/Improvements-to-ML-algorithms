from preprocessing.data_preprocessing import load_and_split_data
from encoding.encoding_categorical import encoding_categorical_choose
from regression_models.regression_model import get_rmse_score
from imputation_models.imputation import imputation_choose


if __name__ == '__main__':
    # Load data
    X, y = load_and_split_data()

    # Choose Encoding method
    X = encoding_categorical_choose(X, y)

    # Choose Imputation method
    X, y = imputation_choose(X, y)

    # Report RMSE of LightGBM Regressor with default data
    print('\nLGBMRegressor model. RMSE: %.3f (%.3f)\n' % (get_rmse_score(X, y)))
