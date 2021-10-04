from preprocessing.data_preprocessing import load_and_split_data
from preprocessing.missing_data_analysis import miss_percent
from regression_models.regression_model import get_rmse_score

if __name__ == '__main__':
    # Load data
    X, y = load_and_split_data()

    # Calculate initial rate of missed data
    miss_percent(X)

    # Imputation method
    # X, y = complete_case(X, y)

    # Report RMSE of LightGBM Regressor with default data
    print('Default LGBMRegressor model\nRMSE: %.3f (%.3f)' % (get_rmse_score(X, y)))

