from preprocessing.missing_data_analysis import miss_percent

def complete_case(X, y):
    # Drop rows with Nan
    X = X.dropna()
    y = y.iloc[X.index]

    # Calculate rate of missed data after imputation_models
    print('\nRate of missing value after imputation')
    miss_percent(X)

    return X, y