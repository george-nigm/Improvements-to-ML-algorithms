from preprocessing.missing_data_analysis import miss_percent

def complete_case(X, y):
    # Drop rows with Nan
    X = X.dropna()
    y = y.iloc[X.index]

    return X, y