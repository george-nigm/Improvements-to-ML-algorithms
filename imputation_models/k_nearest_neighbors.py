from sklearn.impute import KNNImputer
import pandas as pd

def knn(X):

    columns_names = X.columns
    knn_imputer = KNNImputer(n_neighbors=3)
    X = knn_imputer.fit_transform(X)

    new_df = pd.DataFrame(columns=columns_names, data=X)

    return new_df