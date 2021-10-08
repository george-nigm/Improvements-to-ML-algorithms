from sklearn.impute import KNNImputer

def knn(X):
    
    knn_imputer = KNNImputer(n_neighbors=3)
    X = knn_imputer.fit_transform(X)

    return X