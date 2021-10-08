import pandas as pd

def miss_count(X,y):

    print(f"Shape of X: {X.shape[0]} * {X.shape[1]}")
    print(f"Shape of y: {len(y)}\n")

    print(f"Count of missing values: {X.isnull().sum().sum()} / {X.shape[0] * X.shape[1]} "
          f"({round(100*X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),3)}%)")
    print(f"Count of missing values: {y.isnull().sum()} / {len(y)} "
          f"({round(100 * y.isnull().sum() / len(y), 3)}%)\n")

    # it will give the total null values present in dataframe
    print("Total Null values count: \n", X.isnull().sum().sort_values(ascending=False))


def miss_percent(X):
    if(type(X) == pd.DataFrame):
        print(f"Count of missing values: {X.isna().sum().sum()} / {X.shape[0] * X.shape[1]} "
              f"({round(100 * X.isna().sum().sum() / (X.shape[0] * X.shape[1]), 3)}%)")

    if(type(X) == pd.Series):
        print(f"Count of missing values: {X.isna().sum()} / {len(X)} "
              f"({round(100 * X.isna().sum() / len(X), 3)}%)")




