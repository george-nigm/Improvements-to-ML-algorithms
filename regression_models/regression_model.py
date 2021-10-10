import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def get_rmse_score(X, y):
    # Initializing model.
    model = lgb.LGBMRegressor(random_state=0)

    # initializing separator.
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Cross-validation.
    n_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')


    if 'anomalie' in X.columns:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        model.fit(X_train, y_train)
        r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)

        print(f"\nThe relative importance of the {X.columns[-1]} indicator: {r.importances_mean[-1]:.7f} "
              f"({r.importances_std[-1]:.7f})")
    else:
        pass

    return -n_scores.mean(), n_scores.std()

