import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

def get_rmse_score(X, y):
    # Initializing model.
    model = lgb.LGBMRegressor()

    # initializing separator.
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Cross-validation.
    n_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

    return n_scores.mean(), n_scores.std()