"""Model training & evaluation: Linear Regression, Random Forest, XGBoost."""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # falls back to sklearn's gradient boosting
    HAS_XGB = False

RANDOM_STATE = 42


def time_split(df, roles, test_frac=0.2):
    """Chronological split: train on the past, test on the most recent trips."""
    if "date" in roles:
        order = df.sort_values([roles["date"]] + ([roles["time"]] if "time" in roles else [])).index
        cut = int(len(order) * (1 - test_frac))
        return order[:cut], order[cut:]
    idx = np.random.RandomState(RANDOM_STATE).permutation(df.index)
    cut = int(len(idx) * (1 - test_frac))
    return idx[:cut], idx[cut:]


def _preprocessor(num, cat, scale):
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_step, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), cat),
    ])


def build_models(num, cat):
    if HAS_XGB:
        gb_name = "XGBoost"
        gb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        gb_name = "Gradient Boosting (HistGB)"
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4,
                                           l2_regularization=1.0, random_state=RANDOM_STATE)
    return {
        "Baseline (mean)": Pipeline([("prep", _preprocessor(num, cat, False)), ("model", DummyRegressor())]),
        "Linear Regression": Pipeline([("prep", _preprocessor(num, cat, True)), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("prep", _preprocessor(num, cat, True)),
                                      ("model", RidgeCV(alphas=np.logspace(-2, 3, 20)))]),
        "Random Forest": Pipeline([("prep", _preprocessor(num, cat, False)),
                                   ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                                   max_features=0.5, n_jobs=-1,
                                                                   random_state=RANDOM_STATE))]),
        gb_name: Pipeline([("prep", _preprocessor(num, cat, False)), ("model", gb)]),
    }


def metrics(y_true, y_pred):
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                R2=r2_score(y_true, y_pred))


def evaluate_models(models, X_tr, y_tr, X_te, y_te, cv_folds=5):
    rows, fitted = [], {}
    cv = KFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    for name, pipe in models.items():
        cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1,
                             scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
        pipe.fit(X_tr, y_tr)
        m = metrics(y_te, pipe.predict(X_te))
        rows.append({"model": name,
                     "CV_MAE": -cvr["test_neg_mean_absolute_error"].mean(),
                     "CV_RMSE": -cvr["test_neg_root_mean_squared_error"].mean(),
                     "CV_R2": cvr["test_r2"].mean(),
                     "Test_MAE": m["MAE"], "Test_RMSE": m["RMSE"], "Test_R2": m["R2"]})
        fitted[name] = pipe
        print(f"  {name:<28} test MAE {m['MAE']:.2f}  RMSE {m['RMSE']:.2f}  R² {m['R2']:.3f}")
    return pd.DataFrame(rows).set_index("model").round(3), fitted


def tune(pipe, X_tr, y_tr, n_iter=20):
    """Light randomized search on the tree model chosen as best."""
    model = pipe.named_steps["model"]
    if isinstance(model, RandomForestRegressor):
        space = {"model__n_estimators": [200, 400, 600], "model__max_depth": [None, 8, 12, 16],
                 "model__min_samples_leaf": [1, 2, 3, 5, 8], "model__max_features": [0.3, 0.5, 0.7, 1.0]}
    elif HAS_XGB and isinstance(model, XGBRegressor):
        space = {"model__n_estimators": [200, 400, 700], "model__max_depth": [2, 3, 4, 6],
                 "model__learning_rate": [0.02, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0],
                 "model__min_child_weight": [1, 3, 5], "model__reg_lambda": [0.5, 1, 3]}
    elif isinstance(model, HistGradientBoostingRegressor):
        space = {"model__max_iter": [200, 400, 700], "model__max_depth": [2, 3, 4, 6, None],
                 "model__learning_rate": [0.02, 0.05, 0.1], "model__min_samples_leaf": [10, 20, 40],
                 "model__l2_regularization": [0, 0.5, 1, 3]}
    else:
        return pipe, None
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                                scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_


def importance(pipe, X_te, y_te, out_dir, top=15):
    r = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE,
                               scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_te.columns, "importance": r.importances_mean, "std": r.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1.2))
    ax.barh(t["feature"], t["importance"], xerr=t["std"], color="#2E6FBA")
    ax.set(title="Permutation importance (increase in MAE when shuffled)", xlabel="Δ MAE (minutes)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "07_feature_importance.png"), dpi=130); plt.close(fig)
    return imp.round(4)


def diagnostic_plots(y_te, y_pred, results, out_dir, best_name):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    r = results.sort_values("Test_MAE")
    axes[0].barh(r.index, r["Test_MAE"], color=["#E07B39" if i == best_name else "#8A8F98" for i in r.index])
    axes[0].invert_yaxis(); axes[0].set(title="Test MAE by model (lower is better)", xlabel="MAE (min)")
    lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    axes[1].scatter(y_te, y_pred, s=10, alpha=.45, color="#2E6FBA")
    axes[1].plot(lim, lim, ls="--", color="#E07B39")
    axes[1].set(title=f"{best_name}: predicted vs actual", xlabel="Actual delay (min)", ylabel="Predicted (min)")
    res = y_te - y_pred
    axes[2].hist(res, bins=40, color="#2E6FBA")
    axes[2].axvline(0, color="#E07B39", ls="--")
    axes[2].set(title="Residuals (actual − predicted)", xlabel="Minutes")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "06_model_results.png"), dpi=130); plt.close(fig)


def save_model(pipe, meta, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"pipeline": pipe, **meta}, path)
