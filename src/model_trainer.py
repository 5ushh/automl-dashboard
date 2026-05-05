import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
)

XGBOOST_AVAILABLE = False
LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    pass


def get_classification_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    return models


def get_regression_models():
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf"),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    return models


def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))
    avg = "weighted" if n_classes > 2 else "binary"

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average=avg, zero_division=0),
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
    }
    try:
        if n_classes == 2:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["ROC AUC"] = roc_auc_score(y_test, proba)
        else:
            proba = model.predict_proba(X_test)
            metrics["ROC AUC"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
    except Exception:
        metrics["ROC AUC"] = None
    return metrics


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "R2 Score": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
    }


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        return dict(zip(feature_names, np.abs(coef)))
    return None


def train_all_models(X_train, X_test, y_train, y_test, task_type, progress_callback=None):
    models = get_classification_models() if task_type == "classification" else get_regression_models()
    results = {}
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        try:
            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            if task_type == "classification":
                metrics = evaluate_classification(model, X_test, y_test)
            else:
                metrics = evaluate_regression(model, X_test, y_test)

            results[name] = {
                "model": model,
                "metrics": metrics,
                "training_time": round(elapsed, 4),
            }
        except Exception as e:
            results[name] = {"model": None, "metrics": {}, "training_time": 0, "error": str(e)}

        if progress_callback:
            progress_callback((i + 1) / total)

    return results


def get_best_model(results, task_type):
    primary = "Accuracy" if task_type == "classification" else "R2 Score"
    best_name, best_score = None, -np.inf
    for name, info in results.items():
        score = info["metrics"].get(primary, -np.inf)
        if score is not None and score > best_score:
            best_score = score
            best_name = name
    return best_name, results[best_name] if best_name else (None, None)
