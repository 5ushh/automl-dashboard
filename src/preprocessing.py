import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

HIGH_CARDINALITY_THRESHOLD = 50
VARIANCE_THRESHOLD = 0.01
CLASSIFICATION_UNIQUE_LIMIT = 20


def detect_task_type(df, target_col):
    col = df[target_col]
    if col.dtype == object:
        return "classification"
    if col.nunique() <= CLASSIFICATION_UNIQUE_LIMIT:
        return "classification"
    return "regression"


def full_preprocess(df, target_col, impute_strategy="auto", scaler_type="standard", test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    df = df.copy()
    dropped_cols = []

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    task_type = detect_task_type(df, target_col)

    # Encode target if classification
    target_encoder = None
    if task_type == "classification" and y.dtype == object:
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=target_col)

    # Drop high cardinality categorical columns
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        if X[col].nunique() > HIGH_CARDINALITY_THRESHOLD:
            X = X.drop(columns=[col])
            dropped_cols.append(col)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Imputation
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    if impute_strategy == "drop":
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]
    else:
        num_strat = "median" if impute_strategy in ("auto", "median") else "mean"
        if num_cols:
            num_imputer = SimpleImputer(strategy=num_strat)
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # Label encode remaining categorical columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str).map(
            lambda x: x if x in le.classes_ else le.classes_[0]
        ))
        label_encoders[col] = le

    # Variance thresholding
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    X_train_arr = selector.fit_transform(X_train)
    X_test_arr = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()
    X_train = pd.DataFrame(X_train_arr, columns=selected_features)
    X_test = pd.DataFrame(X_test_arr, columns=selected_features)

    # Feature scaling
    fitted_scaler = None
    if scaler_type == "standard":
        fitted_scaler = StandardScaler()
    elif scaler_type == "minmax":
        fitted_scaler = MinMaxScaler()

    if fitted_scaler:
        X_train = pd.DataFrame(fitted_scaler.fit_transform(X_train), columns=selected_features)
        X_test = pd.DataFrame(fitted_scaler.transform(X_test), columns=selected_features)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "task_type": task_type,
        "features": selected_features,
        "target_encoder": target_encoder,
        "scaler": fitted_scaler,
        "dropped_cols": dropped_cols,
    }
