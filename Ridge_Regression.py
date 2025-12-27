from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import *

df = pd.read_csv('/content/drive/MyDrive/DS_Proj3/train.csv')
df_test = pd.read_csv('/content/drive/MyDrive/DS_Proj3/test.csv')

df_full = df.drop(columns=["Id"])
y = df_full["SalePrice"]
X = df_full.drop(columns=["SalePrice"])

for c in X.columns:
    if X[c].dtype == object:
        if X[c].dropna().str.replace(".", "", 1).str.isnumeric().all():
            X[c] = pd.to_numeric(X[c])

numeric_cols = X.select_dtypes(include=[int, float]).columns
categorical_cols = X.select_dtypes(exclude=[int, float]).columns

skew = X[numeric_cols].skew().abs()
skewed = skew[skew > 0.75].index.tolist()
non_skewed = list(set(numeric_cols) - set(skewed))

log_tf = FunctionTransformer(np.log1p, validate=False)

linear_preprocess = ColumnTransformer([
    ("skew", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", log_tf),
        ("scale", RobustScaler())
    ]), skewed),
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", RobustScaler())
    ]), non_skewed),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("prep", linear_preprocess),
    ("ridge", Ridge(alpha=1.0, random_state=42))
])

model_tgt = TransformedTargetRegressor(
    regressor=model,
    func=np.log1p,
    inverse_func=np.expm1
)

model_tgt.fit(X_train, y_train)
preds = model_tgt.predict(X_val)

print("RMSE:", np.sqrt(mean_squared_error(y_val, preds)))
print("MAE:", mean_absolute_error(y_val, preds))
print("R2:", r2_score(y_val, preds))

test_ids = df_test["Id"]
test_proc = df_test.drop(columns=["Id"])

for c in test_proc.columns:
    if test_proc[c].dtype == object:
        if test_proc[c].dropna().str.replace(".", "", 1).str.isnumeric().all():
            test_proc[c] = pd.to_numeric(test_proc[c])

sub = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": model_tgt.predict(test_proc)
})
sub.to_csv("submission_RidgeRegression.csv", index=False)
