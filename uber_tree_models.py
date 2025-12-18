"""
Tree-based models for Uber demand forecasting
Models:
  - Random Forest Regressor
  - Gradient Boosting Regressor (Histogram-based)

Target: Pickups per (grid_cell, hour)
Split: Train Apr–Jul, Val Aug, Test Sep
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance


# -----------------------
# Config
# -----------------------
GRID = 0.01

FILES = [
    "uber-raw-data-apr14.csv",
    "uber-raw-data-may14.csv",
    "uber-raw-data-jun14.csv",
    "uber-raw-data-jul14.csv",
    "uber-raw-data-aug14.csv",
    "uber-raw-data-sep14.csv",
]

LAT_MIN, LAT_MAX = 40.5, 41.0
LON_MIN, LON_MAX = -74.3, -73.6


# -----------------------
# Load + preprocess
# -----------------------
dfs = []
for f in FILES:
    df = pd.read_csv(f, usecols=["Date/Time", "Lat", "Lon"])
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)

raw["dt"] = pd.to_datetime(raw["Date/Time"], errors="coerce")
raw = raw.dropna()

raw = raw[
    raw["Lat"].between(LAT_MIN, LAT_MAX) &
    raw["Lon"].between(LON_MIN, LON_MAX)
].copy()

raw["hour_ts"] = raw["dt"].dt.floor("h")

raw["lat_bin"] = np.floor(raw["Lat"] / GRID).astype(int)
raw["lon_bin"] = np.floor(raw["Lon"] / GRID).astype(int)

agg = (
    raw.groupby(["hour_ts", "lat_bin", "lon_bin"])
       .size()
       .reset_index(name="pickups")
)

# Time features
agg["hour_of_day"] = agg["hour_ts"].dt.hour
agg["day_of_week"] = agg["hour_ts"].dt.dayofweek
agg["month"] = agg["hour_ts"].dt.month
agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(int)


# -----------------------
# Train / Val / Test split
# -----------------------
train = agg[agg["month"].isin([4, 5, 6, 7])]
val   = agg[agg["month"] == 8]
test  = agg[agg["month"] == 9]

FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "lat_bin",
    "lon_bin",
]

X_train = train[FEATURES]
y_train = train["pickups"]

X_val = val[FEATURES]
y_val = val["pickups"]

X_test = test[FEATURES]
y_test = test["pickups"]

# Log-transform target for tree models
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
y_test_log  = np.log1p(y_test)


# -----------------------
# Models
# -----------------------
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

gb = HistGradientBoostingRegressor(
    max_depth=8,
    learning_rate=0.05,
    max_iter=200,
    random_state=42
)

rf.fit(X_train, y_train_log)
gb.fit(X_train, y_train_log)


# -----------------------
# Evaluation helper
# -----------------------
def evaluate(model, X, y_true):
    y_pred = np.expm1(model.predict(X))
    y_pred = np.clip(y_pred, 1e-6, None)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MPD": mean_poisson_deviance(y_true, y_pred),
        "y_pred": y_pred
    }


# -----------------------
# Compute metrics
# -----------------------
rows = []

for name, model in [("Random Forest", rf), ("Gradient Boosting", gb)]:
    for split, X, y in [
        ("Train", X_train, y_train),
        ("Val",   X_val,   y_val),
        ("Test",  X_test,  y_test),
    ]:
        res = evaluate(model, X, y)
        rows.append({
            "Model": name,
            "Split": split,
            "MAE": res["MAE"],
            "RMSE": res["RMSE"],
            "MeanPoissonDeviance": res["MPD"],
        })

metrics_df = pd.DataFrame(rows)

print("\n================ Tree Model Performance ================")
print(metrics_df.to_string(index=False, float_format="%.4f"))
print("========================================================\n")


# -----------------------
# Plots (Test set only)
# Choose better model - RF
# -----------------------

best_model = rf

y_test_pred = np.expm1(best_model.predict(X_test))

# Keep strictly positive for consistency with Poisson-type metrics
y_test_pred = np.clip(y_test_pred, 1e-6, None)

# Sample for clarity
n = min(5000, len(y_test))
idx = np.random.RandomState(42).choice(len(y_test), n, replace=False)

# Actual vs Predicted
plt.figure()
plt.scatter(y_test.iloc[idx], y_test_pred[idx], s=8, alpha=0.3)
maxv = max(y_test.iloc[idx].max(), y_test_pred[idx].max())
plt.plot([0, maxv], [0, maxv], linestyle="--")
plt.xlabel("Actual pickups (Test)")
plt.ylabel("Predicted pickups (Test)")
plt.title("Actual vs Predicted pickups (Random Forest)")
plt.tight_layout()
plt.show()

# Residuals
resid = y_test_pred[idx] - y_test.iloc[idx].values

plt.figure()
plt.hist(resid, bins=50)
plt.xlabel("Residual (pred − actual)")
plt.ylabel("Frequency")
plt.title("Residual distribution (Random Forest, Test)")
plt.tight_layout()
plt.show()

