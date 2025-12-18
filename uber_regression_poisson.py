"""
Uber demand forecasting (Regression baseline) - Poisson Regression
Target: pickups per (grid cell, hour)
Split: Train Apr–Jul, Val Aug, Test Sep

Install deps (Terminal):
  pip install pandas numpy scikit-learn matplotlib
"""

import os
os.makedirs("figures", exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance


# -----------------------------
# Config
# -----------------------------
GRID = 0.01  # degrees (~1km). Keep as decided.
FILES = [
    "uber-raw-data-apr14.csv",
    "uber-raw-data-may14.csv",
    "uber-raw-data-jun14.csv",
    "uber-raw-data-jul14.csv",
    "uber-raw-data-aug14.csv",
    "uber-raw-data-sep14.csv",
]

# Broad NYC-ish bounding box to remove outliers (simple, practical)
LAT_MIN, LAT_MAX = 40.5, 41.0
LON_MIN, LON_MAX = -74.3, -73.6

ALPHAS_TO_TRY = [0.0, 0.0005, 0.001, 0.01, 0.1]


# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path} (put it in the same folder as this script)")

    # Use only needed columns to save memory
    # Some CSVs may include extra columns; handle that safely.
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]  # strip whitespace in headers

    required = ["Date/Time", "Lat", "Lon"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}. Found columns: {df.columns.tolist()}")

    df = df[required].copy()

    # Downcast floats to reduce memory
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce").astype("float32")
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce").astype("float32")

    return df


def make_features(agg: pd.DataFrame) -> pd.DataFrame:
    # Time features
    agg["hour_of_day"] = agg["hour_ts"].dt.hour.astype(int)
    agg["day_of_week"] = agg["hour_ts"].dt.dayofweek.astype(int)  # Mon=0
    agg["month"] = agg["hour_ts"].dt.month.astype(int)
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(int)

    # Cell id for one-hot (acts like "neighborhood")
    agg["cell_id"] = agg["lat_bin"].astype(str) + "_" + agg["lon_bin"].astype(str)
    return agg


def eval_split(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    # Clip for Poisson deviance stability (model already outputs >=0, but safe)
    y_pred = np.clip(y_pred, 0, None)

    return {
        "Split": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MeanPoissonDeviance": mean_poisson_deviance(y_true, y_pred),
    }


# -----------------------------
# Load + preprocess raw rows
# -----------------------------
print("Reading CSV files...")
dfs = []
for f in FILES:
    print(f"  - {f}")
    dfs.append(safe_read_csv(f))

raw = pd.concat(dfs, ignore_index=True)

print(f"\nRaw rows loaded: {len(raw):,}")

# Parse datetime and basic cleaning
raw["dt"] = pd.to_datetime(raw["Date/Time"], errors="coerce")
raw = raw.dropna(subset=["dt", "Lat", "Lon"]).copy()

# Filter broad NYC bounds to remove outliers
raw = raw[
    raw["Lat"].between(LAT_MIN, LAT_MAX) &
    raw["Lon"].between(LON_MIN, LON_MAX)
].copy()

print(f"Rows after datetime+NYC bounds filter: {len(raw):,}")

# Floor to hour
raw["hour_ts"] = raw["dt"].dt.floor("h")

# Grid bins (integer indices)
raw["lat_bin"] = np.floor(raw["Lat"] / GRID).astype("int32")
raw["lon_bin"] = np.floor(raw["Lon"] / GRID).astype("int32")

# -----------------------------
# Aggregate to (cell, hour): target y = pickup counts
# -----------------------------
print("\nAggregating to (grid_cell, hour)...")
agg = (
    raw.groupby(["hour_ts", "lat_bin", "lon_bin"])
       .size()
       .reset_index(name="pickups")
)

agg["hour_ts"] = pd.to_datetime(agg["hour_ts"])  # ensure datetime
agg = make_features(agg)

print(f"Aggregated rows (cell-hour): {len(agg):,}")
print(f"Active grid cells (unique cell_id): {agg['cell_id'].nunique():,}")

# -----------------------------
# Train/Val/Test split by month
# Train: Apr–Jul (4–7), Val: Aug (8), Test: Sep (9)
# -----------------------------
train = agg[agg["month"].isin([4, 5, 6, 7])].copy()
val   = agg[agg["month"] == 8].copy()
test  = agg[agg["month"] == 9].copy()

print("\nSplit sizes (cell-hour rows):")
print(f"  Train (Apr–Jul): {len(train):,}")
print(f"  Val (Aug):       {len(val):,}")
print(f"  Test (Sep):      {len(test):,}")

# -----------------------------
# Model: Poisson Regression baseline
# -----------------------------
feature_cols_cat = ["hour_of_day", "day_of_week", "month", "cell_id"]
feature_cols_num = ["is_weekend"]

X_train = train[feature_cols_cat + feature_cols_num]
y_train = train["pickups"]
X_val   = val[feature_cols_cat + feature_cols_num]
y_val   = val["pickups"]
X_test  = test[feature_cols_cat + feature_cols_num]
y_test  = test["pickups"]

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ("num", "passthrough", feature_cols_num),
    ]
)

def make_model(alpha: float) -> Pipeline:
    return Pipeline(steps=[
        ("pre", pre),
        ("reg", PoissonRegressor(alpha=alpha, max_iter=1000))
    ])

# Simple alpha selection using validation MPD
print("\nTuning alpha (using Validation Mean Poisson Deviance)...")
best_alpha = None
best_val_mpd = float("inf")

for a in ALPHAS_TO_TRY:
    m = make_model(a)
    m.fit(X_train, y_train)
    val_pred = m.predict(X_val)
    mpd = mean_poisson_deviance(y_val, np.clip(val_pred, 0, None))
    print(f"  alpha={a:<7}  val_MPD={mpd:.4f}")
    if mpd < best_val_mpd:
        best_val_mpd = mpd
        best_alpha = a

print(f"\nBest alpha selected: {best_alpha} (val MPD={best_val_mpd:.4f})")

# Refit on Train+Val, evaluate on Test
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

final_model = make_model(best_alpha)
final_model.fit(X_trainval, y_trainval)

# Predictions
train_pred = final_model.predict(X_train)
val_pred   = final_model.predict(X_val)
test_pred  = final_model.predict(X_test)

# -----------------------------
# Metrics table
# -----------------------------
metrics = []
metrics.append(eval_split("Train (Apr–Jul)", y_train, train_pred))
metrics.append(eval_split("Val (Aug)", y_val, val_pred))
metrics.append(eval_split("Test (Sep)", y_test, test_pred))

metrics_df = pd.DataFrame(metrics)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

metrics_df.to_csv("figures/metrics_table.csv", index=False)

print("\n================= Evaluation Metrics =================")
print(metrics_df.to_string(index=False))
print("======================================================\n")

# -----------------------------
# Helpful plots
# -----------------------------
# 1) Average pickups by hour of day (Train)
avg_by_hour = train.groupby("hour_of_day")["pickups"].mean().reset_index()

plt.figure()
plt.bar(avg_by_hour["hour_of_day"], avg_by_hour["pickups"])
plt.xlabel("Hour of day")
plt.ylabel("Avg pickups per (cell-hour) [Train]")
plt.title("Average demand pattern by hour (Train)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("figures/avg_demand_by_hour_train.png", dpi=300, bbox_inches="tight")
plt.show()

# 2) Actual vs Predicted scatter (Test) - sample for readability
sample_n = min(5000, len(test))
sample_idx = np.random.RandomState(42).choice(len(test), size=sample_n, replace=False)

y_true_s = y_test.iloc[sample_idx].values
y_pred_s = np.clip(test_pred[sample_idx], 0, None)

plt.figure()
plt.scatter(y_true_s, y_pred_s, s=8, alpha=0.3)
maxv = max(y_true_s.max(), y_pred_s.max())
plt.plot([0, maxv], [0, maxv], linestyle="--")
plt.xlabel("Actual pickups (Test)")
plt.ylabel("Predicted pickups (Test)")
plt.title("Actual vs Predicted pickups (Test sample)")
plt.tight_layout()
plt.savefig("figures/actual_vs_pred_test.png", dpi=300, bbox_inches="tight")
plt.show()

# 3) Residual distribution (Test)
resid = y_pred_s - y_true_s
plt.figure()
plt.hist(resid, bins=50)
plt.xlabel("Residual (pred - actual) on Test sample")
plt.ylabel("Frequency")
plt.title("Residual distribution (Test sample)")
plt.tight_layout()
plt.savefig("figures/residuals_test.png", dpi=300, bbox_inches="tight")
plt.show()
