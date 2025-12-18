"""
Unsupervised Hotspot Clustering (K-means) for Uber NYC pickups

- Clusters peak-hour pickups (4-8pm) into spatial hotspots
- Uses 0.01° grid aggregation + sample_weight for speed/stability
- Produces: inertia curve, hotspot map, cluster center table

Deps:
  pip install numpy pandas matplotlib scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# -----------------------
# Config
# -----------------------
GRID = 0.01  # degrees (~1 km)
FILES = [
    "uber-raw-data-apr14.csv",
    "uber-raw-data-may14.csv",
    "uber-raw-data-jun14.csv",
    "uber-raw-data-jul14.csv",
    "uber-raw-data-aug14.csv",
    "uber-raw-data-sep14.csv",
]

# Broad NYC-ish bounds to remove obvious outliers
LAT_MIN, LAT_MAX = 40.5, 41.0
LON_MIN, LON_MAX = -74.3, -73.6

# Peak hours for operational hotspots
PEAK_HOURS = {16, 17, 18, 19, 20}  # 4–8pm inclusive

# Choose K range for elbow plot
K_MIN, K_MAX = 5, 30

# After viewing the elbow plot, set your chosen k here:
FINAL_K = 15

# Optional robustness check: also cluster all pickups
RUN_ALL_PICKUPS = False

# Output folder
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def load_all(files):
    dfs = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f} (put CSVs in the same folder as this script)")
        df = pd.read_csv(f, usecols=["Date/Time", "Lat", "Lon"])
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    raw["dt"] = pd.to_datetime(raw["Date/Time"], errors="coerce")
    raw = raw.dropna(subset=["dt", "Lat", "Lon"]).copy()

    raw = raw[
        raw["Lat"].between(LAT_MIN, LAT_MAX) &
        raw["Lon"].between(LON_MIN, LON_MAX)
    ].copy()

    raw["hour_of_day"] = raw["dt"].dt.hour.astype(int)
    return raw


def aggregate_to_weighted_cells(df, grid=GRID):
    """
    Convert raw points -> grid cells with weights (pickup counts).
    Returns a DataFrame with columns:
      lat_c, lon_c, weight
    """
    lat_bin = np.floor(df["Lat"].to_numpy() / grid).astype(np.int32)
    lon_bin = np.floor(df["Lon"].to_numpy() / grid).astype(np.int32)

    tmp = pd.DataFrame({"lat_bin": lat_bin, "lon_bin": lon_bin})
    cell_counts = tmp.value_counts().reset_index(name="weight")

    # Convert bins back to cell centers
    cell_counts["lat_c"] = (cell_counts["lat_bin"].astype(float) + 0.5) * grid
    cell_counts["lon_c"] = (cell_counts["lon_bin"].astype(float) + 0.5) * grid

    return cell_counts[["lat_c", "lon_c", "weight"]]


def inertia_sweep(X, w, k_min=K_MIN, k_max=K_MAX, random_state=42):
    ks, inertias = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X, sample_weight=w)
        ks.append(k)
        inertias.append(km.inertia_)
        print(f"k={k:>2}  inertia={km.inertia_:.2f}")
    return np.array(ks), np.array(inertias)


def fit_final_kmeans(X, w, k, random_state=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(X, sample_weight=w)
    labels = km.predict(X)
    centers = km.cluster_centers_
    return km, labels, centers


def cluster_summary(cell_df, labels, centers):
    """
    Create a professional table:
      cluster_id, center_lat, center_lon, total_pickups (weight sum), num_cells
    """
    out = cell_df.copy()
    out["cluster"] = labels

    summary = (
        out.groupby("cluster")
           .agg(total_pickups=("weight", "sum"),
                num_cells=("weight", "size"),
                mean_lat=("lat_c", "mean"),
                mean_lon=("lon_c", "mean"))
           .reset_index()
           .sort_values("total_pickups", ascending=False)
    )

    # Use learned centers (more correct than mean_lat/mean_lon)
    centers_df = pd.DataFrame(centers, columns=["center_lat", "center_lon"])
    centers_df["cluster"] = np.arange(len(centers_df))

    summary = summary.merge(centers_df, on="cluster", how="left")

    # Nice column order
    summary = summary[["cluster", "center_lat", "center_lon", "total_pickups", "num_cells"]]
    return summary


def plot_elbow(ks, inertias, title, fname):
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (weighted within-cluster SSE)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_hotspots(cell_df, labels, centers, title, fname):
    """
    Scatter of grid cell centers colored by cluster label.
    Point size reflects weight (scaled).
    """
    # Scale point sizes for visibility
    w = cell_df["weight"].to_numpy()
    sizes = 5 + 40 * (w / (np.percentile(w, 95) + 1e-9))
    sizes = np.clip(sizes, 5, 60)

    plt.figure()
    plt.scatter(cell_df["lon_c"], cell_df["lat_c"], c=labels, s=sizes, alpha=0.5)
    plt.scatter(centers[:, 1], centers[:, 0], s=150, marker="X")  # centers
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------
# Main
# -----------------------
print("Loading data...")
raw = load_all(FILES)
print(f"Rows after cleaning: {len(raw):,}")

# Peak-hour subset
peak = raw[raw["hour_of_day"].isin(PEAK_HOURS)].copy()
print(f"Peak-hour rows (4–8pm): {len(peak):,}")

print("\nAggregating peak-hour pickups to weighted grid cells...")
cells_peak = aggregate_to_weighted_cells(peak, GRID)
print(f"Active peak-hour cells: {len(cells_peak):,}")

X_peak = cells_peak[["lat_c", "lon_c"]].to_numpy()
w_peak = cells_peak["weight"].to_numpy()

print("\nElbow sweep (peak hours):")
ks, inertias = inertia_sweep(X_peak, w_peak, K_MIN, K_MAX)

elbow_path = os.path.join(OUTDIR, "kmeans_elbow_peak.png")
plot_elbow(ks, inertias, "K-means elbow curve (Peak hours 4–8pm)", elbow_path)

print(f"\nFitting final K-means for peak hours with k={FINAL_K}...")
km_peak, labels_peak, centers_peak = fit_final_kmeans(X_peak, w_peak, FINAL_K)

summary_peak = cluster_summary(cells_peak, labels_peak, centers_peak)
print("\n===== Peak-hour hotspot cluster summary (top 10 by pickups) =====")
print(summary_peak.head(10).to_string(index=False))
print("===============================================================\n")

summary_peak_path = os.path.join(OUTDIR, "cluster_centers_peak.csv")
summary_peak.to_csv(summary_peak_path, index=False)
print(f"Saved peak-hour cluster table: {summary_peak_path}")

map_path = os.path.join(OUTDIR, "kmeans_hotspots_peak.png")
plot_hotspots(
    cells_peak, labels_peak, centers_peak,
    f"K-means hotspots (Peak hours 4–8pm), k={FINAL_K}",
    map_path
)


# ============================================================
# ALL-PICKUPS clustering (robustness check)
# ============================================================

RUN_ALL_PICKUPS = True 

if RUN_ALL_PICKUPS:
    print("\n>>> Running ALL-PICKUPS clustering <<<")

    print("\nAggregating ALL pickups to weighted grid cells...")
    cells_all = aggregate_to_weighted_cells(raw, GRID)
    print(f"Active all-hour cells: {len(cells_all):,}")

    # Coordinates and weights
    X_all = cells_all[["lat_c", "lon_c"]].to_numpy()
    w_all = cells_all["weight"].to_numpy()

    # -----------------------
    # Elbow sweep
    # -----------------------
    print("\nElbow sweep (all pickups):")
    ks_all, inertias_all = inertia_sweep(X_all, w_all, K_MIN, K_MAX)

    elbow_all_path = os.path.join(OUTDIR, "kmeans_elbow_all.png")
    plot_elbow(
        ks_all,
        inertias_all,
        "K-means elbow curve (All pickups)",
        elbow_all_path
    )
    print(f"Saved elbow plot: {elbow_all_path}")

    # -----------------------
    # Final K-means
    # -----------------------
    print(f"\nFitting final K-means for ALL pickups with k={FINAL_K}...")
    km_all, labels_all, centers_all = fit_final_kmeans(
        X_all, w_all, FINAL_K
    )

    # -----------------------
    # Cluster summary
    # -----------------------
    summary_all = cluster_summary(cells_all, labels_all, centers_all)

    print("\n===== All-pickups hotspot cluster summary (top 10 by pickups) =====")
    print(summary_all.head(10).to_string(index=False))
    print("==================================================================")

    # Save full table for appendix
    summary_all_path = os.path.join(OUTDIR, "cluster_centers_all.csv")
    summary_all.to_csv(summary_all_path, index=False)
    print(f"\nSaved all-pickups cluster table: {summary_all_path}")

    # -----------------------
    # Map
    # -----------------------
    map_all_path = os.path.join(OUTDIR, "kmeans_hotspots_all.png")
    plot_hotspots(
        cells_all,
        labels_all,
        centers_all,
        f"K-means hotspots (All pickups), k={FINAL_K}",
        map_all_path
    )
    print(f"Saved all-pickups hotspot map: {map_all_path}")

print("\nDone.")
