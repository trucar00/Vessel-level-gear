import pandas as pd
import numpy as np
import random
from pathlib import Path
import pickle
import gc

# CREATE train, validation and test. Exclude global validation and test mmsis. 

GEARS = ["Trål", "Krokredskap", "Snurrevad", "Garn"]

BASE_FEATURES   = ["cog_interp_sin", "cog_interp_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES
WINDOW = 120
TRAIN_SLIDE  = 60
TEST_SLIDE = 120

FOLDER_BASE = "../../LSTM/three_months/resampled"
TRAIN_FILES = [
    f"{FOLDER_BASE}/2023_1_3_feats.parquet", 
    f"{FOLDER_BASE}/2023_4_6_feats.parquet", 
    f"{FOLDER_BASE}/2023_7_9_feats.parquet", 
    f"{FOLDER_BASE}/2023_10_12_feats.parquet"
    ]

VAL_TEST_FILES = [
    f"{FOLDER_BASE}/2024_1_3_feats.parquet", 
    f"{FOLDER_BASE}/2024_4_6_feats.parquet", 
    f"{FOLDER_BASE}/2024_7_9_feats.parquet", 
    f"{FOLDER_BASE}/2024_10_12_feats.parquet"
]


def all_mmsis_in(files):
    s = set()
    for f in files:
        mmsis = pd.read_parquet(f, columns=["mmsi"])["mmsi"]
        mmsis = pd.to_numeric(mmsis, errors="coerce").dropna().astype("int64")
        s.update(mmsis.unique())
    return s

def get_global_val_test_mmsis(which, path="../../train_val_test_mmsis_FINAL.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    return set(split_df.loc[split_df["split"] == which, "mmsi"])
 
# Temporal split by year, with global MMSI split for unseen validation/test vessels.
val_mmsis = get_global_val_test_mmsis(which="validation")
test_mmsis = get_global_val_test_mmsis(which="test")
all_mmsis_in_train = all_mmsis_in(TRAIN_FILES)
train_mmsis = all_mmsis_in_train - val_mmsis - test_mmsis
assert train_mmsis.isdisjoint(val_mmsis), "Train/val MMSIs overlap!"
assert train_mmsis.isdisjoint(test_mmsis), "Train/test MMSIs overlap!"
print(f"Train (all 2023) vessels: {len(train_mmsis)} | Val (2024) vessels: {len(val_mmsis)} | Test (2024) vessels: {len(test_mmsis)}")

def add_features(df, mmsis):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df.loc[df['report'] == 'Not', 'report'] = 'Snurrevad' # TRY BOTH WITH AND WITHOUT NOT AS SNURREVAD, many not reports are in fact snurrevad...
    df = df[df["mmsi"].isin(mmsis)]
    df = df[df["report"].isin(GEARS)]
    m = df["date_time_utc"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    return df

# ------------------------------------------------------------------
# Normalization stats -- fit on TRAIN (2023) only
# ------------------------------------------------------------------
def get_mu_sigma(mu_sigma_path="parameters_cnn_gear_2023_train.pkl"):
    mu_sigma_path = Path(f"{mu_sigma_path}")
    if mu_sigma_path.exists():
        print(f"Loading mu/sigma from {mu_sigma_path}")
        with open(mu_sigma_path, "rb") as f:
            params = pickle.load(f)
        mu, sigma = params["mu"], params["sigma"]
    else:
        sum_x  = pd.Series(0.0, index=FEATURES)
        sum_x2 = pd.Series(0.0, index=FEATURES)
        count = 0
        needed_cols = ["mmsi", "date_time_utc", "report"] + BASE_FEATURES
        for f in TRAIN_FILES:
            df = pd.read_parquet(f, columns=needed_cols)
            print("mmsis in training param df before: ", df["mmsi"].nunique())
            df["mmsi"] = df["mmsi"].astype("int64")
            df = df[df["mmsi"].isin(train_mmsis)]
            df.loc[df["report"] == "Not", "report"] = "Snurrevad"
            df = df[df["report"].isin(GEARS)]
            print("mmsis in training param df after: ", df["mmsi"].nunique())
            df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
            month = df["date_time_utc"].dt.month
            df["month_sin"] = np.sin(2 * np.pi * month / 12)
            df["month_cos"] = np.cos(2 * np.pi * month / 12)
            x = df[FEATURES]
            sum_x  += x.sum()
            sum_x2 += (x ** 2).sum()
            count  += len(x)
        mu = sum_x / count
        sigma = np.sqrt((sum_x2 / count) - mu ** 2).replace(0, 1)
        
        with open(mu_sigma_path, "wb") as f:
            pickle.dump({"mu": mu, "sigma": sigma}, f)
        print(f"Fit mu/sigma on 2023 train set and saved to {mu_sigma_path}")
    
    return mu, sigma

mu, sigma = get_mu_sigma()

def iter_windows(arr, window, slide):
    n, p = arr.shape

    def with_mask(w, real_len):
        out = np.zeros((window, p + 1), dtype=np.float32)
        out[:real_len, :p] = w[:real_len]
        out[:real_len, p] = 1.0
        return out

    if n < window:
        yield with_mask(arr, n), 0, n
        return

    starts = list(range(0, n - window + 1, slide))
    if starts[-1] != n - window:
        starts.append(n - window)

    for s in starts:
        yield with_mask(arr[s:s + window], window), s, window

def build_windows(files, mmsis, slide):
    X, meta = [], []          # meta[i] describes X[i]
    for f in files:
        df = pd.read_parquet(f, engine="pyarrow")
        df = add_features(df, mmsis)
        df[FEATURES] = (df[FEATURES] - mu) / sigma
        for seg_id, d in df.groupby("segment_id", sort=False):
            d = d.sort_values("date_time_utc")
            gear  = d["report"].iloc[0]
            mmsi  = int(d["mmsi"].iloc[0])
            times = d["date_time_utc"].to_numpy()
            arr   = d[FEATURES].to_numpy(dtype=np.float32)
            for w, start, real_len in iter_windows(arr, WINDOW, slide):
                if real_len < 30: # less than 15 minutes of real data -> skip
                    continue
                X.append(w)
                meta.append({
                    "mmsi":        mmsi,
                    "segment_id":  seg_id,
                    "source_file": Path(f).name,   # segment_id is only unique *within* a file
                    "gear":        gear,
                    "start":       start,           # row offset into the sorted segment
                    "real_len":    real_len,        # non-padded length
                    "t_start":     times[start],
                    "t_end":       times[start + real_len - 1],
                })
    return np.asarray(X, np.float32), pd.DataFrame(meta)

X_train, meta_train = build_windows(TRAIN_FILES, mmsis=train_mmsis, slide=TRAIN_SLIDE)
print("X_train:", X_train.shape)
y_train = meta_train["gear"].to_numpy()
groups_train = meta_train["mmsi"].to_numpy()
print(pd.Series(y_train).value_counts())

X_val, meta_val = build_windows(VAL_TEST_FILES, mmsis=val_mmsis, slide=TRAIN_SLIDE)
print("X_val:", X_val.shape)
y_val = meta_val["gear"].to_numpy()
groups_val = meta_val["mmsi"].to_numpy()
print(pd.Series(y_val).value_counts())

X_test_unseen, meta_test_unseen = build_windows(VAL_TEST_FILES, mmsis=test_mmsis, slide=TEST_SLIDE)
print("X_test_unseen:", X_test_unseen.shape)
y_test_unseen = meta_test_unseen["gear"].to_numpy()
groups_test_unseen = meta_test_unseen["mmsi"].to_numpy()
print(pd.Series(y_test_unseen).value_counts())


random.seed(42)
train_mmsis_list = random.sample(sorted(train_mmsis), k=len(train_mmsis) // 2)
print(f"Nr of train mmsis to use for seen test: ", len(train_mmsis_list))

X_test_seen, meta_test_seen = build_windows(VAL_TEST_FILES, mmsis=train_mmsis_list, slide=TEST_SLIDE)
print("X_test_seen:", X_test_seen.shape)
y_test_seen = meta_test_seen["gear"].to_numpy()
groups_test_seen = meta_test_seen["mmsi"].to_numpy()
print(pd.Series(y_test_seen).value_counts())


np.save("datasets/X_train.npy", X_train)
np.save("datasets/y_train.npy", y_train)
np.save("datasets/groups_train.npy", groups_train)

np.save("datasets/X_val.npy", X_val)
np.save("datasets/y_val.npy", y_val)
np.save("datasets/groups_val.npy", groups_val)

np.save("datasets/X_test_unseen.npy", X_test_unseen)
np.save("datasets/y_test_unseen.npy", y_test_unseen)
np.save("datasets/groups_test_unseen.npy", groups_test_unseen)
meta_test_unseen.to_parquet("datasets/meta_test_unseen.parquet", index=False)

np.save("datasets/X_test_seen.npy", X_test_seen)
np.save("datasets/y_test_seen.npy", y_test_seen)
np.save("datasets/groups_test_seen.npy", groups_test_seen)
meta_test_seen.to_parquet("datasets/meta_test_seen.parquet", index=False)