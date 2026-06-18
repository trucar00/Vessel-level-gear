import pandas as pd
import numpy as np
from tqdm import tqdm

WINDOW = 120
SLIDE  = 60

# CREATE tr

GEARS = ["Trål", "Krokredskap", "Not", "Snurrevad", "Garn"]

BASE_FEATURES   = ["cog_interp_sin", "cog_interp_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES

FILES = ["2024_1_3_feats.parquet"]   # add the other quarters here


def add_features(df):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df.loc[df['report'] == 'Not', 'report'] = 'Snurrevad'
    df = df[df["report"].isin(GEARS)]
    m = df["date_time_utc"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    return df

def iter_windows(arr, window, slide):
    n = len(arr)
    if n < window:                                   # pad short bouts (keeps rare gears)
        pad = np.zeros((window - n, arr.shape[1]), dtype=np.float32)
        yield np.vstack([arr, pad])
        return
    starts = list(range(0, n - window + 1, slide))
    if starts[-1] != n - window:                     # capture the tail
        starts.append(n - window)
    for s in starts:
        yield arr[s:s + window]

def build_windows(files):
    X, y, groups = [], [], []
    for f in files:
        df = add_features(pd.read_parquet(f, engine="pyarrow"))
        for seg_id, d in tqdm(df.groupby("segment_id", sort=False)):
            d = d.sort_values("date_time_utc")
            gear = d["report"].iloc[0]          # single gear per segment
            mmsi = d["mmsi"].iloc[0]
            arr = d[FEATURES].to_numpy(dtype=np.float32)
            for w in iter_windows(arr, WINDOW, SLIDE):
                X.append(w)
                y.append(gear)
                groups.append(mmsi)
    return (np.asarray(X, np.float32),
            np.asarray(y),
            np.asarray(groups))

X, y, groups = build_windows(FILES)
print("X:", X.shape)
print(pd.Series(y).value_counts())

np.save("datasets/X_gear.npy", X)
np.save("datasets/y_gear.npy", y)
np.save("datasets/groups_gear.npy", groups)