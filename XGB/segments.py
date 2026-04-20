import pandas as pd
import numpy as np
from tqdm import tqdm

# -- HELPER FUNCTIONS --
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c

    return dist

def angle_wrap(a):
    return (a + 180) % 360 - 180
# ---------------------------

# READY FOR CREATING SEGMENTS

def remove_vessels_few_days(df, days=5):
    span = df.groupby("mmsi")["date_time_utc"].agg(["min", "max"])

    span["duration_days"] = (span["max"] - span["min"]).dt.total_seconds() / (3600 * 24)
    valid_mmsi = span[span["duration_days"] >= days].index

    # Filter original dataframe
    df_filtered = df[df["mmsi"].isin(valid_mmsi)]
    return df_filtered


GEAR = "line"

df = pd.read_csv(f"Data/{GEAR}_jan_2024.csv")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df = remove_vessels_few_days(df)
df["traj_num"] = df["trajectory_id"].astype(str).str.rsplit("-", n=1).str[-1].astype(int)

df = df.sort_values(["mmsi", "traj_num", "date_time_utc"])

window = pd.Timedelta(hours=4)
slide = pd.Timedelta(hours=2)

all_segments = []

for traj_id, d in tqdm(df.groupby("trajectory_id", sort=False)):
    d = d.sort_values("date_time_utc")
    d["dt"] = d["date_time_utc"].diff().dt.total_seconds()
    # Generate features on trajectory level

    # Distance between consecutive points
    lon = d["lon"].values
    lat = d["lat"].values
    
    dist = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dist = np.insert(dist, 0, np.nan)
    d["dist_to_prev"] = dist
    
    # Speed
    d["speed_calc_ms"] = d["dist_to_prev"] / d["dt"]

    # Acceleration
    d["accel"] = d["speed_calc_ms"].diff() / d["dt"]

    # Jerk
    d["jerk"] = d["accel"].diff() / d["dt"]

    # Derivative of course
    d["dcog"] = d["cog"].diff().apply(angle_wrap) / d["dt"]

    feature_cols = ["dt", "dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog"]
    d = d.dropna(subset=feature_cols).copy()
    if d.empty:
        continue

    start = d["date_time_utc"].iloc[0]
    end = start + window
    segment_id = 0
    while end < d["date_time_utc"].iloc[-1]:
        segment = d[
            (d["date_time_utc"] >= start) &
            (d["date_time_utc"] < end)
        ].copy()

        if len(segment) == 0:
            start += slide
            end += slide
            continue

        segment = segment.sort_values(by="date_time_utc")
        segment["segment_id"] = segment_id

        path_length = segment["dist_to_prev"].sum()
        lat_s = segment["lat"].values
        lon_s = segment["lon"].values
        net_disp = haversine(lat_s[0], lon_s[0], lat_s[-1], lon_s[-1])

        all_segments.append({
            "mmsi": segment["mmsi"].iloc[0],
            "gear": GEAR,
            "trajectory_id": traj_id,
            "segment_id": segment_id,
            "mean_dt": segment["dt"].mean(),
            "std_dt": segment["dt"].std(),

            "mean_speed": segment["speed_calc_ms"].mean(),
            "std_speed": segment["speed_calc_ms"].std(),
            "min_speed": segment["speed_calc_ms"].min(),
            "max_speed": segment["speed_calc_ms"].max(),

            "mean_acc": segment["accel"].mean(),
            "std_acc": segment["accel"].std(),
            "mean_abs_acc": segment["accel"].abs().mean(),

            "mean_dcog": segment["dcog"].mean(),
            "std_dcog": segment["dcog"].std(),
            "mean_abs_dcog": segment["dcog"].abs().mean(),
            "cum_abs_turn": segment["cog"].diff().apply(angle_wrap).abs().sum(),

            "path_length": path_length,
            "net_displacement": net_disp,
            "straightness": net_disp / path_length if path_length > 0 else 0,

            "start_lon": segment["lon"].iloc[0],
            "start_lat": segment["lat"].iloc[0],
            "end_lon": segment["lon"].iloc[-1],
            "end_lat": segment["lat"].iloc[-1],
        })

        segment_id += 1
        start += slide
        end += slide

df_segments = pd.DataFrame(all_segments)

print(df_segments.shape) # fishing + steaming segments * 11 
print(df_segments.head())
df_segments.to_csv(f"segments_{GEAR}.csv", index=False)