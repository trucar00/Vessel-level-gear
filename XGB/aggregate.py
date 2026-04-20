import pandas as pd

GEAR = "trawl"

seg = pd.read_csv(f"XGB/segments_{GEAR}.csv")

feature_cols = [
    "mean_speed",
    "std_speed",
    "min_speed",
    "max_speed",
    "mean_acc",
    "std_acc",
    "mean_abs_acc",
    "mean_dcog",
    "std_dcog",
    "mean_abs_dcog",
    "cum_abs_turn",
    "path_length",
    "net_displacement",
    "straightness",
    "mean_dt",
    "std_dt",
]

agg_dict = {}
for col in feature_cols:
    agg_dict[col] = ["mean", "std", "min", "max", "median"]

vessel_df = seg.groupby("mmsi").agg(agg_dict)

# Flatten column names
vessel_df.columns = [
    f"{col}_{stat}" for col, stat in vessel_df.columns
]
vessel_df = vessel_df.reset_index()

seg["low_speed_seg"] = (seg["mean_speed"] < 1.5).astype(int)
seg["mid_speed_seg"] = ((seg["mean_speed"] >= 1.5) & (seg["mean_speed"] < 4)).astype(int)
seg["high_speed_seg"] = (seg["mean_speed"] >= 4).astype(int)

seg["high_turn_seg"] = (seg["mean_abs_dcog"] > seg["mean_abs_dcog"].median()).astype(int)
seg["straight_seg"] = (seg["straightness"] > 0.8).astype(int)

prop_df = seg.groupby("mmsi").agg({
    "low_speed_seg": "mean",
    "mid_speed_seg": "mean",
    "high_speed_seg": "mean",
    "high_turn_seg": "mean",
    "straight_seg": "mean",
    "segment_id": "count"
}).rename(columns={"segment_id": "n_segments"}).reset_index()

vessel_df = vessel_df.merge(prop_df, on="mmsi", how="left")
vessel_df["gear"] = GEAR

print(vessel_df.head())
print(vessel_df.shape)