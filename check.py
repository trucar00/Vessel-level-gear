import pandas as pd
import matplotlib.pyplot as plt

def remove_vessels_few_days(df, days=5):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    span = df.groupby("mmsi")["date_time_utc"].agg(["min", "max"])

    span["duration_days"] = (span["max"] - span["min"]).dt.total_seconds() / (3600 * 24)
    valid_mmsi = span[span["duration_days"] >= days].index

    # Filter original dataframe
    df_filtered = df[df["mmsi"].isin(valid_mmsi)]
    return df_filtered

trawl = pd.read_csv("Data/trawl_jan_2024.csv")
trawl = remove_vessels_few_days(trawl)

line = pd.read_csv("Data/line_jan_2024.csv")
line = remove_vessels_few_days(line)

purse = pd.read_csv("Data/not_jan_2024.csv")
purse = remove_vessels_few_days(purse)

#print(f"trawler {trawl["mmsi"].nunique()} liner: {line["mmsi"].nunique()} purse: {purse["mmsi"].nunique()}")

for mmsi, d in trawl.groupby("mmsi"):

    print(d.head())

    fig, ax = plt.subplots(figsize=(10,8))
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values(by="date_time_utc")
    start = d["date_time_utc"].iloc[0]
    end = d["date_time_utc"].iloc[-1]
    for traj_id, dd in d.groupby("trajectory_id"):
        ax.plot(dd["lon"], dd["lat"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Vessel {mmsi} between {start} and {end}")
    ax.legend()
    plt.show()