import pandas as pd
import matplotlib.pyplot as plt

checked = pd.read_csv("checked.csv")

ais = pd.read_csv("test_segment_messages_line.csv")


df = ais.merge(checked, on="segment_id", how="left")

print(df.head())

color_map = {
    0: "red",
    1: "blue",
    2: "green"
}

plt.figure(figsize=(10, 8))

for seg_id, d in df.groupby("segment_id"):
    pred = d["pred"].iloc[0]   # same for entire segment
    color = color_map.get(pred, "black")

    plt.plot(d["lon"], d["lat"], color=color, alpha=0.7)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("AIS Segments colored by prediction")
plt.grid(True)

plt.show()