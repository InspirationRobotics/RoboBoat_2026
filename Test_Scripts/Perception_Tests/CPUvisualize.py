"""
This script visualizes CPU usage from cpu_usage_log.csv
"""

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg" if you have PyQt installed
import matplotlib.pyplot as plt

# Load the CSV file
LOG_FILE = "Test_Scripts/Perception_Tests/cpu_usage_log.csv"
df = pd.read_csv(LOG_FILE)

# Convert timestamp to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Create a figure with two subplots (stacked vertically)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

# Plot CPU Usage on the first subplot
ax[0].plot(df["Timestamp"], df["CPU Usage (%)"], label="CPU Usage (%)", color="blue", linestyle="-", marker="o")
ax[0].set_ylabel("CPU Usage (%)")
ax[0].set_title("CPU Usage Over Time")
ax[0].legend()
ax[0].grid(True)

# Plot GPS & Camera Response Times on the second subplot
ax[1].plot(df["Timestamp"], df["GPS Response Time (ms)"], label="GPS Response Time (ms)", color="red", linestyle="--", marker="s")
ax[1].plot(df["Timestamp"], df["Camera Response Time (ms)"], label="Camera Response Time (ms)", color="green", linestyle="--", marker="d")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Response Time (ms)")
ax[1].set_title("GPS & Camera Response Times Over Time")
ax[1].legend()
ax[1].grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show both plots in the same figure
plt.show()
