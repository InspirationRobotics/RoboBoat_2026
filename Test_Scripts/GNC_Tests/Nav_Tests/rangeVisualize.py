import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("/home/chaser/Downloads/OAK_D LR depth test.xlsx")
print(df.head())

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), sharex=True)

# Plot data on the first subplot
ax.plot(df["Actual distance (m)"], df["Error"], label="Error", color="blue", linestyle="-", marker="o")
ax.set_xlabel("distance (m)")
ax.set_ylabel("Error (%)")
ax.set_title("OAK_D LR Range")
ax.legend()
ax.grid(True)

plt.show()