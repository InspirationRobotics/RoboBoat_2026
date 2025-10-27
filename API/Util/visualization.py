import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Define the points
points = {
    "A": (27.373150999552347, -82.45271944999526),
    "B0": (27.373150998894577, -82.4529219928497),
    "B1": (27.37326299868445, -82.45299162917827),
    "B2": (27.373445998456045, -82.45309426603173),
    "C1": (27.373457998684437, -82.453031630249),
    "C2": (27.373339998456053, -82.45289826540117),
    "C3": (27.373150999086427, -82.45287135713609),
    "C4": (27.373258998894574, -82.45296699339329),
    "D1": (27.37336599991778, -82.45256890743596),
    "D2": (27.373433999853823, -82.45262754337242),
    "E1": (27.37330099991778, -82.45256490734674),
    "E2": (27.37325099991778, -82.4525599072781),
}



# Create the map
plt.figure(figsize=(10, 8))
m = Basemap(projection='mill',
            llcrnrlon=-82.454, llcrnrlat=27.372,
            urcrnrlon=-82.451, urcrnrlat=27.374,
            resolution='h')
m.drawcoastlines()
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgray', lake_color='aqua')

# Plot the points
for label, (lat, lon) in points.items():
    x, y = m(lon, lat)
    plt.plot(x, y, marker='o', markersize=5, label=label)
    plt.text(x, y, label, fontsize=12, ha='right')

plt.title("Visualization of Given Points")
plt.legend()
plt.show()
