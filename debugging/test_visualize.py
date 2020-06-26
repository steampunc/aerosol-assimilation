import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.shapereader as shprdr
import xarray as xr

data = xr.open_dataset("data/filtered_aod_data.nc")

print(data.geospatial_bounds[9:-2].split(","))
fig, axes = plt.subplots(ncols=4, nrows=2, subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(4):
    for j in range(2):
        axes[j][i].contourf(data.lon, data.lat, data.AODANA.values[i + 4 * j], transform=ccrs.PlateCarree(), cmap="viridis")

        axes[j][i].coastlines()

plt.show()
