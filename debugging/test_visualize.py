import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.shapereader as shprdr
import xarray as xr

data = xr.open_dataset("../data/filtered_aod_data.nc")

print(data.geospatial_bounds[9:-2].split(","))
fig, axes = plt.subplots(ncols=1, nrows=1, subplot_kw={'projection': ccrs.PlateCarree()})

axes.scatter(data.Longitude, data.Latitude, c=np.isnan(data.AOD550.values), transform=ccrs.PlateCarree(), cmap="viridis")

axes.coastlines()

plt.show()
