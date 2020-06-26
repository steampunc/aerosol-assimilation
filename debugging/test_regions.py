import pickle
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

coords = None
merra = xr.open_dataset("data/VIIRS-20200320/merra.nc")

with open("coords", "rb") as coordFile:
    coords = pickle.load(coordFile)

fig, axes = plt.subplots(ncols=1, nrows=1)
axes.pcolormesh(merra.TOTEXTTAU[0])
for coord in coords:
    [x, y] = map(int, coord.split("_"))
    merra.TOTEXTTAU[0][y][x] = 100
    axes.scatter(x, y, color="red")

plt.show()
