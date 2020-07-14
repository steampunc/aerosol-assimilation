import pickle
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

cnn_mask_vals = xr.open_dataset("data/processed/full_viirs.nc")
cnn_mean_deviation = xr.open_dataset("data/processed/mean_viirs.nc")
viirs_means = xr.open_dataset("data/processed/pure_mean.nc")
merra_vals = xr.open_dataset("data/processed/merra.nc")

def generateHistogram(data, figname):
    print(data.values.shape)
    data = data.values.reshape(-1)
    data = data[~np.isnan(data)]
    bins = np.arange(-0.1, 1.4, 0.01)
    plt.xlim([-0.1, 2])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.axvline(x=data.mean(), color="red", linewidth=1)
    plt.axvline(x=(data.std() - data.mean()), color="green", linewidth=1)
    plt.axvline(x=(data.std() + data.mean()), color="green", linewidth=1)
    plt.title(figname + " Mean: {:.4f}, Std: {:.4f}".format(data.mean(), data.std()))
    plt.xlabel("datapoint value (bin width=0.005)")
    plt.ylabel("Number of datapoints")
    plt.savefig(figname)
    plt.close()

generateHistogram(cnn_mask_vals.NNPROCESS, "cnn_mask_hist.png")
generateHistogram(cnn_mean_deviation.NNPROCESS, "mean_dev_hist.png")
generateHistogram(viirs_means.NNPROCESS, "viirs_mean_hist.png")
generateHistogram(merra_vals.NNPROCESS, "merra_value_hist.png")


"""
fig, axes = plt.subplots(ncols=1, nrows=4, subplot_kw={'projection': ccrs.PlateCarree()})
for t in range(24):
    axes[0].pcolormesh(cnn_mask_vals.NNPROCESS[t].lon, cnn_mask_vals.NNPROCESS[t].lat, cnn_mask_vals.NNPROCESS[t], vmin=0, vmax=0.5)
    axes[0].set_title("Network architecture 1 - CNN with mask") 
    axes[1].pcolormesh(cnn_mean_deviation.NNPROCESS[t].lon, cnn_mean_deviation.NNPROCESS[t].lat, cnn_mean_deviation.NNPROCESS[t], vmin=0, vmax=0.5)
    axes[1].set_title("Network architecture 2 - CNN mean deviation")
    axes[2].pcolormesh(viirs_means.NNPROCESS[t].lon, viirs_means.NNPROCESS[t].lat, viirs_means.NNPROCESS[t], vmin=0, vmax=0.5)
    axes[2].set_title("Mean of VIIRS values across window")
    axes[3].pcolormesh(merra_vals.NNPROCESS[t].lon, merra_vals.NNPROCESS[t].lat, merra_vals.NNPROCESS[t], vmin=0, vmax=0.5)
    axes[3].set_title("Pure MERRA values")

for i in range(len(axes)):
    axes[i].set_xlim([-28.6151967, 73.0259083])
    axes[i].set_ylim([7.3599682,36.5153833])
    axes[i].coastlines()
plt.show()
"""
