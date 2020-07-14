import numpy as np
import pickle
import xarray as xr
import matplotlib.pyplot as plt

bins = np.arange(-0.1, 1.4, 0.005)
plt.xlim([-0.1, 2])
plt.hist(data, bins=bins, alpha=0.5)
plt.axvline(x=data.mean(), color="red", linewidth=1)
plt.axvline(x=(data.std() - data.mean()), color="green", linewidth=1)
plt.axvline(x=(data.std() + data.mean()), color="green", linewidth=1)
plt.title("MERRA Distribution in Test Dataset")
plt.xlabel("MERRA TOTEXTTAU value (bin width=0.005)")
plt.ylabel("Number of datapoints")
plt.savefig("closer_merra_distribution.png")
