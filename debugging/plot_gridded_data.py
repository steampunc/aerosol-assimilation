import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.path as paths

def plotPolygon(ax, polygon, color, linewidth=2, pt=False):
    if pt:
        ax.scatter(polygon[0], polygon[1], transform = ccrs.Geodetic(), color=color)
        return
    for i in range(1, len(polygon)):
        ax.plot([polygon[i - 1][0], polygon[i][0]],
                [polygon[i - 1][1], polygon[i][1]],
                color=color, linewidth=linewidth,
               transform = ccrs.Geodetic())

def plotData(plot_dict, metadata, axes):
    
    boundary = plot_dict["roi"][1]
    
    numLonPoints = 100j
    numLatPoints = 80j
    lon_grid, lat_grid = np.mgrid[boundary["minLon"]:boundary["maxLon"]:numLonPoints, 
                                  boundary["minLat"]:boundary["maxLat"]:numLatPoints]
    
    

    ptColor = "red"
    if inSwath(plot_dict["roi"][0], metadata[plot_dict["closest_viirs"]]["bounds"]):
        ptColor = "green"
    
    
    axes.pcolormesh(lon_grid, lat_grid, plot_dict["gridded_viirs"])
    
def inSwath(roi, swath_polygon):
    path = paths.Path(np.array(swath_polygon))
    for point in roi:
        if not path.contains_point(point):
            return False
    return True 
    
    

folder = "data/20200401/"
metadata = None
with open(folder + "metadata", "rb") as md_file:
    metadata = pickle.load(md_file)
    

fig, axes = plt.subplots(ncols=5, nrows=2, subplot_kw={'projection': ccrs.PlateCarree()})
for x in range(5):
    for y in range(2):
        files = os.listdir(folder + "processed/")
        plot_dict = None
        with open(folder + "processed/" + files[12 * x + 5 * y], "rb") as datafile:
            plot_dict = pickle.load(datafile)
        plotData(plot_dict, metadata, axes[y][x])
    
fig.set_size_inches(22, 11)
plt.show()
