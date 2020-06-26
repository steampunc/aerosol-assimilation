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
    
    
    plotPolygon(axes, plot_dict["roi"][0], ptColor, linewidth=1)
    plotPolygon(axes, [plot_dict["roi"][0][1], metadata[plot_dict["closest_viirs"]]["center"]], "yellow", linewidth=1)
    plotPolygon(axes, metadata[plot_dict["closest_viirs"]]["bounds"], "red", linewidth=1)
    plotPolygon(axes, metadata[plot_dict["closest_viirs"]]["center"], "blue", linewidth=10, pt=True)
    
def inSwath(roi, swath_polygon):
    path = paths.Path(np.array(swath_polygon))
    for point in roi:
        if not path.contains_point(point):
            return False
    return True 
    
    

folder = "data/20200329/"
metadata = None
with open(folder + "metadata", "rb") as md_file:
    metadata = pickle.load(md_file)
    

fig, axes = plt.subplots(ncols=1, nrows=1, subplot_kw={'projection': ccrs.PlateCarree()})
axes.coastlines()
i = 0
for file in os.listdir(folder + "processed/"):
    i = i + 1
    print(i)
    if i > 500:
        break
    plot_dict = None
    with open(folder + "processed/" + file, "rb") as datafile:
        plot_dict = pickle.load(datafile)
    print(file)
    plotData(plot_dict, metadata, axes)
    
fig.set_size_inches(22, 11)
plt.show()
