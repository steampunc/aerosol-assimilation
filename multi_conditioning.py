import os
import tarfile
import xarray as xr
import numpy as np
import pickle
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.path as paths
import cartopy.crs as ccrs
import multiprocessing as mp
import sys

def log(message):
    verbose = False
    if verbose:
        print(message)
    else:
        with open("logs/" + mp.current_process().name + ".log", "a") as logfile:
            logfile.write("\n" + str(message))

## Extracts boundary metadata from a particular netcdf file
def extractBounds(filename):
    dataset = xr.open_dataset(filename)
    pp = dataset.geospatial_bounds[9:-2].split(",")
    polygon = []
    center = np.array([0, 0])
    for point in pp:
        pt = np.array(list(map(float, point.split(" "))))
        polygon.append(pt)
        center = center + pt
    np_poly = np.array(polygon)
    
    return {"bounds": np_poly, "btime": dataset.time_coverage_start,
            "etime": dataset.time_coverage_end, "center": center / len(polygon)}

## Gets the boundary metadata of all VIIRS files within a directory
def getVIIRSMetadata(folder):
    metadata = {}
    tar_files = os.listdir(folder)
    
    for tar_f in tar_files:
            
        if (tar_f == "metadata" or tar_f == "merra.nc" or tar_f == "processed"):
            continue
        log("Extracting " + tar_f)
        tar = tarfile.open(folder + tar_f)
        tar.extractall(path="data/curr_data")
        tar.close()
            
        files = os.listdir("data/curr_data")
        for f in files:
            log("Opening " + f)
            
            ## THIS IS WHERE YOU CAN APPLY SOME SORT OF PROCESSING STEP TO ALL THE DATA
            metadata[tar_f + "/" + f] = extractBounds("data/curr_data/" + f)
        
            log("Deleting " + f)
            os.remove("data/curr_data/" + f)
        log("Done extracting " + tar_f)
        
    log("Done extracting")

    with open(folder + "metadata", "wb") as handle:
        pickle.dump(metadata, handle)
        
    return metadata

## Gets the VIIRS file closest to a given point
def getClosestVIIRSSwath(point, metadata):
        m_files = np.array(list(metadata.keys()))
        m_vals = np.array(list(metadata.values()))
        c_points = []
        for vals in m_vals:
            c_points.append(vals["center"])
        # Gets closest point
        c_array = np.asarray(c_points)
        dists = np.sum((c_array - point)**2, axis=1)
        return [m_files[np.argmin(dists)], np.min(dists)]

## Extracts and opens the VIIRS data corresponding to the full filename/path
def getVIIRSData(folder, filename, destination):
    levels = filename.split("/")
    
    log("Extracting {} to {}".format(levels[0], destination))
    tar = tarfile.open(folder + levels[0])
    tar.extractall(path=destination)
    tar.close()
    
    log("Reading " + levels[1])
    data = xr.open_dataset(destination + levels[1])
    
    for f in os.listdir(destination):
        os.remove(destination + f)
    
    return data

## Converts VIIRS data to an array for linear interpolation.
def getVIIRSArrays(viirs_data):
    sample_points = []
    aod_array = []
    
    aods = viirs_data.AOD550
    aod_array = aods.values.reshape(1, -1)[0]
    lons = aods.Longitude.values.reshape(1, -1)[0]
    lats = aods.Latitude.values.reshape(1, -1)[0]
    sample_points = np.column_stack((lons, lats))
    return [sample_points, aod_array]

## Creates a ROI for the particular MERRA datapoint
def getMERRAROI(lon, lat):
    d_lon = 0.625
    d_lat = 0.5
    polygon = []
    bounds = {"minLon": lon - d_lon, "maxLon": lon + d_lon,
              "minLat": lat - d_lat, "maxLat": lat + d_lat}
    polygon.append(np.array([bounds["minLon"], bounds["minLat"]]))
    polygon.append(np.array([bounds["minLon"], bounds["maxLat"]]))
    polygon.append(np.array([bounds["maxLon"], bounds["maxLat"]]))
    polygon.append(np.array([bounds["maxLon"], bounds["minLat"]]))
    polygon.append(np.array([bounds["minLon"], bounds["minLat"]]))
    return [np.array(polygon), bounds]

        
def inSwath(roi, swath_polygon):
    path = paths.Path(np.array(swath_polygon))
    for point in roi:
        if not path.contains_point(point):
            return False
    return True

## Performs nearest neighbor linear interpolation on the VIIRS data array.
def VIIRStoGrid(s_points, data_array, boundary):
    numLonPoints = 100j
    numLatPoints = 80j
    
    lon_grid, lat_grid = np.mgrid[boundary["minLon"]:boundary["maxLon"]:numLonPoints, 
                                  boundary["minLat"]:boundary["maxLat"]:numLatPoints]
    
    gridded_data = griddata(s_points, data_array, (lon_grid, lat_grid), method="nearest")
    return gridded_data

def getT(m_data, swath_name):
    time = list(map(int, m_data[swath_name]["btime"].split("T")[1][:-1].split(":")))
    if time[1] > 30:
        return min(time[0] + 1, 23)
    else:
        return time[0]


def process(folder, bounds):
    mp.current_process().name = folder.split("/")[1] + mp.current_process().name
    log("Starting process {}".format(mp.current_process().name))
    metadata = None
    if "metadata" in os.listdir(folder):
        with open(folder + "metadata", "rb") as md_file:
            metadata = pickle.load(md_file)
    else:
        metadata = getVIIRSMetadata(folder)
        
    merra_data = xr.open_dataset(folder + "merra.nc")
    active_dir = "temp_dirs/ " + mp.current_process().name + "/"
    if not os.path.exists(active_dir):
        os.makedirs(active_dir)

    prev_swath = ""
    viirs_data = None
    viirs_arrays = None
    num_saved = 0
    lat_diff = float(bounds["maxLat"] - bounds["minLat"])
    for y in range(bounds["minLat"], bounds["maxLat"]):
        log("Processing new lat level for {}".format(mp.current_process().name))
        log("Percent complete: {}".format(float(y - bounds["minLat"]) / lat_diff))
        for x in range(bounds["minLon"], bounds["maxLon"]):
            good_files = os.listdir(folder + "processed/")
            bad_files = os.listdir(folder + "bad_points/")
            curr_name = "{:04d}_{:04d}".format(x, y) 
            if curr_name in good_files or curr_name in bad_files:
                log("Already considered point ({}, {})".format(x, y))
                continue

            # Getting lat/lon of merra gridpoint
            point = [merra_data.lon.values[x], merra_data.lat.values[y]]
            log("\nConsidering merra point at " + str(point) + "\nIndex " + str(x) + ", " + str(y))
            
            # Getting corresponding VIIRs box
            [closest_viirs, distance] = getClosestVIIRSSwath(point, metadata)
            
            # Skipping point if too far from center of swath
            log(distance)
            if (distance > 44):
                log("Too far from center of VIIRS data.")
                continue
                
            
            # Gets the size of the box around the MERRA point
            log("Getting ROI")
            roi = getMERRAROI(point[0], point[1])
            log("Closest VIIR swath: " + closest_viirs)

            # Selecting correct time for observation
            time = getT(metadata, closest_viirs)
            merra_val = merra_data.TOTEXTTAU.values[time][y][x]
            log("MERRA value: " + str(merra_val) + " taken at time " + str(time) + "hrs")
            
            
            if not inSwath(roi[0][:-1], metadata[closest_viirs]["bounds"][:-1]):
                log("Not in satellite swath, so discarding")
                continue
            
            # Updating satellite data if the swath has changed.
            if (prev_swath != closest_viirs):
                log("Retrieving VIIRS data from closest swath.")
                viirs_data = getVIIRSData(folder, closest_viirs, active_dir)
                viirs_arrays = getVIIRSArrays(viirs_data)
            prev_swath = closest_viirs
            
            # Interpolates VIIRS data to 100x80 grid. This is the slowest part.
            log("Gridding viirs")
            gridded_viirs = VIIRStoGrid(viirs_arrays[0], viirs_arrays[1], roi[1])
            log("Counting number of points")
            num_points = np.count_nonzero(~np.isnan(gridded_viirs))
            log("Number of points: " + str(num_points))
            if (num_points < 30):
                log("Discarding datapoint")
                with open(folder + "bad_points/{:04d}_{:04d}".format(x, y), "w") as datafile:
                    datafile.write("bad")
                
                continue
                
            log("Saving data")
            # Saving the VIIRS data to a pickle file
            with open(folder + "processed/{:04d}_{:04d}".format(x, y), "wb") as datafile:
                pickle.dump({"gridded_viirs":gridded_viirs, "roi": roi,
                             "merra_val":merra_val, "closest_viirs":closest_viirs,
                             "curr_process":mp.current_process().name}, datafile)
            num_saved = num_saved + 1
    log("Done processing region: {}".format(bounds))
    log("Saved {} points".format(num_saved))
                
def divideProcess(x_min, y_min, x_max, y_max, shape):
    result = []
    x_step = int((x_max - x_min) / shape[1])
    y_step = int((y_max - y_min) / shape[0])
    for xi in range(0, shape[1]):
        for yi in range(0, shape[0]):
            result.append({"minLon":x_min + xi * x_step, "maxLon":x_min + (xi + 1) * x_step, "minLat":y_min + yi * y_step, "maxLat":y_min + (yi + 1) * y_step, "shape":(y_step, x_step), "i":len(result)})
    return result

def applyProcess(bounds):
    process(folder, bounds)

log("Number of CPUs: {}".format(mp.cpu_count()))

# MERRA SHAPE: 361 lat by 576 lon
nprocesses = 32
folder = sys.argv[1]
process_bounds = divideProcess(20, 70, 556, 300, (nprocesses, 1))
log("Divided process to following bounds:\n{}".format(process_bounds))
log("Number of bounds: {}".format(len(process_bounds)))

log("Prepping metadata")
if "metadata" in os.listdir(folder):
    with open(folder + "metadata", "rb") as md_file:
        metadata = pickle.load(md_file)
else:
    metadata = getVIIRSMetadata(folder)
log("Done prepping metadata\n\n")

if not os.path.exists(folder + "processed"):
    os.makedirs(folder + "processed")
if not os.path.exists(folder + "bad_points"):
    os.makedirs(folder + "bad_points")
with mp.Pool(processes=nprocesses) as pool:
    pool.map(applyProcess, process_bounds)
