import xarray as xr
import multiprocessing as mp
import numpy as np
import os
import random

def grid_array(x_min, y_min, x_max, y_max, shape):
    result = []
    x_step = int((x_max - x_min) / shape[1])
    y_step = int((y_max - y_min) / shape[0])
    for xi in range(0, shape[1]):
        for yi in range(0, shape[0]):
            result.append({"x_min":x_min + xi * x_step, "x_max":x_min + (xi + 1) * x_step, "y_min":y_min + yi * y_step, "y_max":y_min + (yi + 1) * y_step, "shape":(y_step, x_step), "i":len(result)})
    return result

nrandomsamples = 50

def stoch_avg(data, x, y):
    val = data[y][x] + data[y - 1][x] + data[y + 1][x] + data[y][x - 1] + data[y][x + 1]
    val += data[y + 1][x + 1] + data[y + 1][x - 1] + data[y - 1][x + 1] + data[y - 1][x - 1]
    val = 2 * val
    for i in range(nrandomsamples):
        delx = random.randint(-20, 20)
        dely = random.randint(-20, 20)
        val += data[y + dely][x + delx]
    return val / (18 + nrandomsamples)

def sharpen(data, x, y):
    val = 5 * data[y][x] - (data[y - 1][x] + data[y + 1][x] + data[y][x - 1] + data[y][x + 1])
    return val

def edge_filter(data, x, y):
    # Applying sobel operator to look for edges. It's basically just a 1st order finite difference method for computing a gradient
    gx = data[y - 1][x - 1] + 2 * data[y - 1][x] + data[y - 1][x + 1] - (data[y + 1][x - 1] + 2 * data[y + 1][x] + data[y + 1][x + 1])
    gy = data[y - 1][x - 1] + 2 * data[y][x - 1] + data[y + 1][x - 1] - (data[y - 1][x + 1] + 2 * data[y][x + 1] + data[y + 1][x + 1])
    return 2 * (gx * gx + gy * gy)

print("Number of CPUs: {}".format(mp.cpu_count()))
print("Opening dataset")
dataset = xr.open_dataset("aod_npp.nc")
print("Opened")

aods = dataset.AOD550[:]
aods = aods.fillna(0)

avg_result = xr.zeros_like(aods)
edge_result = xr.zeros_like(aods)

def apply_to_region(region, func, data):
    temp = np.zeros(region["shape"])
    for y in range(0, region["shape"][0]):
        for x in range(0, region["shape"][1]):
            temp[y][x] = func(data.values, x + region["x_min"], y + region["y_min"])
        print("Processing y = {} with PID {}".format(y, mp.current_process().name))
    return temp

def apply_avg(region):
    return apply_to_region(region, stoch_avg, aods)

nprocesses = 8 #os.environ["SLURM_NTASKS")
process_shape = (4, 2)
process_grid = []
if not process_shape[0] * process_shape[1] == nprocesses:
    print("ERROR: Process gridding not equal to number of processes. Please reshape your grids.")
    exit()
else:
    y_min = int(aods.shape[0] * 0.5)
    y_max = int(aods.shape[0] * 0.9)
    x_min = int(aods.shape[1] * 0.5)
    x_max = int(aods.shape[1] * 0.9)
    process_grid = grid_array(x_min, y_min, x_max, y_max, process_shape)

with mp.Pool(processes=nprocesses) as pool:
    computed_results = pool.map(apply_avg, process_grid)

    for i in range(nprocesses):
        grid = process_grid[i]
        avg_result[grid["y_min"]:grid["y_max"], grid["x_min"]:grid["x_max"]] = computed_results[i]

dataset["avg_AOD550"] = avg_result

print(avg_result.max(), avg_result.min())

def apply_edges(region):
    return apply_to_region(region, edge_filter, avg_result)

def apply_sharpen(region):
    return apply_to_region(region, sharpen, aods)

with mp.Pool(processes=nprocesses) as pool:
    computed_results = pool.map(apply_edges, process_grid)

    for i in range(nprocesses):
        grid = process_grid[i]
        edge_result[grid["y_min"]:grid["y_max"], grid["x_min"]:grid["x_max"]] = computed_results[i]

dataset["edges_AOD550"] = edge_result

dataset.to_netcdf("processed.nc")
