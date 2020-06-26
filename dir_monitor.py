import os
import time

time_counter = 0.0
directory = "data/VIIRS-20200320/threaded_processed/"
init_data = float(len(os.listdir(directory)))
while True:
    time.sleep(10)
    time_counter = time_counter + 10.0
    curr_data = float(len(os.listdir(directory)))
    print("Running for {} seconds with {} datapoints. Averaging {} points per min.".format(time_counter, curr_data, (curr_data - init_data) * 60.0 / time_counter))
