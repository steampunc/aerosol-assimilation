import torch
import pickle
import os
import numpy as np

def log(message):
    verbose = False
    if verbose:
        print(message)
    else:
        with open(str(os.getpid()) + ".log", "a") as logfile:
            logfile.write("\n" + str(message))

class AODDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        super().__init__()
        self.samples = []
        self.coordinates = []
        self.times = []
        datapoints = os.listdir(folder)
        l_dp = len(datapoints)
        for i in range(len(datapoints)):
            if (i % 100 == 0):
                log(float(i) / float(l_dp))
            with open(folder + datapoints[i], "rb") as picklefile:
                data = pickle.load(picklefile)
                # Formatting and normalizing the data
                data["gridded_viirs"] = np.nan_to_num(np.exp(data["gridded_viirs"])) / 149.0
                self.samples.append((torch.from_numpy(data["gridded_viirs"]), torch.from_numpy(np.array(data["merra_val"]))))
                self.coordinates.append(datapoints[i])
                self.times.append(data["time"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
    def getCoord(self, index):
        return self.coordinates[index]
        
