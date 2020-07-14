# Generates side-by-side images of the interpolated windows along with their NaN masks.

import torch
import os
import torch.multiprocessing as mp
import pickle
from torchvision import transforms
import numpy as np
from aoddataset import AODDataset 
import matplotlib.pyplot as plt

def log(message):
    verbose = True 
    if verbose:
        print(message)
    else:
        with open("logs/" + str(os.getpid()) + ".log", "a") as logfile:
            logfile.write(str(message) + "\n")


dataset = None
if os.path.isfile("data/mask_test"):
    with open("data/mask_test", "rb") as test_pickle:
        log("Opening test_pickle")
        dataset = pickle.load(test_pickle)
        log("Done")

log("dataset size: {}".format(len(dataset)))

fig, axes = plt.subplots(ncols=4, nrows=3)
for x in range(4):
    for y in range(3):
        if x % 2 == 1:
            continue
        img1 = transforms.ToPILImage()(dataset[x * 4 + y][0][0].squeeze(0).detach().cpu())
        img2 = transforms.ToPILImage()(dataset[x * 4 + y][0][1].squeeze(0).detach().cpu().float())
        axes[y][x + 0].imshow(img1)
        axes[y][x + 1].imshow(img2)
plt.show()
