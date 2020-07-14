import os
import tarfile
import sys

directory = sys.argv[1]

for tar_f in os.listdir(directory):
    if (tar_f == "merra.nc" or tar_f == "processed" or tar_f == "metadata"):
        continue
    print("Extracting {}".format(tar_f))
    tar = tarfile.open(directory + tar_f)
    tar.extractall(path=directory)
    tar.close()

    os.remove(directory + tar_f)
