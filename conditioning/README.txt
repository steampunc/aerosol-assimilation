This is the "suite" of tools to collect and preprocess VIIRS/MERRA data to be passed along to the neural network model for assimilation. The general workflow:

Select a date with both VIIRS and MERRA data available.
Download MERRA.nc file locally
Run get_VIIRS.sh path/to/ftp/server/tar/files destination/directory/
Unpack data with `python unpack_data.py destination/directory/`
Scp MERRA nc file to Orion in the same destination/directory/
Submit batch job for multithreaded processing program:
sbatch queue_job.sh destination/directory/

