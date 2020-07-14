#!/bin/sh
#
# This is the number of cores requested (_not the number of compute nodes_)
#SBATCH --ntasks=40
#SBATCH --nodes=1
#
# Specifying time for job, in this case 10 minutes
#SBATCH --time=3:00:00
#
# I looked for a while but couldn't figure out what account I should put, either
# for Orion or Hera. This is where the run is charged to, so I figure it's 
# important to get it right.
#SBATCH --account=da-cpu
#
# Setting the name of the job
#SBATCH --job-name precondition_aods
#
# Considering the current dir as the working directory
#SBATCH --chdir=.
#
# I should probably select the lowest priority queue, which on Hera is
# windfall(?). Is this better practice than using the default batch priority?
#SBATCH --qos batch
#
# I'm selecting the orion partition rather than other ones
#SBATCH --partition=orion

# Load some relevant modules with module load
module purge
module use -a /work/noaa/marine/marineda/modulefiles
module load anaconda/1.7.2
module list

# Activating the venv to resolve dependencies. This is kinda sketch but it works
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/noaa/marine/Stylianos.Flampouris/save/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/noaa/marine/Stylianos.Flampouris/save/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/work/noaa/marine/Stylianos.Flampouris/save/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/noaa/marine/Stylianos.Flampouris/save/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate fboire

# I could do some other prep stuff here, cp'ing data, etc

# I see that I need srun to launch something using MPI, but I don't know if that
# applies to python's multiprocessing library. I could just try running it both 
# with and without srun to see if the process fails (assuming that would do no 
# lasting damage), but let me know if you know already that slurm expects/needs 
# it to be a certain way.
echo "Working in dir: "
echo $1
echo "\nStarting python program"
python multi_conditioning.py $1
