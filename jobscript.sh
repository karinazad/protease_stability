#!/bin/bash
#SBATCH -A p30802               # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 00:30:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=18G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=6     # Number of Cores (Processors)
#SBATCH --mail-user=karinazadorozhnny2022@u.northwestern.edu  # Designate email address for job communications
#SBATCH --mail-type=BEGIN     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=/projects/30802/protease_stability>    # Path for output must already exist
#SBATCH --error=/projects/30802/protease_stability     # Path for errors must already exist
#SBATCH --job-name="test"       # Name of job


# unload any modules that carried over from your command line session
module purge

# add a project directory to your PATH (if needed)
#export PATH=$PATH:/projects/30802/protease_stability/src

# load modules you need to use
conda activate pythonenv

# A command you actually want to execute:
python src/evals/hyperparameter_search.py