#!/bin/bash
#SBATCH --job-name=RDC_sPOD       # Job name
#SBATCH --ntasks=1            # Run on a single CPU
#SBATCH --mem=16gb            # Job memory request
#SBATCH --time=150:00:00       # Time limit
#SBATCH --output=/work/burela/RDC_sPOD_%j.log  # Standard output log
##SBATCH --partition=gbr
##SBATCH --nodelist=node748
#SBATCH --chdir=/homes/math/burela/RDC

export PYTHONUNBUFFERED=1

# Print job info
pwd; hostname; date

export PYTHONPATH="$PWD:$PYTHONPATH"

echo "sPOD run for RDC"

# Common command-line arguments
which_variable=$1
test_scenario=$2
sPOD_dim=$3
mu=$4
tau=$5
gamma1=$6
gamma2=$7
gamma3=$8
nmodes1=$9
nmodes2=${10}
nmodes3=${11}


# Decide which Python script to run based on the parameters.
python3 main.py "/work/burela/data_output" $which_variable $test_scenario $sPOD_dim $mu $tau $gamma1 $gamma2 $gamma3 $nmodes1 $nmodes2 $nmodes3 "/work/burela"


date