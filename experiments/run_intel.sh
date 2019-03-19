#!/bin/bash

jobname=$1
scriptname=$2
args=$3

PBS="#!/bin/bash\n\
#PBS -N ${jobname}\n\
#PBS -l nodes=1\n\
#PBS -l walltime=24:00:00\n\

cd \$PBS_O_WORKDIR\n\

python ${scriptname} ${args}"

echo -e ${PBS}
echo -e ${PBS} | qsub
echo "done"