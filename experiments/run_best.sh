#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Parameters: grammar dataset solution"
	exit 1
fi

dset=$1
solution=$2

PBS="#!/bin/bash\n\
#PBS -N ${dset}-${evals}-${i}\n\
#PBS -l nodes=1\n\
#PBS -l walltime=24:00:00\n\
#PBS -o output/${dset}-${evals}-${i}.out\n\
#PBS -e error/${dset}-${evals}-${i}.err\n\
cd \$PBS_O_WORKDIR\n\
python run_solution.py ../grammars/cnn.bnf ${dset} ${solution}"

echo -e ${PBS}
echo -e ${PBS} | qsub
sleep 0.5
echo "done"
