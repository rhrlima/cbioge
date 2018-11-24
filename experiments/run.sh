#!/bin/bash

if [ "$#" -ne 4 ]; then
	echo "Parameters: dataset evals from_ckpt num_exp"
	exit 1
fi

dset=$1
evals=$2
ckpt=$3
num_exp=$4

for i in `seq $num_exp`; do

	PBS="#!/bin/bash\n\
	#PBS -N ${dset}-${evals}-${i}\n\
	#PBS -l nodes=1\n\
	#PBS -l walltime=24:00:00\n\
	#PBS -o output/${dset}-${evals}-${i}.out\n\
	#PBS -e error/${dset}-${evals}-${i}.err\n\
	cd \$PBS_O_WORKDIR\n\
	python run_experiment.py ../grammars/cnn.bnf ../datasets/${dset}.pickle -e ${evals} -f ${dset}-${evals}-${i} -c ${ckpt}"

	echo -e ${PBS}
	echo -e ${PBS} | qsub
	sleep 0.5
	echo "done"
done

qstat
