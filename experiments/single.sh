#'/bin/bash

dset=$1
evals=$2
ckpt=$3
id=$4

PBS="#!/bin/bash\n\
#PBS -N ${dset}-${evals}-${id}\n\
#PBS -l nodes=1\n\
#PBS -l walltime=24:00:00\n\
#PBS -o output/${dset}-${evals}-${id}.out\n\
#PBS -e error/${dset}-${evals}-${id}.err\n\

cd \$PBS_O_WORKDIR\n\

python run_experiment.py ../grammars/cnn.bnf ../datasets/${dset}.pickle -e ${evals} -f ${dset}-${evals}-${id} -c ${ckpt}"

echo -e ${PBS}
echo -e ${PBS} | qsub
echo "done"

qstat
