import os

jobname = 'test'
scriptname = 'grammatical_evolution.py'
args = '../grammars/cnn.bnf ../datasets/mnist.pickle'

script = f"""
	#!/bin/bash\n
	#PBS -N {name}\n
	#PBS -l nodes=1\n
	#PBS -l walltime=24:00:00\n
	
	cd \\$PBS_O_WORKDIR\n
	python {scriptname} {args}
	"""
os.system(f'echo -e {script} | qsub')
os.system('sleep 0.5')

print('done')

