import re
import subprocess
import sys
import os

# get the output from 'qstat' command
data = subprocess.check_output(['qstat']).decode('utf-8').split('\n')

# get the jobnames and add them to a list
running = []
for i, line in enumerate(data):
  line = re.sub('\\s+', '@', line)
  if 1 < i < len(data) - 1:
    running.append(line.split('@')[1])


scriptname = 'run_experiment.py'
runs = 5
datasets = ['c10', 'mnist']
evals = ['600', '1000']

# runs the configs that are not in the running list
for d in datasets:
	for e in evals:
		for r in range(runs):
			jobname = f'{d}-{e}-{r+1}'
			if jobname not in running:
				args = f"'../grammars/cnn.bnf ../datasets/{d}.pickle -p 20 -ep 50 -b 128 -e {e} -f {jobname}'"
				command = f'./run_intel.sh {jobname} {scriptname} {args}'
				os.system(command)