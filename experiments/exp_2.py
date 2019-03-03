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

runs = 5
datasets = ['c10']
evals = ['600', '1000']

# compare the running jobs with the configs we want to run
for d in datasets:
	for e in evals:
		for r in range(runs):
			jobname = f'{d}-{e}-{r+1}'
			command = f"'../grammars/cnn.bnf ../datasets/{d}.pickle -p 20 -ep 50 -b 128 -e {e} -f {jobname}'"
			print(command in running, command)