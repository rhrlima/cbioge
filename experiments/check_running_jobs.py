import re
import subprocess
import os

data = subprocess.check_output(['qstat']).decode('utf-8')
data = data.split('\n')

running = []
for i, line in enumerate(data):
  line = re.sub('\\s+', '@', line)
  if 1 < i < len(data) - 1:
    #print(line.split('@')[1])
    running.append(line.split('@')[1])

datasets = ['c10', 'c100']
evals = ['600', '1000']
runs = 5

for d in datasets:
  for e in evals:
    for r in range(runs):
      c = f'{d}-{e}-{r+1}'
      if c not in running:
        #print(c, 'NOT running')
        os.system(f'./single.sh {d} {e} 1 {r+1}')
