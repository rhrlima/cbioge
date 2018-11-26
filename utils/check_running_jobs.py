import os
import re

data = '''Job ID                    Name             User            Time Use S Queue
------------------------- ---------------- --------------- -------- - -----
199061.c009                c10-600-2        u22446          480:28:5 R batch
199072.c009                c10-1000-2       u22446          503:56:5 R batch
199073.c009                c10-1000-3       u22446          293:59:5 R batch
199074.c009                c10-1000-4       u22446          465:03:5 R batch
199075.c009                c10-1000-5       u22446          455:53:2 R batch
199076.c009                c100-1000-1      u22446                 0 Q batch
199077.c009                c100-1000-2      u22446                 0 Q batch
199078.c009                c100-1000-3      u22446                 0 Q batch
199079.c009                c100-1000-4      u22446                 0 Q batch
199080.c009                c100-1000-5      u22446                 0 Q batch
199082.c009                c100-600-2       u22446                 0 Q batch
199311.c009                c100-600-1       u22446                 0 Q batch
199316.c009                c100-600-3       u22446                 0 Q batch
199317.c009                c100-600-4       u22446                 0 Q batch
199318.c009                c100-600-5       u22446                 0 Q batch
199920.c009                c10-600-1        u22446                 0 Q batch
199921.c009                c10-600-3        u22446                 0 Q batch
199922.c009                c10-600-4        u22446                 0 Q batch
199923.c009                c10-600-5        u22446                 0 Q batch
0'''

data = data.split('\n')

datasets = ['c10', 'c100']
evals = ['600', '1000']
runs = 5

running = []
for i, line in enumerate(data):
	line = re.sub('\\s+', '@', line)
	if 1 < i < len(data) - 1:
		job_id = line.split('@')[1]
		running.append(job_id)

configs = []
for d in datasets:
	for e in evals:
		for r in range(runs):
			c = f'{d}-{e}-{r+1}'
			if c in running:
				print(c, 'running')
			else:
				print(c, 'NOT running')