import os
import subprocess
from time import sleep

print('master script running')

for i in range(5):

	query = ['qsub', 'slave-job', 
		'-N', 'slave-{0}'.format(i), 
		'-o', 'slave-{0}.out'.format(i), 
		'-F' '\'slave.py\'', str(i)]
	p = subprocess.Popen(query, 
		stdout=subprocess.PIPE, 
		shell=True)

	output, err = p.communicate()
	print('slave {} script output:\n'.format(i), output)

print('master script done')