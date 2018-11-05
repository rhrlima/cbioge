from multiprocessing import Pool
from time import sleep

def f(num):
	print('num', num)
	sleep(2)

print('master script running')

pool = Pool(processes=5)

for i in range(50):

	pool.apply_async(func=f, args=(i,))

pool.close()
pool.join()

print('master script done')