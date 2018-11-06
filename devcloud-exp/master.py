from multiprocessing import Pool
from time import sleep

def foo(num):
	print('num', num)
	sleep(2)
	return num

def after(num):
	print('after', num)


print('master script running')

pool = Pool(processes=5)

for i in range(10):
	pool.apply_async(func=foo, args=(i,), callback=after)
#pool.map_async(foo, range(10), callback=after)

pool.close()
pool.join()

print('master script done')