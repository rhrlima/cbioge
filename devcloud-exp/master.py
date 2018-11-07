from multiprocessing import Pool
import time
import random

def foo(num):
	wait = random.randint(0, 5)
	print('foo', num, wait)
	time.sleep(wait)
	return num*2


def after(num):
	print('after', num)


print('master script running')

pool = Pool(processes=2)

#for i in range(15):
#	pool.apply_async(func=foo, args=(i,))
res = pool.map_async(foo, range(20))

pool.close()
pool.join()

print(res.get())

print('master script done')