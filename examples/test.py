import numpy as np


def repair(number, numbers, depth=0, possibilities=None):

	print('#'*depth, 'depth', depth, possibilities)
	if len(numbers) == 10:
		return True

	if number not in numbers:
		numbers.append(number)
		return repair(np.random.randint(0, 10), numbers, depth+1)
	else:

		if possibilities is None:
			possibilities = [np.random.randint(0, 10) for _ in range(5)] #rand poss

		if number in possibilities:
			possibilities.remove(number)

		for p in possibilities:
			valid = repair(p, numbers, depth, possibilities)
			if valid:
				return True
	print('#'*depth, 'depth', depth, possibilities)
	return False	


if __name__ == '__main__':
	
	numbers = []
	print(repair(0, numbers))
	print(numbers)