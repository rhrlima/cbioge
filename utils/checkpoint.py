import pickle

ckpt_folder = 'checkpoints'

def save_solution(solution):
	try:
		pickled_solution = pickle.dumps(solution)
	except Exception as e:
		print(e)
		return None
	return pickled_solution


def load_solution(pickle_data):
	try:
		solution = pickle.loads(pickle_data)
	except Exception as e:
		print(e)
		return None
	return solution


def save_population(population, filename='pop.ckpt'):
	
	pickle_population = [save_solution(s) for s in population]
	with open(filename, 'wb') as f:
		pickle.dump(pickle_population, f)
	print('population saved to file "{}"'.format(filename))
	

def load_population(filename='pop.ckpt'):

	with open(filename, 'rb') as f:
		temp = pickle.load(f)
		
	population = []
	for s in temp:
		solution = load_solution(s)
		population.append(solution)
	print('population loaded from file "{}"'.format(filename))
	return population


def save_args(args, filename='args.ckpt'):

	with open(filename, 'wb') as f:
		pickle.dump(args, f)
	print('args saved to file "{}"'.format(filename))


def load_args(filename):

	with open(filename, 'rb') as f:
		args = pickle.load(f)
	print('args loaded from file "{}"'.format(filename))
	return args