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


def save_population(population, evals, filename='pop.ckpt'):
	
	pickle_population = [save_solution(s) for s in population]
	data = {
		'evals': evals,
		'population': pickle_population
	}
	
	with open(filename, 'wb') as f:
		pickle.dump(data, f)
	print('population saved to file "{}"'.format(filename))
	

def load_population(filename='pop.ckpt'):

	with open(filename, 'rb') as f:
		temp = pickle.load(f)
		
	evals = temp['evals']
	population = []
	for s in temp['population']:
		solution = load_solution(s)
		population.append(solution)
	print('population loaded from file "{}"'.format(filename))
	return population, evals


def save_args(args, filename='args.ckpt'):

	with open(filename, 'wb') as f:
		pickle.dump(args, f)
	print(f'args saved to file "{filename}"')


def load_args(filename):

	with open(filename, 'rb') as f:
		args = pickle.load(f)
	print(f'args loaded from file "{filename}"')
	return args