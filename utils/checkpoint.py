import pickle

#def save_model


def save_solution(solution):
	try:
		pickled_solution = pickle.dumps(solution)
	except Exception as e:
		print(e)
		return None
	return	pickled_solution


def load_solution(pickle_data):
	try:
		solution = pickle.loads(pickle_data)
	except Exception as e:
		print(e)
		return None
	return	solution


def save_population(population):
	pass