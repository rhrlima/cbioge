import sys, os
sys.path.append('..')

from algorithms import pge
#from problems import problem
from utils import checkpoint

genes = [1, 2, 3, 4, 5]
solution = pge.Solution(genes)

solution.phenotype = "{'key': '32', 'otherkey': 'othervalue'}"
solution.fitness = 2.4
solution.evaluated = True
solution.data['key'] = 'value'

print(solution)
print(solution.fitness)
print(solution.evaluated)
print(solution.data['key'])
print(solution.phenotype)

print('saving')
pickled_solution = checkpoint.save_solution(solution)

if pickled_solution:
	print('sucesso')

other_solution = checkpoint.load_solution(pickled_solution)

if other_solution:
	print(other_solution)
	print(other_solution.fitness)
	print(other_solution.evaluated)
	print(other_solution.data['key'])
	print(other_solution.phenotype)

print(solution == other_solution)

pop = [pge.Solution([i, i, i, i]) for i in range(10)]

checkpoint.save_population(pop)
checkpoint.load_population()

checkpoint.save_args(['a', 1, pop[0]])