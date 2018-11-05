import sys, os
sys.path.append('..')

from algorithms import ppge
from grammars import grammar
from problems import problem

DEBUG = False
problem.DEBUG = False
ppge.DEBUG = True

# dataset and grammar
pickle_file = '../datasets/mnist/mnist.pickle'
grammar_file = '../grammars/cnn.bnf'

# read grammar and setup parser
grammar.load_grammar(grammar_file)

# reading dataset
my_problem = problem.CnnProblem()
my_problem.load_dataset_from_pickle(pickle_file)
pge.problem = my_problem
pge.MAX_PROCESSES = 10

# problem parameters
my_problem.batch_size = 512
my_problem.epochs = 1

# changing pge default parameters
#pge.SEED = 42
pge.POP_SIZE = 10
pge.MAX_EVALS = 20

print('--config--')
print('DATASET', pickle_file)
print('GRAMMAR', grammar_file)

print('SEED', pge.SEED)
print('POP', pge.POP_SIZE)
print('EVALS', pge.MAX_EVALS)
print('CROSS', pge.CROSS_RATE)
print('MUT', pge.MUT_RATE)
print('PRUN', pge.PRUN_RATE)
print('DUPL', pge.DUPL_RATE)

print('--running--')
best = pge.execute()

print('--best solution--')
print(best, best.fitness)
best.phenotype.summary()

print('--testing--')
score = best.phenotype.evaluate(
	my_problem.x_test, 
	my_problem.y_test, 
	verbose=0)
print('loss: {}\taccuracy: {}'.format(score[0], score[1]))
