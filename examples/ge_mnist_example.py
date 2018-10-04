import sys, os
sys.path.append('..')

from algorithms import ge
from grammars import grammar
from problems import problem

# dataset and grammar
pickle_file = '../datasets/mnist/mnist.pickle'
grammar_file = '../grammars/cnn.bnf'

# read grammar and setup parser
grammar.load_grammar(grammar_file)

# reading dataset
mnist_problem = problem.CnnProblem()
mnist_problem.load_dataset_from_pickle(pickle_file) 
ge.problem = mnist_problem

# changing GE default parameters
ge.SEED = 42
ge.POP_SIZE = 50
ge.MAX_EVALS = 2000

print('--config--')
print('SEED', ge.SEED)
print('POP', ge.POP_SIZE)
print('EVALS', ge.MAX_EVALS)
print('CROSS', ge.CROSS_RATE)
print('MUT', ge.MUT_RATE)
print('PRUN', ge.PRUN_RATE)
print('DUPL', ge.DUPL_RATE)

print('--running ge--')
best = ge.execute()

print('--best solution--')
print(best)
best.phenotype.summary()

print('--testing--')
best.phenotype.compile
score = best.phenotype.evaluate(mnist_problem.x_test, mnist_problem.y_test, 1)
print('Score -> loss: {}\taccuracy: {}'.format(score[0], score[1]))
