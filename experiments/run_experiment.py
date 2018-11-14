import sys, os
sys.path.append('..')

from algorithms import pge
from grammars import BNFGrammar
from problems import CnnProblem
from utils import checkpoint

#problem.DEBUG = False
#pge.DEBUG = False

# dataset and grammar
pickle_file = None
grammar_file = None

# checkpoint folder and flag
folder = None
checkp = False

if __name__ == '__main__':

	if len(sys.argv) < 3:
		print('expected: <grammar> <dataset> [checkp folder [from checkp]]')
		exit()

	grammar_file = sys.argv[1] 
	pickle_file = sys.argv[2]
	if len(sys.argv) > 3: folder = sys.argv[3]
	if len(sys.argv) > 4: checkp = sys.argv[4]

	# read grammar and setup parser
	#grammar.load_grammar(grammar_file)
	parser = BNFGrammar(grammar_file)

	# reading dataset
	problem = CnnProblem(parser, pickle_file)
	#my_problem.load_dataset_from_pickle(pickle_file)
	pge.problem = problem

	# problem parameters
	problem.batch_size = 128
	problem.epochs = 1

	# checkpoint folder
	checkpoint.ckpt_folder = folder if folder else 'checkpoints/'

	# changing pge default parameters
	#pge.SEED = 42
	pge.POP_SIZE = 2
	pge.MAX_EVALS = 10 # 300 gen

	print('--config--')
	print('DATASET', pickle_file)
	print('GRAMMAR', grammar_file)
	print('CKPT', folder, checkp)

	print('SEED', pge.SEED)
	print('POP', pge.POP_SIZE)
	print('EVALS', pge.MAX_EVALS)
	print('CROSS', pge.CROSS_RATE)
	print('MUT', pge.MUT_RATE)
	print('PRUN', pge.PRUN_RATE)
	print('DUPL', pge.DUPL_RATE)

	print('--running--')
	best = pge.execute(checkpoint=checkp)

	print('--best solution--')
	print(best.fitness, best)
	best.phenotype.summary()

	print('--testing--')
	score = best.phenotype.evaluate(
		problem.x_test, 
		problem.y_test, 
		verbose=0)
	print('loss: {}\taccuracy: {}'.format(score[0], score[1]))