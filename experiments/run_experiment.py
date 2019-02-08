import sys, os
sys.path.append('..')

#disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import keras
import numpy as np

from algorithms import pge
from grammars import BNFGrammar
from problems import CnnProblem
from utils import checkpoint
from keras.models import model_from_json


def get_arg_parsersed():

	parser = argparse.ArgumentParser(prog='script.py', description='run experiment')

	parser.add_argument('grammar', type=str, help='grammar file in bnf format')
	parser.add_argument('dataset', type=str, help='dataset file in pickle format')
	
	parser.add_argument('-f', '--folder', dest='folder', default='checkpoints')
	parser.add_argument('-c', '--checkpoint', default=True, type=str2bool)

	parser.add_argument('-sd', '--seed', default=None, type=int)

	parser.add_argument('-ep', '--epochs', default=5, type=int)
	parser.add_argument('-b', '--batch', default=128, type=int)
	parser.add_argument('-v', '--verbose', default=0, type=int)

	parser.add_argument('-p', '--population', default=True, type=str2bool)
	parser.add_argument('-e', '--evals', default=20, type=int)

	parser.add_argument('-cr', '--crossover', default=0.8, type=float)
	parser.add_argument('-mt', '--mutation', default=0.1, type=float)
	parser.add_argument('-pr', '--prune', default=0.1, type=float)
	parser.add_argument('-dp', '--duplication', default=0.1, type=float)
	parser.add_argument('-min', '--mingenes', default=2, type=int)
	parser.add_argument('-max', '--maxgenes', default=10, type=int)

	parser.add_argument('-mp', '--maxprocesses', default=8, type=int)

	return parser.parse_args()


def str2bool(value):
	if value.lower() in ('true', 't', '1'):
		return True
	elif value.lower() in ('false', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

	# parses the arguments
	args = get_arg_parsersed()

	# read grammar and setup parser
	parser = BNFGrammar(args.grammar)

	# problem dataset and parameters
	problem = CnnProblem(parser, args.dataset)
	problem.batch_size = args.batch
	problem.epochs = args.epochs

	# checkpoint folder
	checkpoint.ckpt_folder = args.folder

	# changing pge default parameters
	pge.problem = problem
	
	pge.SEED 		= args.seed
	pge.POP_SIZE 	= args.population
	pge.MAX_EVALS 	= args.evals
	pge.CROSS_RATE 	= args.crossover
	pge.MUT_RATE 	= args.mutation
	pge.PRUN_RATE 	= args.prune
	pge.DUPL_RATE 	= args.duplication
	pge.MIN_GENES 	= args.mingenes
	pge.MAX_GENES 	= args.maxgenes

	pge.MAX_PROCESSES = args.maxprocesses

	print('--config--')
	print('DATASET', args.dataset)
	print('GRAMMAR', args.grammar)
	print('CKPT', args.folder, args.checkpoint)

	print('SEED', pge.SEED)
	print('POP', pge.POP_SIZE)
	print('EVALS', pge.MAX_EVALS)
	print(f'CROSS({pge.CROSS_RATE}) MUT({pge.MUT_RATE}) PRUN({pge.PRUN_RATE}) DUPL({pge.DUPL_RATE})')

	print('--running--')
	best = pge.execute(checkpoint=args.checkpoint)

	print('--best solution--')
	print(best.fitness, best)
	
	if best.phenotype:
		model = keras.models.model_from_json(best.phenotype)
		model.summary()

		model.compile(loss='categorical_crossentropy', 
			optimizer='adam', metrics=['accuracy'])

		print('--training--')
		hist = model.fit(problem.x_train, problem.y_train, 
			batch_size=128, epochs=50, verbose=args.verbose)
		print('loss: {}\taccuracy: {}'.format(
			np.mean(hist.history['loss']), np.mean(hist.history['acc'])))

		print('--testing--')
		score = model.evaluate(problem.x_test, problem.y_test, verbose=args.verbose)
		print('loss: {}\taccuracy: {}'.format(score[0], score[1]))
