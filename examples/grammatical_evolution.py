import sys, os
sys.path.append('..')

#disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import keras
import numpy as np

from algorithms import GrammaticalEvolution
from algorithms import TournamentSelection
from algorithms import OnePointCrossover
from algorithms import PointMutation
from algorithms import GEPrune
from algorithms import GEDuplication

from grammars import BNFGrammar
from problems import CnnProblem
from utils import checkpoint
from keras.models import model_from_json


def get_arg_parsersed():

	parser = argparse.ArgumentParser(prog='script.py', description='run experiment')

	# not optional
	parser.add_argument('grammar', type=str, help='grammar file in bnf format')
	parser.add_argument('dataset', type=str, help='dataset file in pickle format')
	
	# checkpoint
	parser.add_argument('-f', '--folder', dest='folder', default='checkpoints')
	parser.add_argument('-c', '--checkpoint', default=True, type=str2bool)

	# problem
	parser.add_argument('-sd', '--seed', default=None, type=int)
	parser.add_argument('-ep', '--epochs', default=1, type=int)
	parser.add_argument('-b', '--batch', default=32, type=int)
	parser.add_argument('-v', '--verbose', default=0, type=int)

	# algorithm
	parser.add_argument('-p', '--population', default=5, type=int)
	parser.add_argument('-e', '--evals', default=10, type=int)
	parser.add_argument('-cr', '--crossover', default=0.8, type=float)
	parser.add_argument('-mt', '--mutation', default=0.1, type=float)
	parser.add_argument('-pr', '--prune', default=0.1, type=float)
	parser.add_argument('-dp', '--duplication', default=0.1, type=float)
	parser.add_argument('-min', '--mingenes', default=2, type=int)
	parser.add_argument('-max', '--maxgenes', default=10, type=int)
	parser.add_argument('-mp', '--maxprocesses', default=2, type=int)

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

	# checkpoint folder
	#checkpoint.ckpt_folder = args.folder

	# read grammar and setup parser
	parser = BNFGrammar(args.grammar)

	# problem dataset and parameters
	problem = CnnProblem(parser, args.dataset)
	problem.batch_size = args.batch
	problem.epochs = args.epochs

	# genetic operators to GE
	selection = TournamentSelection(maximize=True)
	crossover = OnePointCrossover(cross_rate=args.crossover)
	mutation = PointMutation(mut_rate=args.mutation, min_value=0, max_value=255)
	prune = GEPrune(prun_rate=args.prune)
	duplication = GEDuplication(dupl_rate=args.duplication)

	# changing ge default parameters
	algorithm = GrammaticalEvolution(problem)
	#algorithm.DEBUG = True
	algorithm.POP_SIZE = args.population
	algorithm.MAX_EVALS = args.evals
	algorithm.MAX_PROCESSES = args.maxprocesses
	algorithm.selection = selection
	algorithm.crossover = crossover
	algorithm.mutation = mutation
	algorithm.prune = prune
	algorithm.duplication = duplication

	print('--config--')
	print('DATASET', args.dataset)
	print('GRAMMAR', args.grammar)
	print('EPOCHS', args.epochs)
	print('BATCH', args.batch)
	print('CKPT', args.folder, args.checkpoint)

	print('POP', args.population)
	print('EVALS', args.evals)

	print('--running--')
	best = algorithm.execute(args.checkpoint)

	#print('--best solution--')
	#print(best.fitness, best)
	
	# if best.phenotype:
	# 	model = keras.models.model_from_json(best.phenotype)
	# 	model.summary()

	# 	model.compile(loss='categorical_crossentropy', 
	# 		optimizer='adam', metrics=['accuracy'])

	# 	print('--training--')
	# 	hist = model.fit(problem.x_train, problem.y_train, 
	# 		batch_size=128, epochs=50, verbose=args.verbose)
	# 	print('loss: {}\taccuracy: {}'.format(
	# 		np.mean(hist.history['loss']), np.mean(hist.history['acc'])))

	# 	print('--testing--')
	# 	score = model.evaluate(problem.x_test, problem.y_test, verbose=args.verbose)
	# 	print('loss: {}\taccuracy: {}'.format(score[0], score[1]))
