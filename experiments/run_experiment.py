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

	parser = argparse.ArgumentParser(
		prog='script.py', 
		description='run experiment')

	parser.add_argument('grammar', 
		#dest='grammar_file', 
		type=str, 
		help='grammar file in bnf format')

	parser.add_argument('dataset', 
		#dest='dataset', 
		type=str, 
		help='dataset file in pickle format')

	parser.add_argument('-e', '--evals', 
		#dest='evals', 
		default=20, 
		type=int, 
		help='number of max evaluations')

	parser.add_argument('-f', '--folder', 
		dest='folder', 
		default='checkpoints', 
		help='folder where checkpoints will be saved')

	parser.add_argument('-c', '--checkpoint', 
		#dest='checkpoint', 
		default=False, 
		type=str2bool, 
		help='indication whether the experiment should continue from checkpoint')

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
	problem.batch_size = 128
	problem.epochs = 50

	# checkpoint folder
	checkpoint.ckpt_folder = args.folder

	# changing pge default parameters
	pge.problem = problem
	pge.POP_SIZE = 20
	pge.MAX_EVALS = args.evals
	pge.MAX_PROCESSES = 8

	print('--config--')
	print('DATASET', args.dataset)
	print('GRAMMAR', args.grammar)
	print('CKPT', args.folder, args.checkpoint)

	print('SEED', pge.SEED)
	print('POP', pge.POP_SIZE)
	print('EVALS', pge.MAX_EVALS)
	print('CROSS', pge.CROSS_RATE)
	print('MUT', pge.MUT_RATE)
	print('PRUN', pge.PRUN_RATE)
	print('DUPL', pge.DUPL_RATE)

	#
	exit()

	print('--running--')
	best = pge.execute(checkpoint=args.checkpoint)

	print('--best solution--')
	print(best.fitness, best)

	if best.phenotype:
		model = keras.models.model_from_json(best.phenotype)
		model.summary()

		opt = keras.optimizers.Adam(
			lr=0.01, 
			beta_1=0.9, 
			beta_2=0.999, 
			epsilon=1.0 * 10**-8, 
			decay=0.001, 
			amsgrad=False)

		model.compile(loss='categorical_crossentropy', 
			optimizer='adam', 
			metrics=['accuracy'])

		print('--training--')
		hist = model.fit(problem.x_train, problem.y_train, 
			batch_size=128, epochs=50, verbose=0)
		print('loss: {}\taccuracy: {}'.format(
			np.mean(hist.history['loss']), np.mean(hist.history['acc'])))

		print('--testing--')
		score = model.evaluate(problem.x_test, problem.y_test, verbose=0)
		print('loss: {}\taccuracy: {}'.format(score[0], score[1]))