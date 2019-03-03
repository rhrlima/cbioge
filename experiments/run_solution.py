import sys
import os
sys.path.append('..')

import argparse
import keras
import numpy as np

from grammars import BNFGrammar
from problems import CnnProblem

#disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_arg_parsersed():

	parser = argparse.ArgumentParser(prog='script.py', description='run experiment')

	parser.add_argument('grammar', type=str, help='grammar file in bnf format')
	parser.add_argument('dataset', type=str, help='dataset file in pickle format')
	parser.add_argument('solution', type=str)
	parser.add_argument('-e', '--epochs', default=500, type=int)
	parser.add_argument('-b', '--batch', default=32, type=int)
	parser.add_argument('-v', '--verbose', default=0, type=int)

	return parser.parse_args()


if __name__ == '__main__':

	# parses the arguments
	args = get_arg_parsersed()

	# read grammar and setup parser
	parser = BNFGrammar(args.grammar)

	# problem dataset and parameters
	problem = CnnProblem(parser, args.dataset)
	problem.batch_size = args.batch
	problem.epochs = args.epochs

	print('--config--')

	print('DATASET', args.dataset)
	print('GRAMMAR', args.grammar)
	print('SOLUTION', args.solution)
	print('EPOCHS', args.epochs)
	print('BATCH', args.batch)

	print('--running--')
	
	solution = [int(s) for s in args.solution.replace(' ', '').split(',')]
	json_model = problem.map_genotype_to_phenotype(solution)
	
	print(json_model)
	
	if json_model:
		model = keras.models.model_from_json(json_model)
		model.summary()

		model.compile(
			loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy']
		)

		print('--training--')
		hist = model.fit(problem.x_train, problem.y_train, batch_size=args.batch, 
			epochs=args.epochs, verbose=args.verbose)
		print('loss: {}\taccuracy: {}'.format(np.mean(hist.history['loss']), 
			np.mean(hist.history['acc'])))

		print('--testing--')
		score = model.evaluate(problem.x_test, problem.y_test, verbose=0)
		print('loss: {}\taccuracy: {}'.format(score[0], score[1]))
