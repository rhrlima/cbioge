import sys, os
sys.path.append('..')

#disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np

from algorithms import pge
from grammars import BNFGrammar
from problems import CnnProblem
from utils import checkpoint
from keras.models import model_from_json

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
	parser = BNFGrammar(grammar_file)

	# problem dataset and parameters
	problem = CnnProblem(parser, pickle_file)	
	problem.batch_size = 128
	problem.epochs = 50

	# checkpoint folder
	folder = 'checkpoints/' if not folder else folder
	checkpoint.ckpt_folder = folder 

	# changing pge default parameters
	pge.problem = problem
	pge.POP_SIZE = 20
	pge.MAX_EVALS = 600
	pge.MAX_PROCESSES = 8

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