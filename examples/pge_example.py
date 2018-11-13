import sys, os
sys.path.append('..')

#disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)

from algorithms import pge
from grammars import grammar
from problems import problem
from utils import checkpoint

import json
import keras
import numpy as np

DEBUG = False
problem.DEBUG = False
pge.DEBUG = False

# dataset and grammar
pickle_file = '../datasets/mnist/mnist.pickle'
grammar_file = '../grammars/cnn.bnf'

checkpoint.ckpt_folder = 'pge_ckpt'

# read grammar and setup parser
grammar.load_grammar(grammar_file)

# reading dataset
my_problem = problem.CnnProblem()
my_problem.load_dataset_from_pickle(pickle_file)
pge.problem = my_problem
pge.MAX_PROCESSES = 10

# problem parameters
my_problem.batch_size = 128
my_problem.epochs = 50

# changing pge default parameters
#pge.SEED = 42
pge.POP_SIZE = 20
pge.MAX_EVALS = 300 #15 gen

print('--config--')
print('DATASET', pickle_file)
print('GRAMMAR', grammar_file)
print('CKPT', checkpoint.ckpt_folder)

print('BSIZE', my_problem.batch_size)
print('EPOCHS', my_problem.epochs)

print('SEED', pge.SEED)
print('POP', pge.POP_SIZE)
print('EVALS', pge.MAX_EVALS)
print('CROSS', pge.CROSS_RATE)
print('MUT', pge.MUT_RATE)
print('PRUN', pge.PRUN_RATE)
print('DUPL', pge.DUPL_RATE)

print('--running--')
best = pge.execute(checkpoint=False)

print('--best solution--')
print(best, best.fitness)

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

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print('--training--')
	hist = model.fit(my_problem.x_train, my_problem.y_train, batch_size=128, 
		epochs=1, verbose=0)
	print('loss: {}\taccuracy: {}'.format(
		np.mean(hist.history['loss']), np.mean(hist.history['acc'])))

	print('--testing--')
	score = model.evaluate(my_problem.x_test, my_problem.y_test, verbose=0)
	print('loss: {}\taccuracy: {}'.format(score[0], score[1]))