import sys
sys.path.append('..')

from grammars import grammar
from problems import problem

VERBOSE = True

GRAMMAR_FILE = '../grammars/cnn.bnf'
DATASET_FILE = '../datasets/cifar-10/cifar-10.pickle'

BATCH_SIZE = 256
EPOCHS = 100

solutions = [
	[235, 220, 235,  13, 210],
	[ 63, 209, 120, 138, 130, 117],
	[166, 165],
	[ 94, 117],
	[ 34, 171],
]

grammar.load_grammar(GRAMMAR_FILE)

p = problem.CnnProblem()
p.load_dataset_from_pickle(DATASET_FILE)

for genes in solutions:
	#genes = solutions[0]
	print('solution:', genes)

	model = p.map_genotype_to_phenotype(genes)
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print('train')
	h = model.fit(p.x_train, p.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)
	print('loss: {}\taccuracy: {}'.format(h.history['loss'], h.history['acc']))

	print('valid')
	score = model.evaluate(p.x_valid, p.y_valid, verbose=VERBOSE)
	print('loss: {}\taccuracy: {}'.format(score[0], score[1]))

	print('test')
	score = model.evaluate(p.x_valid, p.y_valid, verbose=VERBOSE)
	print('loss: {}\taccuracy: {}'.format(score[0], score[1]))