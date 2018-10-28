import sys
sys.path.append('..')

from grammars import grammar
from problems import problem

VERBOSE = True

GRAMMAR_FILE = '../grammars/cnn.bnf'
DATASET_FILE = '../datasets/notmnist/notmnist.pickle'

BATCH_SIZE = 256
EPOCHS = 100

solutions = [
	[217, 21, 150],
	[51, 193],
	[63, 85],
	[102, 23, 221, 61, 171, 58, 102, 23, 221, 61, 171],
	[225, 123, 209, 253, 84, 205],
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