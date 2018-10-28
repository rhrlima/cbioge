import sys
sys.path.append('..')

from grammars import grammar
from problems import problem

VERBOSE = True

GRAMMAR_FILE = '../grammars/cnn.bnf'
DATASET_FILE = '../datasets/mnist/mnist.pickle'

BATCH_SIZE = 256
EPOCHS = 100

solutions = [
	[114, 22, 204, 177],
	[4, 99, 69],
	[204, 94, 253, 201, 69, 181, 186, 43, 94, 253, 201, 69, 204, 7, 253, 201, 69, 181, 186, 43, 94, 253],
	[213, 121, 215],
	[37, 141, 108]
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