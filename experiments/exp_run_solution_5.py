import sys
sys.path.append('..')

from grammars import grammar
from problems import problem

VERBOSE = True

GRAMMAR_FILE = '../grammars/cnn.bnf'
DATASET_FILE = '../datasets/cifar-100/cifar-100.pickle'

BATCH_SIZE = 256
EPOCHS = 100

solutions = [
	[ 92,  43, 136, 151, 204,  85, 193,  47, 185, 221],
	[ 59,   8, 227,  77, 117,  91, 178, 173],
	[219, 118, 138,  67,],
	[175, 113,  13,  12, 180, 136, 244, 231, 136, 175, 113,  13, 175, 113,  13,  12, 180, 136, 244, 231],
	[ 13,  37,],
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