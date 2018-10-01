import ge
import grammar
import problem


# dataset and grammar
pickle_file = '../datasets/mnist/mnist.pickle'
grammar_file = 'cnn2.bnf'

# read grammar and setup parser
grammar.load_grammar(grammar_file)

# reading dataset
mnist_problem = problem.CnnProblem()
mnist_problem.load_dataset_from_pickle(pickle_file) 
ge.problem = mnist_problem

# changing GE default parameters
ge.POP_SIZE = 1
ge.MAX_EVALS = 1

# ----

print('Running GE')
best = ge.execute()

print('Best Solution')
print(best)
best.phenotype.summary()

print('Testing')
best.phenotype.compile
score = best.phenotype.evaluate(mnist_problem.x_test, mnist_problem.y_test, 1)
print('Score -> loss: {}\taccuracy: {}'.format(score[0], score[1]))