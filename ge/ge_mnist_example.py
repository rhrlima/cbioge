
import ge
import grammar
import problem


pickle_file = '../datasets/mnist/mnist.pickle'
grammar_file = 'cnn2.bnf'


mnist_problem = problem.CnnProblem()
mnist_problem.load_dataset_from_pickle(pickle_file)
mnist_problem.input_shape = (1, 28, 28)
mnist_problem.num_classes = 10


# 
ge.problem = mnist_problem

grammar.load_grammar(grammar_file)

ge.execute()