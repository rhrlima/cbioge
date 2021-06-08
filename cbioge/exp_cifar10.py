'''
DSGE para CIFAR 10

epochs 10
batch 32

pop 20
evals 500
selection 5
halfandhalf 60/40
elitism 0.25

'''
import os

from cbioge.algorithms.dsge import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import DSGECrossover
from cbioge.algorithms.mutation import DSGENonterminalMutation
from cbioge.algorithms.operators import ElitistReplacement, HalfAndHalfOperator

from cbioge.datasets.dataset import read_dataset_from_pickle
from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem

from cbioge.utils import checkpoint as ckpt
from cbioge.utils.experiments import check_os


def run_evolution():

    # check if Windows to limit GPU memory and avoid errors
    check_os()

    base_path = 'exp_cifar10'
    ckpt.ckpt_folder = os.path.join(base_path, str(os.getpid()))

    dataset = read_dataset_from_pickle('data/datasets/cifar10.pickle')
    parser = Grammar('data/grammars/cnn.json')

    problem = CNNProblem(parser, dataset)
    problem.epochs = 10
    problem.batch_size = 32
    problem.timelimit = 3600 #1h
    problem.workers = 2
    problem.multiprocessing = 1

    algorithm = GrammaticalEvolution(problem, parser)
    algorithm.pop_size = 20
    algorithm.max_evals = 500
    algorithm.selection = TournamentSelection(t_size=5, maximize=True)
    crossover = DSGECrossover(cross_rate=1.0)
    mutation = DSGENonterminalMutation(mut_rate=1.0, parser=parser, end_index=4)
    algorithm.crossover = HalfAndHalfOperator(op1=crossover, op2=mutation, rate=0.6)
    algorithm.replacement = ElitistReplacement(rate=0.25, maximize=True)

    # 0 - sem log
    # 1 - log da evolução
    # 2 - log do problema
    # 3 - log da gramatica
    verbose = 1
    algorithm.verbose = verbose > 0
    problem.verbose = verbose > 1
    parser.verbose = verbose > 2

    population = algorithm.execute(checkpoint=True)

    # TODO melhorar o post-run
    population.sort(key=lambda x: x.fitness, reverse=True)
    for s in population:
        print(s.fitness, s)


if __name__ == '__main__':
    run_evolution()