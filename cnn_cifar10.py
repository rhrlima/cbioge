import os

from cbioge.algorithms.dsge import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import DSGECrossover
from cbioge.algorithms.mutation import DSGENonterminalMutation
from cbioge.algorithms.operators import ElitistReplacement, HalfAndHalfOperator

from cbioge.datasets.dataset import read_dataset_from_pickle
from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem

from cbioge.utils.experiments import check_os


def run_evolution():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    check_os()

    dataset = read_dataset_from_pickle('data/datasets/cifar10.pickle')
    parser = Grammar('data/grammars/cnn_simple.json')

    problem = CNNProblem(parser, dataset, batch_size=64)

    algorithm = GrammaticalEvolution(problem, parser, 
        pop_size=10,
        max_evals=20, 
        selection=TournamentSelection(t_size=5, maximize=True), 
        crossover=HalfAndHalfOperator(
            op1=DSGECrossover(cross_rate=1.0), 
            op2=DSGENonterminalMutation(mut_rate=1.0, parser=parser, end_index=4), 
            rate=0.6), 
        replacement=ElitistReplacement(rate=0.25, maximize=True), 
        verbose=True)

    population = algorithm.execute()

    population.sort(key=lambda x: x.fitness, reverse=True)
    for s in population:
        print(s.fitness, s)


if __name__ == '__main__':
    run_evolution()