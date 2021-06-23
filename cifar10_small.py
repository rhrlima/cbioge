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

    import logging
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    check_os()

    from_last_checkpoint = True
    base_path = 'small'
    ckpt.ckpt_folder = ckpt.get_latest_pid_or_new(base_path)

    dataset = read_dataset_from_pickle('data/datasets/cifar10.pickle')
    parser = Grammar('data/grammars/cnn.json', max_depth=20)

    problem = CNNProblem(parser, dataset)
    problem.epochs = 50
    problem.batch_size = 128
    problem.workers = 4
    problem.multiprocessing = True
    problem.train_size = 5000

    algorithm = GrammaticalEvolution(problem, parser)
    algorithm.pop_size = 20
    algorithm.max_evals = 100
    algorithm.selection = TournamentSelection(t_size=5, maximize=True)
    crossover = DSGECrossover(cross_rate=1.0)
    mutation = DSGENonterminalMutation(mut_rate=1.0, parser=parser, end_index=4)
    algorithm.crossover = HalfAndHalfOperator(op1=crossover, op2=mutation, rate=0.7)
    algorithm.replacement = ElitistReplacement(rate=0.25, maximize=True)

    verbose = 1
    algorithm.verbose = verbose > 0
    problem.verbose = verbose > 1
    parser.verbose = verbose > 2

    population = algorithm.execute(checkpoint=from_last_checkpoint)

    # TODO melhorar o post-run
    population.sort(key=lambda x: x.fitness, reverse=True)
    for s in population:
        print(s.fitness, s)


if __name__ == '__main__':
    run_evolution()