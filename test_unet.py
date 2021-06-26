import logging, os, warnings
warnings.filterwarnings("ignore")

from cbioge.algorithms.dsge import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import DSGECrossover
from cbioge.algorithms.mutation import DSGENonterminalMutation
from cbioge.algorithms.operators import ElitistReplacement, HalfAndHalfOperator

from cbioge.datasets.dataset import read_dataset_from_pickle
from cbioge.grammars import Grammar
from cbioge.problems import UNetProblem

from cbioge.utils import checkpoint as ckpt
from cbioge.utils import experiments as exp
from cbioge.utils.experiments import check_os


def run_evolution():

    # defines the checkpoint folder
    ckpt.ckpt_folder = exp.get_simple_args('unet').checkpoint

    problem = UNetProblem(
        Grammar('data/grammars/unet_restricted.json'), 
        read_dataset_from_pickle('data/datasets/membrane.pickle'), 
        batch_size=10, 
        epochs=1, 
        workers=2, 
        multiprocessing=True)
    problem.train_size = 10
    problem.valid_size = 10

    algorithm = GrammaticalEvolution(problem, problem.parser, 
        pop_size=20, 
        max_evals=1000, 
        selection=TournamentSelection(t_size=5, maximize=True), 
        crossover=HalfAndHalfOperator(
            op1=DSGECrossover(cross_rate=1.0), 
            op2=DSGENonterminalMutation(mut_rate=1.0, parser=problem.parser, end_index=8), 
            rate=0.6), 
        replacement=ElitistReplacement(rate=0.25, maximize=True), 
        verbose=True)

    population = algorithm.execute(checkpoint=True)

    population.sort(key=lambda x: x.fitness, reverse=True)
    for s in population:
        print(f'{float(s.fitness):.3f}', s)

    print('UNIQUE SOLUTIONS', len(algorithm.all_solutions))


if __name__ == '__main__':
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    check_os()
    run_evolution()
    #test_validity()

# def run_evolution2():

#     # check if Windows to limit GPU memory and avoid errors
#     check_os()

#     base_path = 'exp_unet_test'
#     # deletes folder and sub-folders for clean run
#     if os.path.exists(base_path):
#         shutil.rmtree(base_path)
#     ckpt.ckpt_folder = ckpt.get_new_unique_path(base_path)

#     dataset = read_dataset_from_pickle('data/datasets/membrane.pickle')
#     parser = Grammar('data/grammars/unet_restricted.json')
#     problem = UNetProblem(parser, dataset)

#     problem.epochs = 1
#     problem.batch_size = 5
#     problem.workers = 2
#     problem.multiprocessing = 1

#     problem.train_size = 5
#     problem.valid_size = 5
#     problem.test_size = 5

#     algorithm = GrammaticalEvolution(problem, parser)
#     algorithm.pop_size = 10
#     algorithm.max_evals = 100
#     algorithm.selection = TournamentSelection(t_size=2, maximize=True)
#     algorithm.replacement = ElitistReplacement(rate=0.25, maximize=True)
#     algorithm.crossover = HalfAndHalfOperator(
#         op1=DSGECrossover(cross_rate=1.0), 
#         op2=DSGENonterminalMutation(mut_rate=1.0, parser=parser, end_index=4), 
#         rate=0.6)

#     # 0 - sem log
#     # 1 - log da evolução
#     # 2 - log do problema
#     # 3 - log da gramatica
#     verbose = 0
#     algorithm.verbose = verbose > 0
#     problem.verbose = verbose > 1
#     parser.verbose = verbose > 2

#     population = algorithm.execute(checkpoint=False)

#     # remove and add better post-run
#     population.sort(key=lambda x: x.fitness, reverse=True)
#     for s in population:
#         print(s.fitness, s)


# if __name__ == '__main__':
#     run_evolution()