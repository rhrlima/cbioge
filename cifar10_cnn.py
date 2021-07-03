from cbioge.datasets import Dataset
from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem
from cbioge.algorithms import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import DSGECrossover
from cbioge.algorithms.mutation import DSGENonterminalMutation
from cbioge.algorithms.operators import ElitistReplacement, HalfAndHalfOperator
from cbioge.utils.experiments import check_os, get_simple_args
from cbioge.utils import checkpoint as ckpt

def run_evolution():

    # defines the checkpoint folder
    ckpt.ckpt_folder = get_simple_args('c10cnn').checkpoint

    problem = CNNProblem(
        Grammar('data/grammars/cnn3.json'), 
        Dataset.from_pickle('data/datasets/cifar10_clean.pickle', valid_split=0.1), 
        batch_size=128, 
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
            op2=DSGENonterminalMutation(mut_rate=1.0, parser=problem.parser, end_index=9), 
            rate=0.6), 
        replacement=ElitistReplacement(rate=0.25, maximize=True), 
        verbose=True)

    population = algorithm.execute(checkpoint=True)

    population.sort(key=lambda x: x.fitness, reverse=True)
    for s in population:
        print(f'{float(s.fitness):.3f}', s)

    print('UNIQUE SOLUTIONS', len(algorithm.unique_solutions))


if __name__ == '__main__':
    check_os()
    run_evolution()