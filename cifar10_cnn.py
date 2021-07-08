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
        Dataset.from_pickle('data/datasets/cifar10_clean.pickle', 
            train_size=5000, 
            valid_split=0.1), 
        batch_size=128, 
        epochs=50, 
        workers=2, 
        use_multiprocessing=True)

    algorithm = GrammaticalEvolution(problem, 
        pop_size=20, 
        max_evals=500, 
        selection=TournamentSelection(t_size=5, maximize=True), 
        crossover=HalfAndHalfOperator(
            op1=DSGECrossover(), 
            op2=DSGENonterminalMutation(parser=problem.parser, end_index=9), 
            rate=0.6), 
        replacement=ElitistReplacement(rate=0.25, maximize=True))

    population = algorithm.execute(checkpoint=True)

    population.sort(key=lambda x: x.fitness, reverse=True)
    for s in population:
        print(s.id, f'{float(s.fitness): .2f}', s)

    print('UNIQUE SOLUTIONS', len(algorithm.unique_solutions))


if __name__ == '__main__':
    check_os()
    run_evolution()