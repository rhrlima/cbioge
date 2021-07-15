from cbioge.datasets import Dataset
from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem
from cbioge.algorithms import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import DSGECrossover
from cbioge.algorithms.mutation import DSGENonterminalMutation
from cbioge.algorithms.operators import ElitistReplacement, HalfAndHalfOperator
from cbioge.utils.experiments import check_os, basic_setup
from cbioge.utils.post_run import run_best_from_checkpoint, plot_history


def run_evolution():

    problem = CNNProblem(
        Grammar('cbioge/assets/grammars/res_cnn.json'), 
        Dataset.from_pickle('cbioge/assets/datasets/cifar10.pickle', 
            train_size=5000, 
            valid_split=0.1, 
            test_size=500), 
        batch_size=128, 
        epochs=10, 
        workers=2, 
        use_multiprocessing=True)

    algorithm = GrammaticalEvolution(problem,  
        verbose=True, 
        pop_size=20, 
        max_evals=40, 
        selection=TournamentSelection(t_size=5, maximize=True), 
        crossover=HalfAndHalfOperator(
            op1=DSGECrossover(), 
            op2=DSGENonterminalMutation(parser=problem.parser, end_index=7), 
            rate=0.6), 
        replacement=ElitistReplacement(rate=0.25, maximize=True))

    algorithm.execute(checkpoint=True)

    run_best_and_plot(problem)


def run_best_and_plot(problem, folder=None):

    problem.epochs = 50
    problem.test_eval = True

    best = run_best_from_checkpoint(problem, folder)
    plot_history(best.data['history'], folder)

    print(f"id {best.id} evo {best.data['evo_fit']:.2} pos {best.fitness:.2}")
    print(best)
    if 'mapping' in best.data:
        print(best.data['mapping'])


if __name__ == '__main__':
    check_os()
    basic_setup('c10res', True)
    run_evolution()