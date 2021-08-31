import logging, os, warnings; warnings.filterwarnings("ignore")

from cbioge.datasets import Dataset
from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem
from cbioge.algorithms import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import OnePointCrossover
from cbioge.algorithms.mutation import NonterminalMutation
from cbioge.algorithms.replacement import ElitistReplacement
from cbioge.algorithms.operators import HalfAndHalfOperator
from cbioge.utils.experiments import check_os, basic_setup


def run_evolution():

    problem = CNNProblem(
        Grammar('cbioge/assets/grammars/res_cnn.json'), 
        Dataset.from_pickle('cbioge/assets/datasets/cifar10.pickle', 
            train_size=5000), 
        batch_size=256, 
        epochs=1, 
        train_args= {
            'validation_split': 0.1, 
            'workers': 4, 
            'use_multiprocessing': True, 
        })

    GrammaticalEvolution(problem, 
        pop_size=10, 
        max_evals=20, 
        selection=TournamentSelection(t_size=5, maximize=True), 
        crossover=HalfAndHalfOperator(
            op1=OnePointCrossover(), 
            op2=NonterminalMutation(parser=problem.parser, end_index=9), 
            rate=0.6), 
        replacement=ElitistReplacement(rate=0.25, maximize=True), 
        verbose=True
    ).execute(checkpoint=True)


if __name__ == '__main__':
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # GPU check
    # check_os()

    # defines the checkpoint folder
    basic_setup('test_cnn', True)

    run_evolution()