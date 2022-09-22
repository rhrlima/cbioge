import os
import numpy as np

from keras.models import Model

from cbioge.algorithms import Solution
from cbioge.datasets.dataset import Dataset
from cbioge.problems import DNNProblem
from cbioge.grammars import Grammar
from cbioge.algorithms import GrammaticalEvolution
from cbioge.algorithms.selection import TournamentSelection
from cbioge.algorithms.crossover import OnePointCrossover
from cbioge.algorithms.mutation import PointMutation
from cbioge.algorithms.replacement import ReplaceWorst

class MockProblem(DNNProblem):

    def _build_model(self, mapping: list) -> Model:
        return None

    def evaluate(self, solution: Solution) -> bool:
        return np.random.random()

def test_evolution():

    base_dir = os.path.dirname(os.path.dirname(__file__))

    grammar_file = os.path.join(base_dir, 'assets', 'test_grammar.json')
    pickle_file = os.path.join(base_dir, 'assets', 'pickle_dataset.pickle')

    parser = Grammar(grammar_file)

    problem = MockProblem(
        parser=parser,
        dataset=Dataset.from_pickle(pickle_file)
    )

    algorithm = GrammaticalEvolution(
        problem=problem,
        selection=TournamentSelection(),
        crossover=OnePointCrossover(1.0),
        mutation=PointMutation(parser, 1.0),
        replacement=ReplaceWorst()
    )

    algorithm.execute()
