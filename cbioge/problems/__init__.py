from cbioge.problems.problem import BaseProblem

from cbioge.problems.classification import CNNProblem
from cbioge.problems.segmentation import UNetProblem

# tests?
from cbioge.problems.regression import SymbolicRegressionProblem
from cbioge.problems.stringmatch import StringMatchProblem


__all__ = [
	'BaseProblem',
	
    'CNNProblem',
    'UNetProblem',

    'SymbolicRegressionProblem',
    'StringMatchProblem',
]
