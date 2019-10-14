from problems.problem import BaseProblem

from problems.classification import CNNProblem
from problems.segmentation import UNetProblem

from problems.regression import SymbolicRegressionProblem
from problems.stringmatch import StringMatchProblem


__all__ = [
	'BaseProblem',
	
    'CNNProblem',
    'UNetProblem',

    'SymbolicRegressionProblem',
    'StringMatchProblem',
]
