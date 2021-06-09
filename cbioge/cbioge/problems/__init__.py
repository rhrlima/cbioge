from cbioge.problems.problem import BaseProblem, DNNProblem


from cbioge.problems.classification import CNNProblem
from cbioge.problems.segmentation import UNetProblem
from cbioge.problems.gcnProblem import GCNProblem
from cbioge.problems.regression import SymbolicRegressionProblem
from cbioge.problems.stringmatch import StringMatchProblem


__all__ = [
	'BaseProblem',
	'DNNProblem',
    'CNNProblem',
    'UNetProblem',
    'GCNProblem',
    'SymbolicRegressionProblem',
    'StringMatchProblem',
]
