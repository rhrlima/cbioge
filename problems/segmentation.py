from .problem import BaseProblem
import numpy as np


class ImageSegmentationProblem(BaseProblem):

    def __init__(self, parser, dataset=None):
        self.parser = parser

    def map_genotype_to_phenotype(self, genotype):
        return None

    def evaluate(self, solution):
        return 0
