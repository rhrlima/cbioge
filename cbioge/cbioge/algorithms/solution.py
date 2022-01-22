import copy
from typing import Any, Dict


class Solution:
    '''Represents a solution and common components that can be used in a
    wide range of problems.\n

    Custom data should use the data dictionary to store
    statistics or other useful info accessed by the problem.

    The search engine will use the basic components, and the problem
    (usually a custom class) will make use of most of it, or even more.'''

    def __init__(self,
        genotype: list=[],
        phenotype: Any=None,
        fitness: float=-1.0,
        evaluated: bool=False,
        data: dict={},
        id: int=None # pylint: disable=redefined-builtin
    ):

        self.id = id # pylint: disable=invalid-name
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.evaluated = evaluated
        self.data = data

    def __str__(self):
        return str(self.genotype)

    def __eq__(self, other: 'Solution') -> bool:
        if not isinstance(other, Solution):
            return False
        return self.to_json() == other.to_json()

    def to_json(self) -> Dict[str, Any]:
        return self.__dict__

    def copy(self, deep: bool=False) -> 'Solution':
        if deep:
            # with all data inside
            return copy.deepcopy(self)
        # all defaults but the genotype
        return Solution(copy.deepcopy(self.genotype))

    @classmethod
    def from_json(cls, json_data: dict) -> 'Solution':
        if not isinstance(json_data, dict):
            return cls()
        return cls(**json_data)
