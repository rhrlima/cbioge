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
        genotype: list=None,
        phenotype: Any=None,
        fitness: float=-1.0,
        evaluated: bool=False,
        data: dict=None,
        s_id: int=None
    ):

        self.s_id = s_id
        self.genotype = genotype or []
        self.phenotype = phenotype
        self.fitness = fitness
        self.evaluated = evaluated
        self.data = data or {}

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
            return copy.deepcopy(self)
        return Solution(copy.deepcopy(self.genotype))

    @classmethod
    def from_json(cls, json_data: dict) -> 'Solution':
        if not isinstance(json_data, dict):
            return cls()
        return cls(**json_data)
