import json
import logging
import re
from typing import Any, List

import numpy as np


RAND_INTERVAL_PATTERN = r'\[(\d+[.\d+]*),\s*(\d+[.\d+]*)\]'
INT_PATTERN = r'\d+'
FLOAT_PATTERN = r'\d+.\d+'


class Grammar:
    '''Grammar class'''

    def __init__(self,
        grammar_file: str,
        verbose: bool=False,
        max_depth: int=10,
        precision: int=5
    ):

        self._read_grammar(grammar_file)
        self.max_depth = max_depth
        self.verbose = verbose
        self.precision = precision
        self.logger = logging.getLogger('cbioge')

    def _read_grammar(self, grammar_file: str):
        '''Reads a file expecting a json structure.\n
        Expected keys:
        name: name of the grammar

        blocks: pair of key/value with key being the nonterminal from
        the grammar, and the value a list containing the name of the layer
        from keras and the names of the parameters present in the grammar.

        grammar: pair of key/value with the key being the nonterminal, and
        the value being a list of the productions for that nonterminal'''

        with open(grammar_file) as file:
            data = json.load(file)

        self.name: str = data['name']
        self.rules: dict = data['rules']
        self.nonterm = list(self.rules.keys())
        if "blocks" in data:
            self.blocks = data['blocks']

    def _parse_special_types(self, value: Any) -> Any:
        '''Parses special types present in the grammar.

        Current covered cases:
        int anf floats are returned directly

        Parses a string in the form of "[int, int]" or "[float, float]"
        to the correct types and return a random between the interval.

        ex: [min, max] is parsed to random between min and max.'''

        # TODO get a better way to represent rand(min, max) in the grammar
        if isinstance(value, (int, float)):
            return value

        # value must be str to work
        match = re.match(RAND_INTERVAL_PATTERN, value)

        if match is None:
            return value

        min_ = match.group(1)
        max_ = match.group(2)

        if re.match(FLOAT_PATTERN, min_) and re.match(FLOAT_PATTERN, max_):
            return round(np.random.uniform(float(min_), float(max_)), self.precision)

        if re.match(INT_PATTERN, min_) and re.match(INT_PATTERN, max_):
            return np.random.randint(int(min_), int(max_))

        raise TypeError(f'Type mismatch: \"{value}\"')

    def _recursive_parse_call(self,
        genotype: List[List[int]],
        added: List[List[int]],
        symb: str,
        depth: int) -> List[List[Any]]:

        production = list()

        if genotype[self.nonterm.index(symb)] == []:
            value = np.random.randint(0, len(self.rules[symb]))
            added[self.nonterm.index(symb)].append(value)
            genotype[self.nonterm.index(symb)].append(value)

            if self.verbose:
                self.logger.debug('Not enough values. Adding: %s to %s', value, symb)

        value = genotype[self.nonterm.index(symb)].pop(0)
        expansion = self.rules[symb][value]

        for curr_symb in expansion:
            if curr_symb in self.nonterm:
                production += self._recursive_parse_call(genotype, added, curr_symb, depth+1)
            else:
                production.append(self._parse_special_types(curr_symb))

        return production

    def _recursive_create_call(self,
        max_depth: int,
        genotype: List[List[int]],
        symb: str,
        depth: int=0
    ) -> List[List[int]]:

        value = np.random.randint(0, len(self.rules[symb]))
        expansion = self.rules[symb][value]

        # if expansion is recursive, pick another option
        if depth > max_depth and symb in expansion:

            # TODO possible infinite loop
            old_expansion = expansion
            while symb in expansion:
                value = np.random.randint(0, len(self.rules[symb]))
                expansion = self.rules[symb][value]

            if self.verbose:
                warn_text = f'Max depth reached and next expansion is recursive.\n\
                    {depth} {symb} {old_expansion} changed to: {expansion}'
                self.logger.warning(warn_text)

        genotype[self.nonterm.index(symb)].append(value)

        for curr_symb in expansion:
            if curr_symb in self.nonterm:
                self._recursive_create_call(max_depth, genotype, curr_symb, depth+1)

        return genotype

    def create_solution(self, max_depth: int=None) -> List[List[int]]:
        '''Creates a random solution based on the grammar'''

        max_depth = self.max_depth if max_depth is None else max_depth
        genotype = [[] for _ in range(len(self.nonterm))]
        symb = self.nonterm[0] # assigns initial symbol

        value = np.random.randint(0, len(self.rules[symb]))
        expansion = self.rules[symb][value]

        genotype[self.nonterm.index(symb)].append(value)

        for curr_symb in expansion:
            if curr_symb in self.nonterm:
                self._recursive_create_call(max_depth, genotype, curr_symb)

        return genotype

    def recursive_parse(self, genotype: List[List[int]]) -> List[List[Any]]:
        '''Performs the mapping of a genotype according to the grammar'''

        gen_cpy = [g[:] for g in genotype]
        added = [[] for _ in range(len(self.nonterm))]
        symb = self.nonterm[0]

        production = self._recursive_parse_call(
            genotype=gen_cpy,
            added=added,
            symb=symb, depth=0
        )

        for symb, _ in enumerate(genotype):
            # removes the values left off during the expansion, from the original genotype
            for _ in range(len(gen_cpy[symb])):
                genotype[symb].pop()
            # adds the values present in the 'added' list, to the original genotype
            genotype[symb].extend(added[symb])

        mapping = list(filter(lambda x: x != '&', production))

        if self.verbose:
            self.logger.debug('Genotype: %s', genotype)
            self.logger.debug('Mapping: %s', mapping)
            self.logger.debug('Added: %s', added)
            self.logger.debug('Removed: %s', gen_cpy)

        return mapping
