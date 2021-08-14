import json, logging, re
import numpy as np


class Grammar:

    def __init__(self, grammar_file, verbose=False, max_depth=10):
        self._read_grammar(grammar_file)
        self.max_depth = max_depth
        self.verbose = verbose
        self.precision = 5
        self.logger = logging.getLogger('cbioge')
    
    def _read_grammar(self, grammar_file):
        ''' reads a file expecting a json structure
            expected keys:
                name: name of the grammar

                blocks: pair of key/value with key being the nonterminal from
                the grammar, and the value a list containing the name of the layer
                from keras and the names of the parameters present in the grammar.
                
                grammar: pair of key/value with the key being the nonterminal, and
                the value being a list of the productions for that nonterminal
        '''

        with open(grammar_file) as f:
            data = json.load(f)

        self.name = data['name']
        self.blocks = data['blocks']
        self.rules = data['rules']
        self.nonterm = list(self.rules.keys())

        self._validate_grammar()

    def _validate_grammar(self):
        ''' checks for inconsitencies in the grammar file:
            - missing rules
            - missing blocks
            - unused rules
            - unused blocks
        '''
        # # missing rules using defined blocks
        # print('### blocks')
        # for block in self.blocks:
        #     print(block)
        
        # # missing blocks using defined rules
        # print('### rules')
        # for rule in self.rules:
        #     print(rule)
        return True

    def _parse_special_types(self, value):
        ''' Parses special types present in the grammar.

            Parses a string in the form of "[int, int]" or "[float, float]"
            to the correct types and return a random between the interval.

            ex: [min, max] is parsed to random between min and max.
        '''
        # TODO buscar maneira melhor de representar rand(min, max) na gramatica
        if type(value) is str:
            m = re.match('\\[(\\d+[.\\d+]*),\\s*(\\d+[.\\d+]*)\\]', value)
            if m:
                min_ = eval(m.group(1))
                max_ = eval(m.group(2))
                if type(min_) == int and type(max_) == int:
                    return np.random.randint(min_, max_)
                elif type(min_) == float and type(max_) == float:
                    return round(np.random.uniform(min_, max_), self.precision)
                else:
                    raise TypeError('Type mismatch:', value)
        return value

    def _group_mapping(self, mapping):
        # groups layer name and parameters together
        new_mapping = []
        while len(mapping) > 0:
            if mapping[0] not in self.blocks:
                raise ValueError('Invalid value present in the grammar:', mapping[0])
            new_mapping.append(mapping[:len(self.blocks[mapping[0]])])
            mapping = mapping[len(self.blocks[mapping[0]]):]
        return new_mapping

    def _recursive_parse_call(self, genotype, added, symb, depth):
        ''' recursive method to produce the grammar expansion

            receives the current state of the genotype, the list
            of added values (if needed), the current symbol and depth
        '''
        production = []
       
        if genotype[self.nonterm.index(symb)] == []:
            value = np.random.randint(0, len(self.rules[symb]))
            added[self.nonterm.index(symb)].append(value)
            genotype[self.nonterm.index(symb)].append(value)

            if self.verbose:
                self.logger.debug(f'Not enough values. Adding: {value} to {symb}')

        value = genotype[self.nonterm.index(symb)].pop(0)
        expansion = self.rules[symb][value]
        
        for s in expansion:
            if s not in self.nonterm:
                production.append(self._parse_special_types(s))
            else:
                production += self._recursive_parse_call(genotype, added, s, depth+1)

        return production

    def _recursive_create_call(self, max_depth, genotype, symb, depth=0):

        value = np.random.randint(0, len(self.rules[symb]))
        expansion = self.rules[symb][value]

        # if expansion is recursive, pick another option
        if depth > max_depth and symb in expansion:
            # if self.verbose:
                #print('[create] WARNING. Max depth reached and next expansion is recursive.')
                #print(depth, symb, expansion, end=' ')
            old_expansion = expansion
            while symb in expansion:
                value = np.random.randint(0, len(self.rules[symb]))
                expansion = self.rules[symb][value]
            if self.verbose:
                self.logger.warning('Max depth reached and next expansion is recursive.')
                self.logger.warning(f'{depth} {symb} {old_expansion} changed to: {expansion}')
                #print('changed to:', symb, expansion)

        genotype[self.nonterm.index(symb)].append(value)

        for curr_symb in expansion:
            if curr_symb in self.nonterm:
                self._recursive_create_call(max_depth, genotype, curr_symb, depth+1)

        return genotype

    def create_solution(self, max_depth=None):
        ''' creates a random solution based on the grammar, according to the 
            DSGE method

            returns: a list of lists containing integer values that is a valid
            solution related to the grammar
        '''
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

    def recursive_parse(self, genotype):
        ''' performs the initial call for the grammar expansion according to the
            DSGE method

            it will perform a grammar expansion according to the grammar
            using the values present in the genotype

            it will modify the genotype as needed, adding more values if
            requested, and removing values if they are not used

            returns: the final production, the modified genotype
        '''

        gen_cpy = [g[:] for g in genotype]
        added = [[] for _ in range(len(self.nonterm))]
        symb = self.nonterm[0]

        production = self._recursive_parse_call(genotype=gen_cpy, 
                                                added=added, 
                                                symb=symb, depth=0)

        for symb in range(len(genotype)):
            # removes the values left off during the expansion, from the original genotype
            for _ in range(len(gen_cpy[symb])):
                genotype[symb].pop()
            # adds the values present in the 'added' list, to the original genotype
            genotype[symb].extend(added[symb])

        mapping = self._group_mapping(list(filter(lambda x: x != '&', production)))

        if self.verbose:
            self.logger.debug(f'Genotype: {genotype}')
            self.logger.debug(f'Mapping: {mapping}')
            self.logger.debug(f'Added: {added}')
            self.logger.debug(f'Removed: {gen_cpy}')

        return mapping#, genotype
