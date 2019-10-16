import copy
import re
import numpy as np


class BNFGrammar:

    def __init__(self, grammar_file):
        self.GRAMMAR = {}
        self.NT = []
        self.MAX_LOOPS = 1000

        self.rule_reg = '<[a-zA-Z0-9_-]+>'

        lines = self._read_grammar(grammar_file)
        self._build_structure(lines)

        # for key in self.GRAMMAR:
        #     print(key, self.GRAMMAR[key])
        # print(self.NT)
        # self._check_recursive_rules()

    def _read_grammar(self, grammar_file):
        lines = []
        with open(grammar_file, 'r') as gf:
            for line in gf:
                line = re.sub('\n', '', line) # remove '\n'

                # if line starts with '#' or it is empty, ignore
                if line.startswith('#') or line == '':
                    continue

                # if line does not match 'new rule line', append to previous
                if re.match(self.rule_reg, line) is None:
                    lines[len(lines)-1] += line
                else:
                    lines.append(line)
        return lines

    def _build_structure(self, lines):
        self.GRAMMAR = {}
        self.NT = []
        for line in lines:

            # split into key and productions
            rule, prod = line.split('::=')
            #[p.strip().split(' ') for p in prod.split('|')]
            self.GRAMMAR[rule.strip()] = []
            for i, p in enumerate(prod.split('|')):
                self.GRAMMAR[rule.strip()].append([])
                for value in p.strip().split(' '):
                    value = self._parse_value(value)
                    self.GRAMMAR[rule.strip()][i].append(value)
                    #self.GRAMMAR[rule.strip()].append(value)

            #self.GRAMMAR[rule.strip()] = [p.strip().split(' ') for p in prod.split('|')]
            self.NT.append(rule.strip())

    def _parse_value(self, value):
        try:
            value = value.replace(' ', '')
            return float(value) if '.' in value else int(value)
        except:
            return value

    def _recursive_parse_call(self, genotype, new_gen, symb, depth):

        prod = []
        
        #print('symbol', symb, 'depth', depth, 'genotype', gen)

        #print(genotype[self.NT.index(symb)])
        if genotype[self.NT.index(symb)] == []:
            value = np.random.randint(0, len(self.GRAMMAR[symb]))
            print('pop from empty list, added:', value)
            new_gen[self.NT.index(symb)].append(value)
        else:
            value = genotype[self.NT.index(symb)].pop(0)

        # print('value', value, 'out of', len(self.GRAMMAR[symb]), symb)
        expansion = self.GRAMMAR[symb][value]
        
        print('###', expansion)
        for s in expansion:
            if s not in self.NT:
                prod.append(s)
            else:
                prod += self._recursive_parse_call(genotype, new_gen, s, depth+1)

        return prod

    def dsge_create_solution(self, max_depth=10, genotype=None, symb=None, depth=0):

        '''falta adicionar comportamento quando ultrapassa 
        profundidade maxima'''

        prod = []
        if symb is None:
            gen = [[] for _ in range(len(self.NT))]
            symb = self.NT[0] # assigns initial symbol
        else:
            gen = genotype

        # print('symbol', symb, 'depth', depth, 'genotype', gen)

        value = np.random.randint(0, len(self.GRAMMAR[symb]))
        gen[self.NT.index(symb)].append(value)
        expansion = self.GRAMMAR[symb][value]

        for s in expansion:
            if s in self.NT:
                self.dsge_create_solution(max_depth, gen, s, depth+1)

        return gen

    def dsge_recursive_parse(self, genotype):

        '''falta incorporar comportamento para quando ultrapassa 
        profundidade maxima

        falta parte que corrige a solucao quando a mesma é
        modificada pelos operadores geneticos'''

        #gen = copy.deepcopy(genotype) # saves genotype before use
        gen_cpy = copy.deepcopy(genotype)
        to_add = [[] for _ in range(len(self.NT))]
        symb = self.NT[0] # assigns initial symbol

        prod = self._recursive_parse_call(
            genotype=gen_cpy, new_gen=to_add, symb=symb, depth=0)

        #print('gen', genotype, prod)
        #print('remove', gen_cpy)
        #print('add', to_add)

        # remove extra values
        for genA, genB in zip(genotype, gen_cpy):
            for value in genB:
                genA.remove(value)

        # add new values when necessary
        for genA, genB in zip(genotype, to_add):
            for value in genB:
                genA.append(value)

        return list(filter(lambda x: x != '&', prod))
