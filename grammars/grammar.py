import copy
import numpy as np
import re


class BNFGrammar:

    GRAMMAR = None
    NT = []
    MAX_LOOPS = 1000

    rule_reg = '<[a-zA-Z0-9_-]+>'

    def __init__(self, grammar_file):
        self.GRAMMAR = {}
        self.NT = []

        lines = self._read_grammar(grammar_file)
        self._build_structure(lines)

        # print(self.GRAMMAR)
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

            self.GRAMMAR[rule.strip()] = [p.strip().split(' ') for p in prod.split('|')]
            self.NT.append(rule.strip())

    def _parse_value(self, value):
        try:
            value = value.replace(' ', '')
            m = re.match('\\[(\\d+[.\\d+]*),\\s*(\\d+[.\\d+]*)\\]', value)
            if m:
                min_ = eval(m.group(1))
                max_ = eval(m.group(2))
                if type(min_) == int and type(max_) == int:
                    return np.random.randint(min_, max_)
                elif type(min_) == float and type(max_) == float:
                    return np.random.uniform(min_, max_)
                else:
                    raise TypeError('type mismatch')

            return float(value) if '.' in value else int(value)
        except:
            return value

    def parse(self, codons):
        index = 0
        loop_count = 0
        match = 0

        prod = self.GRAMMAR['<start>']
        while match is not None:
            match = re.search(self.rule_reg, prod)
            if match is not None:
                token = match.group(0)
                repl = codons[index] % len(self.GRAMMAR[token])
                prod = prod.replace(token, self.GRAMMAR[token][repl], 1)
                index += 1
                if index >= len(codons):
                    index = 0

            loop_count += 1

            # maybe an infinite loop
            if loop_count >= self.MAX_LOOPS:
                return None

        prod = prod.replace('\'\'', '@')\
            .replace('\'', '') \
            .split('@')

        return list(filter(lambda x: x != '&', prod))

    def dsge_parse(self, codons):

        temp = copy.deepcopy(codons)
        match = ''
        prod = self.NT[0]

        while match is not None:
            match = re.search(self.rule_reg, prod)
            if match is not None:
                # print('prod', prod)
                token = match.group(0)
                value = temp[self.NT.index(token)].pop(0)
                replacement = self.GRAMMAR[token][value]
                prod = prod.replace(token, replacement, 1)
                # print('token', token)
                # print('value', value)
                # print('repl', replacement)

        print('final', prod)
        print('codons', codons)

        prod = prod.replace('\' \'', '@')\
            .replace('\'', '') \
            .split('@')

        return list(filter(lambda x: x != '&', prod))

    def create_random_derivation(self):

        codons = [[] for _ in range(len(self.NT))]
        prod = self.NT[0]
        match = ''
        while match is not None:
            match = re.search(self.rule_reg, prod)
            if match is not None:
                # print('prod', prod)
                token = match.group(0)
                # value = codons[self.NT.index(token)].pop(0)
                value = random.randint(0, len(self.GRAMMAR[token])-1)
                replacement = self.GRAMMAR[token][value]
                prod = prod.replace(token, replacement, 1)
                codons[self.NT.index(token)].append(value)
                # print('token', token)
                # print('value', value)
                # print('repl', replacement)

        # print(codons)
        print(prod)
        return codons

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

    def _recursive_parse_call(self, genotype, new_gen, symb, depth):

        prod = []
        
        # print('symbol', symb, 'depth', depth, 'genotype', gen)

        #print(genotype[self.NT.index(symb)])
        if genotype[self.NT.index(symb)] == []:
            value = np.random.randint(0, len(self.GRAMMAR[symb]))
            print('pop from empty list, added:', value)
            new_gen[self.NT.index(symb)].append(value)
        else:
            value = genotype[self.NT.index(symb)].pop(0)

        # print('value', value, 'out of', len(self.GRAMMAR[symb]), symb)
        expansion = self.GRAMMAR[symb][value]
        # print('###', expansion)

        for s in expansion:
            if s not in self.NT:
                # print(symb, s)
                prod.append(self._parse_value(s))
            else:
                prod += self._recursive_parse_call(genotype, new_gen, s, depth+1)

        return prod
