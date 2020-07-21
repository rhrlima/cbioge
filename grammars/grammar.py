import re
import numpy as np


class BNFGrammar:

    def __init__(self, grammar_file):
        #self.GRAMMAR = {}
        #self.NT = []
        #self.RECURSIVE = []
        self.verbose = False
        
        self.MAX_LOOPS = 1000

        self.rule_reg = '<[a-zA-Z0-9_-]+>'

        lines = self._read_grammar(grammar_file)

        self._build_structure(lines)

        # for key in self.GRAMMAR:
        #     print(key, self.GRAMMAR[key])
        # print(self.NT)
        #self._check_recursive_rules()

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
        self.RECURSIVE = []
        for line in lines:

            # split into key and productions
            rule, prod = line.split('::=')

            self.GRAMMAR[rule.strip()] = []
            # for every option in productions
            for i, p in enumerate(prod.split('|')):
                #print('##', rule.strip(), '#', p.strip(), rule.strip() in p.strip())
                # if option contains the rule, its recursive
                if rule.strip() in p.strip():
                    self.RECURSIVE.append(p.strip())
                self.GRAMMAR[rule.strip()].append([])
                for value in p.strip().split(' '):
                    value = self._parse_value(value)
                    self.GRAMMAR[rule.strip()][i].append(value)

            self.NT.append(rule.strip())

    def _parse_value(self, value):
        try:
            value = value.replace(' ', '')
            return float(value) if '.' in value else int(value)
        except:
            return value

    def _recursive_parse_call(self, genotype, added, symb, depth):

        prod = []
       
        if genotype[self.NT.index(symb)] == []:
            value = np.random.randint(0, len(self.GRAMMAR[symb]))
            added[self.NT.index(symb)].append(value)
            genotype[self.NT.index(symb)].append(value)

            if self.verbose:
                print('[parse] not enough values. Adding:', value, 'to', symb)

        value = genotype[self.NT.index(symb)].pop(0)
        expansion = self.GRAMMAR[symb][value]
        
        for s in expansion:
            if s not in self.NT:
                prod.append(s)
            else:
                prod += self._recursive_parse_call(genotype, added, s, depth+1)

        return prod

    def dsge_create_solution(self, max_depth=10, genotype=None, symb=None, depth=0):

        prod = []

        if symb is None:
            gen = [[] for _ in range(len(self.NT))]
            symb = self.NT[0] # assigns initial symbol
        else:
            gen = genotype

        value = np.random.randint(0, len(self.GRAMMAR[symb]))
        expansion = self.GRAMMAR[symb][value]

        # if expansion is recursive, pick another
        if depth > max_depth and symb in expansion:
            #print('max depth reached and next expansion is recursive')
            #print(depth, symb, expansion)
            while symb in expansion:
                value = np.random.randint(0, len(self.GRAMMAR[symb]))
                expansion = self.GRAMMAR[symb][value]
            #print('changed to:', symb, expansion)

        gen[self.NT.index(symb)].append(value)

        for s in expansion:
            if s in self.NT:
                self.dsge_create_solution(max_depth, gen, s, depth+1)

        #print(depth)

        return gen

    def dsge_recursive_parse(self, genotype):

        '''falta incorporar comportamento para quando ultrapassa 
        profundidade maxima'''

        gen_cpy = [g[:] for g in genotype]
        added = [[] for _ in range(len(self.NT))]
        symb = self.NT[0] # assigns initial symbol

        prod = self._recursive_parse_call(
            genotype=gen_cpy, added=added, symb=symb, depth=0)

        if self.verbose:
            print('pro', prod)
            print('gen', genotype)
            print('rem', gen_cpy)
            print('add', added)

        # removes excess values and adds missing values
        for symb in range(len(genotype)):
            for _ in range(len(gen_cpy[symb])):
                genotype[symb].pop()
            genotype[symb].extend(added[symb])

        return list(filter(lambda x: x != '&', prod)), genotype

if __name__ == '__main__':
    
    #np.random.seed(0)

    grammar = BNFGrammar('cnn2.bnf')
    solution = grammar.dsge_create_solution()
    #print(solution)
    #print(grammar.dsge_recursive_parse(solution))

    grammar = BNFGrammar('unet_mirror2.bnf')
    solution = grammar.dsge_create_solution()
    #print(solution)
    #print(grammar.dsge_recursive_parse(solution))

    solution = [[0],[1,1,0,2,3,3,2],[],[],[0,0,0,0,0,0,0],[0,0,0,0],[0,0,0],[2,2,4,0],[2,1,4,6],[],[],[2,0,2,2],[0,0,0]]
    print(solution)
    mapping, solution = grammar.dsge_recursive_parse(solution)
    print(solution)