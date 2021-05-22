import json
import numpy as np


class Grammar:

    def __init__(self, grammar_file):
        self._read_grammar(grammar_file)
        self.verbose = 0
    
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
                print('[parse] not enough values. Adding:', value, 'to', symb)

        value = genotype[self.nonterm.index(symb)].pop(0)
        expansion = self.rules[symb][value]
        
        for s in expansion:
            if s not in self.nonterm:
                production.append(s)
            else:
                production += self._recursive_parse_call(genotype, added, s, depth+1)

        return production

    def dsge_create_solution(self, max_depth=10, genotype=None, symb=None, depth=0):
        ''' creates a random solution based on the grammar, according to the 
            DSGE method

            returns: a list of lists containing integer values that is a valid
            solution related to the grammar

            TODO adicionar chamada de função interna pra remover os parametros 
            desnecessarios do metodo principal
        '''

        # prod = []

        if symb is None:
            gen = [[] for _ in range(len(self.nonterm))]
            symb = self.nonterm[0] # assigns initial symbol
        else:
            gen = genotype

        value = np.random.randint(0, len(self.rules[symb]))
        expansion = self.rules[symb][value]

        # if expansion is recursive, pick another option
        if depth > max_depth and symb in expansion:
            #print('max depth reached and next expansion is recursive')
            #print(depth, symb, expansion)
            while symb in expansion:
                value = np.random.randint(0, len(self.rules[symb]))
                expansion = self.rules[symb][value]
            #print('changed to:', symb, expansion)

        gen[self.nonterm.index(symb)].append(value)

        for s in expansion:
            if s in self.nonterm:
                self.dsge_create_solution(max_depth, gen, s, depth+1)

        #print(depth)

        return gen

    def dsge_recursive_parse(self, genotype):
        ''' performs the initial call for the grammar expansion according to the
            DSGE method

            it will perform a grammar expansion according to the grammar
            using the values present in the genotype

            it will modify the genotype as needed, adding more values if
            requested, and removing values if they are not used

            returns: the final production, the modified genotype

            TODO falta incorporar comportamento para quando ultrapassa 
            profundidade maxima. Não tem atrapalhado até agora.
        '''

        gen_cpy = [g[:] for g in genotype]
        added = [[] for _ in range(len(self.nonterm))]
        symb = self.nonterm[0]

        production = self._recursive_parse_call(genotype=gen_cpy, added=added, 
            symb=symb, depth=0)

        if self.verbose:
            print('-----')
            print('production', production)
            print('genetype', genotype)
            print('remover', gen_cpy)
            print('added', added)

        for symb in range(len(genotype)):
            # removes the values left off during the expansion, from the original genotype
            for _ in range(len(gen_cpy[symb])):
                genotype[symb].pop()
            # adds the values present in the 'added' list, to the original genotype
            genotype[symb].extend(added[symb])

        return list(filter(lambda x: x != '&', production)), genotype