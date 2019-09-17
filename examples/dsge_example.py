# REMOVE FUTURE WARNINGS
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import sys; sys.path.append('..')  # workarround
import os
import random

from algorithms import DSGECrossover, DSGEMutation
from algorithms.solutions import GESolution
# from algorithms import GEPrune, GEDuplication, GrammaticalEvolution
from grammars import BNFGrammar
from problems import DNNProblem
# from utils import checkpoint


if __name__ == '__main__':

    parser = BNFGrammar('../grammars/reg2.bnf')
    # problem = DNNProblem(parser, '../datasets/mnist.pickle')

    s = [[0], [1, 0], [2, 0, 3], [1, 1, 0, 0]]
    p = parser.dsge_recursive_parse(s)
    assert ''.join(p) == '(0.5/0.5)+x*x', 'Map error'
    print(s, p)
    
    parser = BNFGrammar('../grammars/cnn.bnf')

    print('creating random population')
    pop = []
    for _ in range(20):
        s = parser.dsge_create_solution(max_depth=0)
        p = parser.dsge_recursive_parse(s)
        print(s, p)
        obj = GESolution(s)
        obj.phenotype = p
        pop.append(obj)
    print()

    print('applying crossover')
    for _ in range(10):
        cross = DSGECrossover(cross_rate=0.9)
        s1 = random.choice(pop)
        s2 = random.choice(pop)
        print('parent1', s1, s1.phenotype)
        print('parent2', s2, s2.phenotype)
        off = cross.execute([s1, s2])
        print('off1', off[0], parser.dsge_recursive_parse(off[0].genotype))
        print('off2', off[1], parser.dsge_recursive_parse(off[1].genotype))
        print()
    print()

    print('applying mutation')
    mut = DSGEMutation(mut_rate=0.1, parser=parser)
    for i, s in enumerate(pop):
        print('from', i, s, s.phenotype)
        mut.execute(s)
        p = parser.dsge_recursive_parse(s.genotype)
        print('to  ', i, s, p)