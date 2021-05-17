import sys; sys.path.append('..')

import argparse
from grammars import BNFGrammar
from problems import CnnProblem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='parse_solution.py')
    parser.add_argument('grammar', type=str)
    parser.add_argument('solution', type=str)

    args = parser.parse_args()

    grammar = BNFGrammar(args.grammar)
    problem = CnnProblem(grammar, None)
    solution = [int(s) for s in args.solution.replace(' ', '').split(',')]

    model = problem.map_genotype_to_phenotype(solution)

    if model:
        print('solution', solution)
        print(model)
    else:
        print('invalid')
