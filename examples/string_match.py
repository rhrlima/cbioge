import sys; sys.path.append('..')

from algorithms import GrammaticalEvolution
from algorithms import TournamentSelection, OnePointCrossover, PointMutation
from algorithms import ReplaceWorst, GEPrune, GEDuplication

from grammars import BNFGrammar

from problems import StringMatchProblem


if __name__ == '__main__':

    # read grammar and setup parser
    parser = BNFGrammar('../grammars/string.bnf')

    # problem parameters
    problem = StringMatchProblem(parser)

    # algorithm
    alg = GrammaticalEvolution(problem)
    alg.maximize = False
    alg.pop_size = 100
    alg.max_evals = 50000
    alg.min_genes = 10
    alg.max_genes = 100
    alg.selection = TournamentSelection(t_size=5)
    alg.crossover = OnePointCrossover(cross_rate=0.75)
    alg.mutation = PointMutation(mut_rate=0.5, min_value=0, max_value=255)
    alg.prune = GEPrune(prun_rate=0.1)
    alg.duplication = GEDuplication(dupl_rate=0.1)
    alg.replacement = ReplaceWorst()

    print('--running--')
    best = alg.execute()

    print('--best solution--')
    if best:
        print(best.fitness, best)
        print(best.phenotype)
    else:
        print('None solution')
