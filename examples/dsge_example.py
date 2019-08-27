import sys; sys.path.append('..')  # workarround
import os

# from algorithms import TournamentSelection, OnePointCrossover, PointMutation
# from algorithms import GEPrune, GEDuplication, GrammaticalEvolution
from grammars import BNFGrammar
from problems import DNNProblem
# from utils import checkpoint


# disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    parser = BNFGrammar('../grammars/cnn.bnf')
    problem = DNNProblem(parser, '../datasets/c10.pickle')

    # s1 = [[0], [1, 0], [2, 0, 3], [1, 1, 0, 0]]
    # p1 = parser.dsge_parse(s1)
    # print(s1)
    # print(p1)

    for _ in range(20):
        s = parser.create_random_derivation()
        # p = parser.dsge_parse(s)
        # print(s, p)
        p = problem.map_genotype_to_phenotype(s)
        print(p is None)
