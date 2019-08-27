import sys; sys.path.append('..')  # workarround
import os

# from algorithms import TournamentSelection, OnePointCrossover, PointMutation
# from algorithms import GEPrune, GEDuplication, GrammaticalEvolution
from grammars import BNFGrammar
# from problems import DNNProblem
# from utils import checkpoint


# disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    parser = BNFGrammar('../grammars/cnn.bnf')
    # problem = DNNProblem(parser, '../datasets/c10.pickle')

    # s1 = [[0], [1, 0], [2, 0, 3], [1, 1, 0, 0]]
    # p1 = parser.dsge_parse(s1)
    # print(s1)
    # print(p1)

    for _ in range(20):
        s = parser.create_random_derivation()
        p = parser.dsge_parse(s)
        print(s, p)

    # parses the arguments
    # args = get_arg_parsersed()

    # # checkpoint folder
    # checkpoint.ckpt_folder = args.folder

    # # read grammar and setup parser
    # parser = BNFGrammar(args.grammar)

    # # problem dataset and parameters
    # problem = CnnProblem(parser, args.dataset)
    # problem.batch_size = args.batch
    # problem.epochs = args.epochs
    # problem.x_train = problem.x_train[:1000]
    # problem.y_train = problem.y_train[:1000]
    # problem.x_valid = problem.x_valid[:1000]
    # problem.y_valid = problem.y_valid[:1000]

    # # genetic operators to GE
    # selection = TournamentSelection(maximize=True)
    # crossover = OnePointCrossover(cross_rate=args.crossover)
    # mutation = PointMutation(mut_rate=args.mutation,
    #                          min_value=0, max_value=255)
    # prune = GEPrune(prun_rate=args.prune)
    # duplication = GEDuplication(dupl_rate=args.duplication)

    # # changing ge default parameters
    # algorithm = GrammaticalEvolution(problem)
    # algorithm.pop_size = args.population
    # algorithm.max_evals = args.evals
    # algorithm.max_processes = args.maxprocesses
    # algorithm.selection = selection
    # algorithm.crossover = crossover
    # algorithm.mutation = mutation
    # algorithm.prune = prune
    # algorithm.duplication = duplication
    # algorithm.verbose = args.verbose

    # print('--config--')
    # print('DATASET', args.dataset)
    # print('GRAMMAR', args.grammar)
    # print('EPOCHS', args.epochs)
    # print('BATCH', args.batch)
    # print('CKPT', args.folder, args.checkpoint)

    # print('POP', args.population)
    # print('EVALS', args.evals)

    # print('--running--')
    # best = algorithm.execute(args.checkpoint)

    # print('--best solution--')
    # if best:
    #     print(best.fitness, best)
    # else:
    #     print('None solution')
