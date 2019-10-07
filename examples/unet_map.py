from problems import UNetProblem
from grammars import BNFGrammar


if __name__ == '__main__':

	parser = BNFGrammar('grammars/unet.bnf')
	problem = UNetProblem(parser, None)

	gen = parser.dsge_create_solution()
	fen = problem.map_genotype_to_phenotype(gen)
	#print(gen)
	#print(fen)