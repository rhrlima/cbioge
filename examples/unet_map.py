from problems import UNetProblem
from grammars import BNFGrammar


if __name__ == '__main__':

	dset = {
		'input_shape': (8, 8, 1)
	}

	parser = BNFGrammar('grammars/unet.bnf')
	problem = UNetProblem(parser, dset)

	#for _ in range(10):
	gen = parser.dsge_create_solution()
	print(gen)
	fen = parser.dsge_recursive_parse(gen)
	problem._map_genotype_to_phenotype(gen)