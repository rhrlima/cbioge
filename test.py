from grammars import BNFGrammar

from algorithms.solution import GESolution
from algorithms.mutation import *
from algorithms.crossover import *
#from problems import UNetProblem


def compare_solutions(sol1, sol2):

	diff = 0
	lenght = 0
	for genes1, genes2 in zip(sol1.genotype, sol2.genotype):
		max_ = max(len(genes1), len(genes2))
		min_ = min(len(genes1), len(genes2))
		for a, b in zip(genes1, genes2):
			print(a, b)
			diff += 1 if a != b else 0
		print("#", max_, min_)
		diff += max_ - min_
		lenght += max_
	return diff / lenght


if __name__ == '__main__':
	
	parser = BNFGrammar('grammars/unet_mirror2.bnf')
	
	s1 = parser.dsge_create_solution()
	s2 = parser.dsge_create_solution()
	#mapping, s2 = parser.dsge_recursive_parse(s1)

	print(len(s1[4]), len(s1[5]), len(s1[6]))
	print(len(s2[4]), len(s2[5]), len(s2[6]))
	#print(s2[4:7])
	#print(mapping)

	# s1 = GESolution(s1)
	# mut1 = DSGERestrictedMutation(1.0, parser, 4)
	# mut2 = DSGEStructuralMutation(1.0, parser, 4)

	# print(s1)
	# mut1.execute(s1)
	# print(s1)
	# mut2.execute(s1)
	# print(s1)

	s1 = GESolution([[0, 0], [0, 0, 0, 0], [0, 0], [0, 0, 0]])
	s2 = GESolution([[9, 9], [9, 9, 9, 9], [9, 9], [9, 9, 9]])
	s3 = GESolution([[0, 0], [0, 0, 0, 0], [0, 0], [0, 0, 0]])
	s4 = GESolution([[0, 0], [0, 0, 0, 0], [0, 0], [9, 0, 0]])
	s5 = GESolution(parser.dsge_create_solution())


	# mut0 = DSGEMutation(1.0, parser)
	# mut1 = DSGETerminalMutation(1.0, parser, 2)
	# mut2 = DSGENonterminalMutation(1.0, parser, 2)

	# cross0 = DSGECrossover(1.0)
	# cross1 = DSGEGeneCrossover(1.0)

	# s3 = cross0.execute([s1, s2])
	# s4 = cross1.execute([s1, s2])

	# print(s1)
	# print(s2)
	# print(s3)
	# print(s4)

	# s2 = s1.copy()
	# m1 = mut0.execute(s1)
	# m2 = mut1.execute(s1)
	# m3 = mut2.execute(s1)
	# m1.genotype[0][0] = 8
	# m2.genotype[0][0] = 9
	# m3.genotype[0][0] = 6

	# print(s1)
	# print(m1)
	# print(m2)
	# print(m3)

	print(compare_solutions(s1, s2))
	print(compare_solutions(s1, s3))
	print(compare_solutions(s1, s4))
	print(compare_solutions(s4, s5))