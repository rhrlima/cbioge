import ge

def one_gene_operators():
	ge.DEBUG = True
	ge.POP_SIZE = 100
	ge.CROSS_RATE = 1.0
	ge.MUT_RATE = 1.0
	ge.PRUN_RATE = 1.0
	ge.DUPL_RATE = 1.0
	
	ge.execute()


if __name__ == '__main__':
	
	one_gene_operators()