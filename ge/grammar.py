import numpy as np
import re
import sys

MAX_LOOPS = 1000
grammar = None


def load_grammar(file):

	global grammar

	lines = []
	with open(file, 'r') as gf:

		for line in gf:

			# remove spaces and '\n'
			line = re.sub('\\s+|\n', '', line)

			# if line does not match 'new rule line', append to previous
			if re.match('<[a-z_]+>::=', line) == None:
				lines[len(lines)-1] += line

			elif line != '':
				lines.append(line)

	grammar = {'<start>': None}
	for line in lines:

		# split into key and productions
		rule, prod = line.split('::=')

		# split productions in options
		grammar[rule] = prod.split('|')

		if grammar['<start>'] == None: grammar['<start>'] = rule


def parse(ind):
	index = 0
	loop_count = 0
	match = 0

	prod = grammar['<start>']
	while match != None:
		match = re.search('<[a-z_]+>', prod)
		if match != None:
			token = match.group(0)
			repl = ind[index] % len(grammar[token])
			prod = prod.replace(token, grammar[token][repl], 1)
			index += 1
			if index >= len(ind): index = 0
		loop_count += 1
		if loop_count >= MAX_LOOPS:
			print('infinite loop')
			return None

	prod = prod.replace('\'\'', '@')\
		.replace('\'', '') \
		.split('@')

	print('prod', prod)
	prod = list(filter(lambda x: x != '&', prod))

	return prod


if __name__ == '__main__':
	
	#test

	rand = np.random

	load_grammar(sys.argv[1])

	gen = rand.randint(0, 255, rand.randint(1, 10))
	fen = parse(gen)

	print(gen)
	print(fen)
