import numpy as np
import re
import sys

MAX_LOOPS = 1000
grammar = None

def load_grammar(file):
	lines = []
	with open(grammar_file, 'r') as gf:
		for line in gf:
			line = re.sub('\\s+|\n', '', line) #remove spaces and '\n'
			if re.match('<[a-z_]+>::=', line) == None:
				lines[len(lines)-1] += line
			elif line != '':
				lines.append(line)

	global grammar
	grammar = {'<start>': None}
	for line in lines:
		rule, prod = line.split('::=')
		grammar[rule] = prod.split('|')
		if grammar['<start>'] == None: grammar['<start>'] = rule
	print(grammar)


def parse(ind):
	index = 0
	loop_count = 0
	match= 0

	prod = grammar['<start>']
	print('start', prod)
	while match != None:
		match= re.search('<[a-z_]+>', prod)
		#print('match', match)
		if match != None:
			token = match.group(0)
			#print('token', token)
			repl = ind[index] % len(grammar[token])
			#print('index', index, 'ind', ind[index], 'size', len(grammar[token]))
			prod = prod.replace(token, grammar[token][repl], 1)
			print('{} --> {}'.format(token, grammar[token][repl]))
			index += 1
			if index >= len(ind): index = 0
		print(prod)
		loop_count += 1
		if loop_count >= MAX_LOOPS:
			print('invalid')
			return None
	return prod


if __name__ == '__main__':
	grammar_file = sys.argv[1]
	
	rand = np.random
	ind = rand.randint(0, 255, rand.randint(1, 10))
		
	load_grammar(grammar_file)
	phen = parse(ind)
	print(phen \
		.replace('\"\"', '@') \
		.replace('\"', '') \
		.split('@')
	)