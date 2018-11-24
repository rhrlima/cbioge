import numpy as np
import re
import sys


class BNFGrammar:

	GRAMMAR = None
	MAX_LOOPS = 1000


	def __init__(self, grammar_file):

		lines = []
		with open(grammar_file, 'r') as gf:

			for line in gf:

				# remove spaces and '\n'
				line = re.sub('\\s+|\n', '', line)

				# if line does not match 'new rule line', append to previous
				if re.match('<[a-z_]+>::=', line) == None:
					lines[len(lines)-1] += line

				elif line != '':
					lines.append(line)

		self.GRAMMAR = {'<start>': None}
		for line in lines:

			# split into key and productions
			rule, prod = line.split('::=')

			# split productions in options
			self.GRAMMAR[rule] = prod.split('|')

			if self.GRAMMAR['<start>'] == None: self.GRAMMAR['<start>'] = rule


	def parse(self, codons):
		index = 0
		loop_count = 0
		match = 0

		prod = self.GRAMMAR['<start>']
		while match != None:
			match = re.search('<[a-z_]+>', prod)
			if match != None:
				token = match.group(0)
				repl = codons[index] % len(self.GRAMMAR[token])
				prod = prod.replace(token, self.GRAMMAR[token][repl], 1)
				index += 1
				if index >= len(codons): index = 0
			
			loop_count += 1

			# maybe an infinite loop
			if loop_count >= self.MAX_LOOPS: 
				return None

		prod = prod.replace('\'\'', '@')\
			.replace('\'', '') \
			.split('@')

		return list(filter(lambda x: x != '&', prod))