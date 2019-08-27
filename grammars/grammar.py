import copy
import re
import random


class BNFGrammar:

    GRAMMAR = None
    NT = []
    MAX_LOOPS = 1000

    rule_reg = '<[a-zA-Z0-9_-]+>'

    def __init__(self, grammar_file):

        lines = []
        with open(grammar_file, 'r') as gf:

            for line in gf:

                # remove '\n'
                line = re.sub('\n', '', line)

                # if line starts with '#', ignore
                if line.startswith('#') or line == '':
                    continue

                # if line does not match 'new rule line', append to previous
                if re.match(self.rule_reg, line) is None:
                    lines[len(lines)-1] += line
                else:
                    lines.append(line)

        self.GRAMMAR = {}
        self.NT = []
        for line in lines:

            # split into key and productions
            rule, prod = line.split('::=')

            self.GRAMMAR[rule.strip()] = [p.strip() for p in prod.split('|')]
            self.NT.append(rule.strip())

        print(self.GRAMMAR)
        print(self.NT)

    def parse(self, codons):
        index = 0
        loop_count = 0
        match = 0

        prod = self.GRAMMAR['<start>']
        while match is not None:
            match = re.search(self.rule_reg, prod)
            if match is not None:
                token = match.group(0)
                repl = codons[index] % len(self.GRAMMAR[token])
                prod = prod.replace(token, self.GRAMMAR[token][repl], 1)
                index += 1
                if index >= len(codons):
                    index = 0

            loop_count += 1

            # maybe an infinite loop
            if loop_count >= self.MAX_LOOPS:
                return None

        prod = prod.replace('\'\'', '@')\
            .replace('\'', '') \
            .split('@')

        return list(filter(lambda x: x != '&', prod))

    def dsge_parse(self, codons):

        temp = copy.deepcopy(codons)
        match = ''
        prod = self.NT[0]

        while match is not None:
            match = re.search(self.rule_reg, prod)
            if match is not None:
                # print('prod', prod)
                token = match.group(0)
                value = temp[self.NT.index(token)].pop(0)
                replacement = self.GRAMMAR[token][value]
                prod = prod.replace(token, replacement, 1)
                # print('token', token)
                # print('value', value)
                # print('repl', replacement)

        # print('final', prod)

        prod = prod.replace('\' \'', '@')\
            .replace('\'', '') \
            .split('@')

        # print('final', prod)
        # print('codons', codons)

        return list(filter(lambda x: x != '&', prod))

    def create_random_derivation(self):

        codons = [[] for _ in range(len(self.NT))]
        prod = self.NT[0]
        match = ''
        while match is not None:
            match = re.search(self.rule_reg, prod)
            if match is not None:
                # print('prod', prod)
                token = match.group(0)
                # value = codons[self.NT.index(token)].pop(0)
                value = random.randint(0, len(self.GRAMMAR[token])-1)
                replacement = self.GRAMMAR[token][value]
                prod = prod.replace(token, replacement, 1)
                codons[self.NT.index(token)].append(value)
                # print('token', token)
                # print('value', value)
                # print('repl', replacement)

        # print(codons)
        # print(prod)
        return codons
