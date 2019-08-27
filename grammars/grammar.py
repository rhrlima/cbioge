import re


class BNFGrammar:

    GRAMMAR = None
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

        self.GRAMMAR = {'<start>': None}
        for line in lines:

            # split into key and productions
            rule, prod = line.split('::=')
            self.GRAMMAR[rule.strip()] = [p.strip() for p in prod.split('|')]

            if self.GRAMMAR['<start>'] is None:
                self.GRAMMAR['<start>'] = rule

    def parse(self, codons):
        index = 0
        loop_count = 0
        match = 0

        prod = self.GRAMMAR['<start>']
        while match is not None:
            match = re.search('<[a-z_]+>', prod)
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
