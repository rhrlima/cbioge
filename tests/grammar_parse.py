import sys, os
sys.path.append('..')

from grammars import BNFGrammar

g = BNFGrammar('../grammars/cnn.bnf')
print(g.parse([1,2,2]))