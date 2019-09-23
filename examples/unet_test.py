import sys; sys.path.append('..')  # workarround

from grammars import BNFGrammar
from problems import ImageSegmentationProblem

parser = BNFGrammar('../grammars/unet.bnf')
problem = ImageSegmentationProblem(parser)