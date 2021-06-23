from cbioge.gcnevolution import GCNGEEvolution
import os

#default data
dataset_path='tests/dataset/test'
dataset='mr'
output='tests/test_output'
grammar='tests/grammars/default_gcn.json'

gcnge = GCNGEEvolution()

runner = gcnge.create_runner(dataset, dataset_path, output, grammar)

def test_build_problem():
    grammar_parser = runner.build()
    assert grammar_parser !=None
