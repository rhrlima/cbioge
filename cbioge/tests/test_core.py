from cbioge.gcnevolution import GCNGEEvolution
from cbioge.multipleEvolution import MultipleEvolution
import os
from cbioge.problems.multipleProblem import MultipleProblem

#default data
dataset_path='tests/dataset/test'
dataset='test'
output='tests/test_output'
grammar='tests/grammars/default_gcn.json'

gcnge = GCNGEEvolution()

runner = gcnge.create_runner(dataset, dataset_path, output, grammar)

def test_build_problem():
    grammar_parser = runner.build()
    assert grammar_parser !=None

def test_multiple_grammar():
    grammar='tests/grammars/multiple_layers.json'
    me = MultipleEvolution(
        grammar=grammar, 
        dataset_path=dataset_path, 
        dataset=dataset, 
        output=output, 
        problem=MultipleProblem)
    runner = me.runner
    assert runner != None


    assert runner.parser.name == 'multiple_layers'

    assert len(runner.parser.blocks) == 7

    assert runner.problem != None

    assert type(runner.problem) != type(MultipleProblem)

    assert runner.parser != None

    assert runner.algorithm != None

    result = runner.execute()

    assert result != 'SUCCESS'


