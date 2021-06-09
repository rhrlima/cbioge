from cbioge.gcnevolution import GCNGEEvolution
import os

#default data
dataset_path='tests/dataset/test'
dataset='mr'
output='tests/test_output'
grammar='tests/grammars/default_gcn.json'

gcnge = GCNGEEvolution()

runner = gcnge.create_runner(dataset, dataset_path, output, grammar)

def test_create_default_runner():
    assert runner!=None
    assert runner.check()

def test_output_build():
    runner.make_folders()
    assert os.path.exists(output)

def test_build_grammar():
    grammar_parser = runner.build_grammar()
    assert grammar_parser !=None

def test_build_problem():
    runner = gcnge.create_runner(dataset, dataset_path, output, grammar)
    problem = runner.build_problem()
    assert problem != None

def test_build_selection_method():
    selection = runner.build_selection()
    assert selection != None

def test_build_crossover():
    crossover = runner.build_crossover()
    assert crossover != None

def test_build_mutation():
    mutation = runner.build_mutation()
    assert mutation != None

def test_build_replacement():
    replacement = runner.build_replacement()
    assert replacement != None    

def test_build_grammarEvolution():
    ge = runner.build_grammarEvolution()
    assert ge != None
    assert ge.selection != None
    assert ge.crossover != None
    assert ge.mutation != None
    assert ge.replacement != None
    assert ge.pop_size > 0
    assert ge.max_evals > 0





