import pytest

import numpy as np

from cbioge.grammars import Grammar

def test_reading_not_valid_grammar():
    with pytest.raises(FileNotFoundError):
        Grammar("not_valid_grammar.json")

def test_reading_grammar_attributes_name():
    grammar = Grammar('tests/data/test_grammar.json')
    assert grammar.name == 'test_grammar'

def test_reading_grammar_attributes_blocks():
    grammar = Grammar('tests/data/test_grammar.json')
    assert grammar.blocks == {
        "conv": ["Conv2D", "filters", "kernel_size"], 
        "dense": ["Dense", "units"]
        }

def test_reading_grammar_attributes_rules():
    grammar = Grammar('tests/data/test_grammar.json')
    assert grammar.rules == {
        "<start>": [
            ["<start>", "<start>"], 
            ["<conv>"], 
            ["<dense>"], 
            ["<conv>", "<dense>"], 
            ["<conv>", "<conv>", "<dense>"], 
            ["<conv>", "<dense>", "<dense>"]
        ],
        "<conv>" : [
            ["conv", "<filters>", "<ksize>"], 
            ["conv", "<filters>", "<ksize>"], 
            ["conv", "<filters>", "<ksize>"]
        ],
        "<dense>" : [
            ["dense", "<units>"], 
            ["dense", "<units>"], 
            ["dense", "<units>"]
        ],
        "<ksize>" : [ [2], [3], [4]],
        "<filters>" : [ [16], [32] ],
        "<units>" : [ [32], [64] ]}

def test_reading_grammar_attributes_nonterm():
    grammar = Grammar('tests/data/test_grammar.json')
    assert grammar.nonterm == [
        '<start>', '<conv>', '<dense>', '<ksize>', '<filters>', '<units>']

def test_group_mapping():
    parser = Grammar('tests/data/test_grammar.json')
    mapping = ['conv', 32, 'same', 'conv', 32, 'same', 'dense', 10]
    assert parser._group_mapping(mapping) == [
        ['conv', 32, 'same'], 
        ['conv', 32, 'same'], 
        ['dense', 10]
    ]

def test_dsge_create_solution():
    np.random.seed(0)
    grammar = Grammar('tests/data/test_grammar.json')
    genotype = grammar.dsge_create_solution()
    print(genotype)
    assert genotype == [[4], [1, 1], [2], [1, 0], [0, 0], [0]]

def test_dsge_create_solution_with_max_depth():
    np.random.seed(312)
    grammar = Grammar('tests/data/test_grammar.json', verbose=True)
    genotype = grammar.dsge_create_solution(max_depth=2)
    assert genotype == [[0, 4, 3], [2, 2, 1], [2, 0], [0, 0, 1], [1, 0, 0], [0, 1]]

def test_dsge_recursive_parse():
    np.random.seed(0)
    grammar = Grammar('tests/data/test_grammar.json')
    genotype = [[4], [0, 0], [0], [1, 1], [1, 0], [1]]
    assert grammar.dsge_recursive_parse(genotype) == [['conv', 32, 3], ['conv', 16, 3], ['dense', 64]]

def test_dsge_create_solution_adding_values():
    np.random.seed(0)
    grammar = Grammar('tests/data/test_grammar.json', True)
    original = [[4], [0], [0], [1], [1], [1]]
    solution = [[4], [0], [0], [1], [1], [1]]
    expected = [[4], [0, 0], [0], [1, 1], [1, 1], [1]]
    mapping = grammar.dsge_recursive_parse(solution)
    assert mapping == [['conv', 32, 3], ['conv', 32, 3], ['dense', 64]]
    assert original != solution
    assert original != expected
    assert solution == expected

def test_dsge_create_solution_removing_values():
    np.random.seed(0)
    grammar = Grammar('tests/data/test_grammar.json')
    original = [[4, 0, 0, 0], [0, 0], [0], [1, 1], [1, 0], [1]]
    solution = [[4, 0, 0, 0], [0, 0], [0], [1, 1], [1, 0], [1]]
    expected = [[4], [0, 0], [0], [1, 1], [1, 0], [1]]
    assert grammar.dsge_recursive_parse(solution) == ([['conv', 32, 3], ['conv', 16, 3], ['dense', 64]])
    assert original != solution
    assert original != expected
    assert solution == expected