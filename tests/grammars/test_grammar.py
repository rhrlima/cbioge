import pytest

import numpy as np

from cbioge.grammars import Grammar

def test_reading_not_valid_grammar():
    with pytest.raises(FileNotFoundError):
        grammar = Grammar("tests/grammars/not_valid_grammar.json")

def test_reading_grammar_attributes_name():
    grammar = Grammar("tests/grammars/test_grammar.json")
    assert grammar.name == 'test_grammar'

def test_reading_grammar_attributes_blocks():
    grammar = Grammar("tests/grammars/test_grammar.json")    
    assert grammar.blocks == {
		"conv": ["Conv2D", "filters", "padding"],
        "dense": ["Dense", "units"]
	}

def test_reading_grammar_attributes_rules():
    grammar = Grammar("tests/grammars/test_grammar.json")    
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
            ["conv", "<filters>", "<padding>"]
        ],
		"<dense>" : [
            ["dense", "<units>"]
        ],
		"<padding>" : [
            ["valid"], 
            ["same"]
        ],
		"<filters>" : [
            [16], 
            [32]
        ],
		"<units>" : [
            [32], 
            [64]
        ]
	}

def test_reading_grammar_attributes_nonterm():
    grammar = Grammar("tests/grammars/test_grammar.json")
    assert grammar.nonterm == [
        '<start>', '<conv>', '<dense>', '<padding>', '<filters>', '<units>']

def test_dsge_create_solution():
    np.random.seed(0)
    grammar = Grammar('tests/grammars/test_grammar.json')
    assert grammar.dsge_create_solution() == [[4], [0, 0], [0], [1, 1], [1, 0], [1]]

def test_dsge_create_solution_with_max_depth():
    np.random.seed(48)
    grammar = Grammar('tests/grammars/test_grammar.json', True)
    assert grammar.dsge_create_solution(max_depth=2) == [
        [0, 3, 0, 0, 2, 0, 4, 3, 0, 2, 5], 
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 1, 0], 
        [1, 1, 0, 0, 0], 
        [1, 0, 0, 0, 0, 1, 1]]

def test_dsge_recursive_parse():
    np.random.seed(0)
    grammar = Grammar('tests/grammars/test_grammar.json')
    solution = [[4], [0, 0], [0], [1, 1], [1, 0], [1]]
    assert grammar.dsge_recursive_parse(solution) == (
        ['conv', 32, 'same', 'conv', 16, 'same', 'dense', 64],
        [[4], [0, 0], [0], [1, 1], [1, 0], [1]])

def test_dsge_create_solution_adding_values():
    np.random.seed(0)
    grammar = Grammar('tests/grammars/test_grammar.json', True)
    solution = [[4], [0], [0], [1, 1], [1, 0], [1]]
    assert grammar.dsge_recursive_parse(solution) == (
        ['conv', 32, 'same', 'conv', 16, 'same', 'dense', 64],
        [[4], [0, 0], [0], [1, 1], [1, 0], [1]]
    )

def test_dsge_create_solution_removing_values():
    np.random.seed(0)
    grammar = Grammar('tests/grammars/test_grammar.json')
    solution = [[4, 0, 0, 0], [0, 0], [0], [1, 1], [1, 0], [1]]
    assert grammar.dsge_recursive_parse(solution) == (
        ['conv', 32, 'same', 'conv', 16, 'same', 'dense', 64],
        [[4], [0, 0], [0], [1, 1], [1, 0], [1]]
    )