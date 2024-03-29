import os

import pytest
import numpy as np

from cbioge.grammars import Grammar

def get_mockup_parser():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, 'assets', 'test_grammar.json')

def test_reading_not_valid_grammar():
    with pytest.raises(FileNotFoundError):
        Grammar("cbioge/assets/grammars/not_valid_grammar.json")

def test_reading_grammar_attributes_name():
    grammar = Grammar(get_mockup_parser())
    assert grammar.name == 'test_grammar'

def test_reading_grammar_attributes_rules():
    grammar = Grammar(get_mockup_parser())
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
    grammar = Grammar(get_mockup_parser())
    assert grammar.nonterm == [
        '<start>', '<conv>', '<dense>', '<ksize>', '<filters>', '<units>']

def test_create_solution():
    np.random.seed(0)
    grammar = Grammar(get_mockup_parser())
    genotype = grammar.create_solution()
    print(genotype)
    assert genotype == [[4], [1, 1], [2], [1, 0], [0, 0], [0]]

def test_create_solution_with_max_depth():
    np.random.seed(312)
    grammar = Grammar(get_mockup_parser(), verbose=True)
    genotype = grammar.create_solution(max_depth=2)
    assert genotype == [[0, 4, 3], [2, 2, 1], [2, 0], [0, 0, 1], [1, 0, 0], [0, 1]]

def test_recursive_parse():
    np.random.seed(0)
    grammar = Grammar(get_mockup_parser())
    genotype = [[4], [0, 0], [0], [1, 1], [1, 0], [1]]

    assert grammar.recursive_parse(genotype) == ['conv', 32, 3, 'conv', 16, 3, 'dense', 64]

def test_create_solution_adding_values():
    np.random.seed(0)
    grammar = Grammar(get_mockup_parser(), True)

    original = [[4], [0], [0], [1], [1], [1]]
    solution = [[4], [0], [0], [1], [1], [1]]
    expected = [[4], [0, 0], [0], [1, 1], [1, 1], [1]]

    mapping = ['conv', 32, 3, 'conv', 32, 3, 'dense', 64]

    assert grammar.recursive_parse(solution) == mapping
    assert original != solution
    assert original != expected
    assert solution == expected

def test_create_solution_removing_values():
    np.random.seed(0)
    grammar = Grammar(get_mockup_parser())

    original = [[4, 0, 0, 0], [0, 0], [0], [1, 1], [1, 0], [1]]
    solution = [[4, 0, 0, 0], [0, 0], [0], [1, 1], [1, 0], [1]]
    expected = [[4], [0, 0], [0], [1, 1], [1, 0], [1]]

    assert grammar.recursive_parse(solution) == ['conv', 32, 3, 'conv', 16, 3, 'dense', 64]
    assert original != solution
    assert original != expected
    assert solution == expected
