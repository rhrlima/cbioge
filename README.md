# CBIOGE

A Grammar-based evolutionary algorithm for the automatic design of Deep Neural Networks

## Description

CBIOGE is a toolkit implemented in Python for the design of deep neural networks using context-free grammars.
The approach is based on the Dynamic Structure Grammatical Evolution (DSGE), a grammar-based evolutionary algorithm, which can be used to design computer programs.

DSGE is defined by three main components:
- A grammar defines the layers and hyperparameters as building blocks of a network, as well as rules that will guide the architecture.
- A mapping procedure that translates one encoded solution using the defined grammar into an actual neural network.
- A search engine that generates and modifies solutions, keeping track of the best ones.

## Main Modules

- Grammar: responsible for reading and parsing the grammars
- Problem: defines how to build and evaluate the networks
- Algorithm: generates, modifies and searches for the best solutions

## Referencing this repo:
````
@inproceedings{lima2021segmentation,
    title = {Automatic Design of Deep Neural Networks Applied to Image Segmentation Problems},
    author = {Ricardo Henrique Remes de Lima
     and Aurora T. R. Pozo
     and Alexander Mendiburu
     and Roberto Santana},
    editor = {Ting Hu
     and Nuno Louren{\c{c}}o
     and Eric Medvet},
    booktitle = {Genetic Programming - 24th European Conference, EuroGP 2021, Held as Part of EvoStar 2021, Virtual Event, April 7-9, 2021, Proceedings},
    series = {Lecture Notes in Computer Science},
    volume = {12691},
    pages = {98--113},
    publisher = {Springer},
    year = {2021},
    doi = {10.1007/978-3-030-72812-0\_7},
}
````