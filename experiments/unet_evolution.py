import glob
import argparse
import json

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from algorithms.dsge import GrammaticalEvolution
from algorithms import TournamentSelection, DSGECrossover, DSGEMutation, ReplaceWorst
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

import matplotlib.pyplot as plt


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    #args.add_argument('name', type=str) #name
    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-trs', '--train', type=int, default=None) #train steps
    args.add_argument('-va', '--valid', type=int, default=None) #valid size
    args.add_argument('-tes', '--test', type=int, default=None) #test steos

    args.add_argument('-e', '--epochs', type=int, default=1) #epochs
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-s', '--shuffle', type=int, default=0) #shuffle

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    args.add_argument('-ps', '--pop', type=int, default=5) #pop
    args.add_argument('-ev', '--evals', type=int, default=10) #evals

    args.add_argument('-v', '--verbose', type=int, default=1) #verbose (1 - evolution, 2 - problem)

    return args.parse_args()


# Plot a line based on the x and y axis value list.
def draw_line(name, x_values, y_values):

    # Plot the number in the list and set the line thickness.
    plt.scatter(x_values, y_values, s=20, edgecolors='none', c='green')

    # Set the line chart title and the text font size.
    plt.title("Unet model Fitnesses", fontsize=19)

    # Set x axes label.
    plt.xlabel("Models", fontsize=10)

    # Set y axes label.
    plt.ylabel("Fitness", fontsize=10)

    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)

    # Save figure
    plt.savefig(f'{name}.png')
    
    # Display the plot in the matplotlib's viewer.
    #plt.show()

if __name__ == '__main__':

    args = get_args()

    print(args)

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    
    problem = UNetProblem(parser)

    problem.read_dataset_from_pickle(args.dataset)

    problem.verbose = (args.verbose>1) # verbose 2 or higher
    problem.epochs = args.epochs
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    selection = TournamentSelection(t_size=2, maximize=True)
    crossover = DSGECrossover(cross_rate=0.9)
    mutation = DSGEMutation(mut_rate=0.01, parser=parser)
    replace = ReplaceWorst(maximize=True)

    algorithm = GrammaticalEvolution(problem, parser)

    algorithm.pop_size = args.pop
    algorithm.max_evals = args.evals

    algorithm.selection = selection
    algorithm.crossover = crossover
    algorithm.mutation = mutation
    algorithm.replace = replace

    algorithm.verbose = (args.verbose>0) # verbose 1 or higher

    population = algorithm.execute()

    for s in population:
        print(s.fitness, s)
