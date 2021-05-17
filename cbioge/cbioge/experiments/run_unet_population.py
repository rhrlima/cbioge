import glob
import argparse
import json

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from algorithms.solutions import GESolution
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from utils.model import *

import matplotlib.pyplot as plt


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('name', type=str) #name
    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-tr', '--train', type=int, default=None) #train steps
    args.add_argument('-va', '--valid', type=int, default=None) #valid size
    args.add_argument('-te', '--test', type=int, default=None) #test steos

    args.add_argument('-e', '--epochs', type=int, default=1) #epochs
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-s', '--shuffle', type=int, default=0) #shuffle
    args.add_argument('-p', '--predict', type=int, default=0) #predict
    args.add_argument('-v', '--verbose', type=int, default=1) #verbose

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    return args.parse_args()


# Plot a line based on the x and y axis value list.
def draw_line(name, x_values, y_values):

    # List to hold x values.
    x_number_values = x_values#[1, 2, 3, 4, 5]

    # List to hold y values.
    y_number_values = y_values#[1, 4, 9, 16, 25]

    # Plot the number in the list and set the line thickness.
    #plt.plot(x_number_values, y_number_values, linewidth=3)
    plt.scatter(x_number_values, y_number_values, s=20, edgecolors='none', c='green')

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

    #np.random.seed(0)

    args = get_args()

    print(args)

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser)
    problem.read_dataset_from_pickle(args.dataset)
    
    problem.epochs = args.epochs
    problem.batch_size = args.batch
    problem.verbose = args.verbose
    problem.workers = args.workers
    problem.multiprocessing = args.multip
    
    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    population = []

    # for _ in range(8):
    #     solution = GESolution(parser.dsge_create_solution())
    #     print(solution)
    #     solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
    #     population.append(solution)

    # #UNET from genotype
    # solution = GESolution([[0], [0, 3, 0, 3, 0, 3, 0, 1, 3], [0, 0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [], [1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]])
    # solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
    # population.append(solution)

    # #original UNET
    # solution = GESolution([])
    # solution.phenotype = unet(problem.input_shape).to_json()
    # population.append(solution)

    # best m2nist
    population.append(GESolution([[0], [3, 1, 2, 1, 3, 3, 1, 3], [0, 0, 0, 0, 1], [1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0], [0, 0, 7, 5, 1, 4, 7, 9], [5, 3], [], [], [2, 1], []]))
    population.append(GESolution([[0], [3], [1], [0, 2, 0, 2, 0, 1, 1], [0, 0, 0], [0], [0, 0], [7, 9, 0], [0, 5, 6], [], [], [2, 1, 1], [0, 0]]))
    # best membrane
    population.append(GESolution([[0], [3], [1], [2], [0], [0], [0], [0], [0], [], [], [1], [0]]))
    population.append(GESolution([[0], [0, 3, 3], [0, 1], [2], [0, 0, 0], [0, 0], [0], [0, 8, 8], [3, 1], [], [], [2, 1, 1, 2], []]))

    accs = []
    for s in population:
        s.phenotype = problem.map_genotype_to_phenotype(s.genotype)
        model = model_from_json(s.phenotype)
        #model.summary()
        loss, acc = problem.evaluate(s.phenotype, predict=args.predict)
        print(loss, acc)
        accs.append(acc)

    draw_line(args.name, range(len(accs)), accs)
