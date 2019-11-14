import os
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

from examples.unet_model import *

import matplotlib.pyplot as plt


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('name', type=str) #name
    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-trs', '--train', type=int, default=5) #train steps
    args.add_argument('-tes', '--test', type=int, default=5) #test steos
    args.add_argument('-aug', '--augment', type=int, default=0) #augmentation
    args.add_argument('-p', '--predict', type=int, default=0) #predict
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-s', '--shuffle', type=int, default=0) #shuffle
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

    args = get_args()

    print(args)

    dset_args = json.loads(open(args.dataset, 'r').read())
    dset_args['train_steps'] = args.train
    dset_args['test_steps'] = args.test

    train_gen = DataGenerator(dset_args['train_path'], dset_args['input_shape'], batch_size=args.batch, shuffle=args.shuffle)
    test_gen = DataGenerator(dset_args['test_path'], dset_args['input_shape'], batch_size=args.batch, shuffle=args.shuffle)

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser, dset_args, train_gen, test_gen)
    problem.verbose = args.verbose
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    population = []

    for _ in range(8):
        solution = GESolution(parser.dsge_create_solution())
        print(solution)
        solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
        population.append(solution)

    #UNET from genotype
    solution = GESolution([[0], [0, 3, 0, 3, 0, 3, 0, 1, 3], [0, 0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [], [1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]])
    solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
    population.append(solution)

    #original UNET
    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()
    population.append(solution)

    print(args)

    accs = []
    for s in population:
        model = model_from_json(s.phenotype)
        #model.summary()
        loss, acc = problem.evaluate(s.phenotype, predict=args.predict)
        print(loss, acc)
        accs.append(acc)

    draw_line(args.name, range(len(accs)), accs)
