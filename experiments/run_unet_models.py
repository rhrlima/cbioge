from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from algorithms.solutions import GESolution
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from examples.unet_model import *

import matplotlib.pyplot as plt


# Plot a line based on the x and y axis value list.
def draw_line(x_values, y_values):

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
    plt.savefig('runs.png')
    
    # Display the plot in the matplotlib's viewer.
    plt.show()


if __name__ == '__main__':
    
    dset_args = {
        "path": "datasets/membrane",
        "train_path": "datasets/membrane/train_posproc",
        "test_path": "datasets/membrane/test_posproc",
        "input_shape": (256, 256, 1),
        "train_ids": [f'{i}.png' for i in range(30)],
        "valid_ids": [],
        "test_ids": [f'{i}.png' for i in range(30)],
        "train_steps": 300,
        "test_steps": 30,
        "aug": dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')
    }

    data_aug = ImageDataGenerator(**dset_args['aug'])
    train_gen = DataGenerator(
        dset_args['train_path'], 
        dset_args['train_ids'], 
        dset_args['input_shape'], 
        batch_size=2, 
        data_aug=data_aug)
    test_gen = DataGenerator(
        dset_args['test_path'], 
        dset_args['test_ids'], 
        dset_args['input_shape'], 
        batch_size=1, 
        shuffle=False)

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser, dset_args)
    problem.train_generator = train_gen
    problem.test_generator = test_gen
    problem.verbose = True

    population = []

    # for _ in range(8):
    #     solution = GESolution(parser.dsge_create_solution())
    #     solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
    #     population.append(solution)

    #UNET from genotype
    solution = GESolution([[0], [0, 3, 0, 3, 0, 3, 0, 1, 3], [0, 0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [], [1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]])
    solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
    population.append(solution)

    #original UNET
    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()
    population.append(solution)

    accs = []
    for s in population:
        model = model_from_json(s.phenotype)
        model.summary()
        acc = .5#problem.evaluate(s.phenotype)
        print(acc)
        accs.append(acc)

    draw_line(range(len(accs)), accs)
