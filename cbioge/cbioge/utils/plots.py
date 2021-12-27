import os
import pickle

import matplotlib.pyplot as plt

from . import checkpoint as ckpt


def _read_data_from_checkpoint(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def _get_best_from_generation(data, maximize=False):
    best_f = max if maximize else min
    return best_f(data, key=lambda s: s['fitness'])


def _get_generations_data(folder_name, sort=True):

    generations = []

    files = ckpt.get_files_with_name('data_*.ckpt', folder_name)

    if sort:
        files.sort(key=lambda f: ckpt.natural_key(f))

    print(files)
    for f in files:
        data = _read_data_from_checkpoint(os.path.join(folder_name, f))
        generations.append(data['population'])

    return generations


def plot_evolution(ckpt_folder, mode='max'):
    '''plots the evolution of one or multiple checkpoint folders

    # Arguments
    ckpt_folder: str with the path to the folder or list of folder names
    mode: min | max | avg (default max) will plot the values according to the mode'''

    if isinstance(ckpt_folder, str): # one folder
        ckpt_folder = [ckpt_folder]

    for folder in ckpt_folder:
        generations = _get_generations_data(folder)
        y_values = [_get_best_from_generation(gen, mode)['fitness'] for gen in generations]
        plt.plot(range(len(generations)), y_values, label=folder)

    plt.legend()
    plt.show()


def botplot_generation(ckpt_folder, filter_invalid=True, invalid_value=-1):
    generations = _get_generations_data(ckpt_folder)

    boxes = []
    for gen in generations:
        values = [s['fitness'] for s in gen]
        if filter_invalid:
            values = list(filter(lambda v: v != invalid_value, values))
        boxes.append(values)

    plt.boxplot(boxes)
    plt.show()


def print_checkpoint_fitness(file_name):

    data = _read_data_from_checkpoint(file_name)

    population = data['population']
    print('pop size', len(population))

    population.sort(key=lambda s: s['fitness'])
    for s in population:
        print(s['id'], s['fitness'])
