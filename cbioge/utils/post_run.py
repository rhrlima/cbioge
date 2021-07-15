import os

import numpy as np
import matplotlib.pyplot as plt

from ..algorithms import GESolution
from . import checkpoint as ckpt


def run_solution(problem, solution):

    cpy = solution.copy(deep=True)
    cpy.data['evo_fit'] = cpy.fitness
    problem.evaluate(cpy, save_weights=True)

    return cpy


def run_best_from_checkpoint(problem, folder=None):
    '''searches for the latest checkpoint, loads and runs the best solution stored'''

    last_ckpt = ckpt.get_most_recent(ckpt.data_name.format('*'), folder)
    print(last_ckpt)

    if last_ckpt is None:
        problem.logger.warning('No checkpoitn found.')
        return None

    data = ckpt.load_data(last_ckpt)

    json_data = max(data['population'], key=lambda x: x['fitness'])

    best = GESolution(json_data=json_data)

    return run_solution(problem, best)


def plot_history(history, folder=None):

    l_epochs = np.arange(len(history['loss']))
    l_data = history['loss']

    a_epochs = np.arange(len(history['acc']))
    a_data = history['acc']

    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(l_epochs+1, l_data)
    axs[0].set_xticks(l_epochs+1)
    #axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].grid(True)

    axs[1].plot(a_epochs+1, a_data, color='g')
    axs[1].set_xticks(a_epochs+1)
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].grid(True)

    fig.tight_layout()

    if folder is None:
        folder = ckpt.ckpt_folder

    fig.savefig(os.path.join(folder, 'acc_loss.png'))
    #plt.show()