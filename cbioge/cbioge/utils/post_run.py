import os, logging

import numpy as np
import matplotlib.pyplot as plt

from ..algorithms import Solution
from . import checkpoint as ckpt


LOGGER = logging.getLogger('cbioge')


def run_solution(problem, solution):

    cpy = solution.copy(deep=True)
    cpy.data['evo_fit'] = cpy.fitness
    problem.evaluate(cpy)

    return cpy


def get_best_from_checkpoint(folder=None):
    '''searches for the latest checkpoint, loads and runs the best solution stored'''

    last_ckpt = ckpt.get_most_recent(ckpt.data_name.format('*'), folder)

    if last_ckpt is None: raise ValueError(f'No checkpoint found.')

    data = ckpt.load_data(last_ckpt, folder)

    json_data = max(data['population'], key=lambda x: x['fitness'])

    return Solution.from_json(json_data)


def plot_history(history, folder=None, name='plot.png'):

    l_epochs = np.arange(len(history['loss']))
    a_epochs = np.arange(len(history['acc']))

    fig, axs = plt.subplots(2, 1, sharex=True)

    plt.title(name)

    axs[0].plot(l_epochs+1, history['loss'], label='loss')
    if 'val_loss' in history: axs[0].plot(l_epochs+1, history['val_loss'], label='val loss')
    #axs[0].set_xticks(l_epochs+1)
    #axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(a_epochs+1, history['acc'], label='acc')
    if 'val_acc' in history: axs[1].plot(a_epochs+1, history['val_acc'], label='val acc')
    #axs[1].set_xticks(a_epochs+1)
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].grid(True)
    axs[1].legend()

    fig.tight_layout()

    if folder is None:
        folder = ckpt.ckpt_folder

    fig.savefig(os.path.join(folder, f'{name}.png'))
    #plt.show()
    plt.close()