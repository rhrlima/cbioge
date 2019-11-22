import glob
import os
import re
import pickle


ckpt_folder = 'checkpoints'


def save_solution(solution):

    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    filename = f'solution{solution.id}.ckpt'

    save_data(solution, os.path.join(ckpt_folder, filename))


def load_solutions():

    solution_files = glob.glob(os.path.join(ckpt_folder, 'solution*.ckpt'))
    return [load_data(file) for file in solution_files]


def save_data(data, filename='data.ckpt'):

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]