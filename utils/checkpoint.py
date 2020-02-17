import glob
import os
import re
import pickle

from algorithms.solutions import GESolution


ckpt_folder = 'checkpoints'


def save_solution(solution):

    json_solution = solution.to_json()
    filename = f'solution_{solution.id}.ckpt'

    save_data(json_solution, filename)


def save_population(population):

    for solution in population:
        save_solution(solution)


def load_solutions():

    solution_files = glob.glob(os.path.join(ckpt_folder, 'solution_*.ckpt'))
    solution_files.sort()

    solutions = []
    for file in solution_files:
        data = load_data(file)
        solutions.append(GESolution(json_data=data))

    return solutions


def save_data(data, filename):

    try:
        if not os.path.exists(ckpt_folder):
            os.mkdir(ckpt_folder)

        complete_path = os.path.join(ckpt_folder, filename)

        with open(complete_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        print(f'[checkpoint] fail to save {filename}')
        return False


def load_data(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def delete_solution_checkpoints(name_pattern):
    solution_files = glob.glob(os.path.join(ckpt_folder, name_pattern))
    [os.remove(file) for file in solution_files]


def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]