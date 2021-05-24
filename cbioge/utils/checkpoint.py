import glob
import os
import re
import pickle

from cbioge.algorithms.solution import GESolution


ckpt_folder = 'checkpoints'

data_name = 'data_{0}.ckpt'
solution_name = 'solution_{0}.ckpt'


def save_solution(solution):

    json_solution = solution.to_json()
    filename = solution_name.format(solution.id)

    save_data(json_solution, filename)


def save_population(population):

    for solution in population:
        save_solution(solution)


def load_solutions():

    solution_files = glob.glob(os.path.join(ckpt_folder, solution_name.format('*')))
    solution_files.sort()

    solutions = []
    for file in solution_files:
        data = load_data(file)
        s = GESolution(json_data=data)
        if s.fitness is None:
            s.fitness = -1
        solutions.append(s)


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