''' Module responsible for helping manage the save and load of data for checkpoints

    Experiments will be stored in a folder named 'checkpoints' by default
'''

import glob, os, re, pickle

from cbioge.algorithms.solution import GESolution

ckpt_folder = 'checkpoints'
data_name = 'data_{0}.ckpt'
solution_name = 'solution_{0}.ckpt'


def get_new_unique_path(base_path, name=None):
    # stores the data inside a sub-folder. Uses PID if name is None
    name = str(os.getpid()) if name is None else name
    return os.path.join(base_path, name)


def get_latest_pid_or_new(base_path):
    # gets the latest pid folder inside base path
    folders = glob.glob(os.path.join(base_path, '*/'))

    if folders == []:
        return get_new_unique_path(base_path)

    folders.sort(reverse=True)
    print(f'latest checkpoint found is {folders[0]}')
    return folders[0]


def natural_key(string_):
    # very helpful to sort file names that contain numbers
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_most_recent(name_pattern):
    data_files = glob.glob(os.path.join(ckpt_folder, name_pattern))

    if data_files == []:
        return None
    else:
        # returns the file containing the higher number in its name
        data_files.sort(key=lambda x: natural_key(x), reverse=True)
        return data_files[0]


# def load_solution(solution_id):
#     filename = solution_name.format(solution_id)
#     return load_data(filename)


# def save_population(population):

#     for solution in population:
#         save_solution(solution)


# def load_solutions():

#     solution_files = glob.glob(os.path.join(ckpt_folder, solution_name.format('*')))
#     solution_files.sort()

#     solutions = []
#     for file in solution_files:
#         data = load_data(file)
#         s = GESolution(json_data=data)
#         solutions.append(s)

#     return solutions


def save_data(data, filename):
    try:
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        complete_path = os.path.join(ckpt_folder, filename)

        with open(complete_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        print(f'[checkpoint] fail to save {filename}')
        return False


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def delete_data(name_pattern):
    # deletes all files that matches the name pattern
    data_files = glob.glob(os.path.join(ckpt_folder, name_pattern))
    [os.remove(file) for file in data_files]