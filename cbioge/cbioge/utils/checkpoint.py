import glob
import os
import re
import pickle

CKPT_FOLDER = 'checkpoints'
DATA_NAME = 'data_{0}.ckpt'
SOLUTION_NAME = 'solution_{0}.ckpt'


def get_new_unique_path(base_path, name=None):
    # stores the data inside a sub-folder. Uses PID if name is None
    name = str(os.getpid()) if name is None else name
    return os.path.join(base_path, name)


def get_latest_pid_or_new(base_path):
    # gets the latest pid folder inside base path
    folders = glob.glob(os.path.join(base_path, '*/'))

    if len(folders) == 0:
        return get_new_unique_path(base_path)

    folders.sort(reverse=True)
    print(f'latest checkpoint found is {folders[0]}')
    return folders[0]


def natural_key(string_):
    # very helpful to sort file names that contain numbers
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_most_recent(name_pattern, folder=None):

    data_files = glob.glob(os.path.join(folder or CKPT_FOLDER, name_pattern))

    if len(data_files) == 0:
        return None

    # returns the file containing the higher number in its name
    return os.path.basename(max(data_files, key=natural_key))


def get_files_with_name(name_pattern, folder=None):
    files = glob.glob(os.path.join(folder or CKPT_FOLDER, name_pattern))
    return [os.path.basename(f) for f in files]


def save_data(data, filename):
    # try saving the data in the checkpoint folder
    try:
        with open(os.path.join(CKPT_FOLDER, filename), 'wb') as f:
            pickle.dump(data, f)
        return True
    except IOError:
        print(f'[checkpoint] fail to save {filename}')
        return False


def load_data(file_name, folder=None):
    # loads the file stored in the checkpoint folder
    with open(os.path.join(folder or CKPT_FOLDER, file_name), 'rb') as f:
        return pickle.load(f)


def delete_data(name_pattern):
    # deletes all files that matches the name pattern
    data_files = glob.glob(os.path.join(CKPT_FOLDER, name_pattern))
    [os.remove(file) for file in data_files] # pylint: disable=expression-not-assigned
