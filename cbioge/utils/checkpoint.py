'''module responsible for the checkpoint system

it helps with saving/loading information that is used by the lib, assuming that
every save/load will be performed on the checkpoint folder

also includes some util functions that might be used in some other places.
'''

import glob, os, re, pickle

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


def get_most_recent(name_pattern, folder=None):

    if folder is None: folder = ckpt_folder

    data_files = glob.glob(os.path.join(folder, name_pattern))

    if data_files == []:
        return None
    else:
        # returns the file containing the higher number in its name
        data_files.sort(key=lambda x: natural_key(x), reverse=True)
        return data_files[0]


def get_files_with_name(name_pattern, folder=None):
    files = glob.glob(os.path.join(folder or ckpt_folder, name_pattern))
    return [os.path.basename(f) for f in files]


def save_data(data, filename):
    # try saving the data in the checkpoint folder
    try:
        # should not be necessary
        # if not os.path.exists(ckpt_folder):
        #     os.makedirs(ckpt_folder)

        with open(os.path.join(ckpt_folder, filename), 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        print(f'[checkpoint] fail to save {filename}')
        return False


def load_data(file_name):
    # loads the file stored in the checkpoint folder
    with open(os.path.join(ckpt_folder, file_name), 'rb') as f:
        return pickle.load(f)


def delete_data(name_pattern):
    # deletes all files that matches the name pattern
    data_files = glob.glob(os.path.join(ckpt_folder, name_pattern))
    [os.remove(file) for file in data_files]