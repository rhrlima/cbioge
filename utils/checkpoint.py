import re
import pickle


ckpt_folder = 'checkpoints'


def save_data(data, filename='data.ckpt'):

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]