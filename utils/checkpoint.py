import re
import pickle


ckpt_folder = 'checkpoints'


def save_data(data, filename='data.ckpt'):

	with open(filename, 'wb') as f:
		pickle.dump(data, f)
		print(f'data saved to file: {filename}')


def load_data(filename):

	with open(filename, 'rb') as f:
		data = pickle.load(f)
		print(f'data loaded from file: {filename}')
	return data


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]