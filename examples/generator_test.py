import glob
import os

from datasets.dataset import DataGenerator
import utils.dataset as utils

if __name__ == '__main__':

	path = 'datasets/membrane/posproc/train'
	
	utils.read_dataset_from_directory(path)

	ids = glob.glob(os.path.join(path, 'image', '*.png'))
	ids = [os.path.basename(id) for id in ids]

	gen = DataGenerator('datasets/membrane/posproc/train', ids, (256, 256, 1), 1)

	img, msk = gen.__getitem__(0)
	print(img.shape, msk.shape)