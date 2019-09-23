import os

from .problem import BaseProblem
import numpy as np

import keras
from keras import callbacks
from keras import optimizers

import skimage.io as io


class ImageSegmentationProblem(BaseProblem):

    parser = None

    steps_per_epoch = 1
    epochs = 1

    loss = 'binary_crossentropy'
    opt = optimizers.Adam(lr = 1e-4)
    metrics = ['accuracy']

    def __init__(self, parser, dataset=None):
        self.parser = parser
        self.dataset = dataset
        self.checkpoint_name = 'model_ckpt.hdf5'

        self.train_generator = None
        self.test_generator = None

    def map_genotype_to_phenotype(self, genotype):
        return None

    def evaluate(self, solution):

        model = solution.phenotype
        if not model:
            return -1

        model.compile(
            optimizer = self.opt,
            loss = self.loss,
            metrics = self.metrics)

        model.summary()

        ckpt = callbacks.ModelCheckpoint(
            self.checkpoint_name,
            monitor='loss',
            save_best_only=True,
            verbose=1)

        model.fit_generator(
            self.train_generator,
            self.steps_per_epoch,
            self.epochs,
            callbacks=[ckpt])

        results = model.predict_generator(self.test_generator, 1, verbose=1)

        return self._calculate_accuracy(results)

    def _calculate_accuracy(self, results):
        acc = 0.0
        for i, pred in enumerate(results):
            true = io.imread(os.path.join(self.dataset['test_path'], f'{i}_true.png'), as_gray=True)
            true = self._adjust_image(true)
            pred = self._adjust_image(pred)
            acc += self._iou_accuracy(true, pred)
        return acc / len(results)

    def _adjust_image(self, img, threshold=0.5):
        img = (img - img.min()) / (img.max() - img.min())
        img[img > threshold ] = 1.
        img[img <= threshold] = 0.
        return img

    def _iou_accuracy(self, true, pred):
        intersection = true * pred
        union = true + ((1. - true) * pred)
        return np.sum(intersection) / np.sum(union)


class Dataset:

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    x_test = None
    y_test = None

    input_shape = None
    num_classes = None

    def _load_from_pickle(pickle_file):
        with open(pickle_file, 'rb') as f:
            temp = pickle.load(f)

            self.x_train = temp['train_dataset']
            self.y_train = temp['train_labels']

            self.x_valid = temp['valid_dataset']
            self.y_valid = temp['valid_labels']

            self.x_test = temp['test_dataset']
            self.y_test = temp['test_labels']

            self.input_shape = temp['input_shape']
            self.num_classes = temp['num_classes']

            del temp

    def _generator(self, images, masks):
        for img, mask in zip(images, masks):
            yield img, mask

    def train_generator(self):

        return self._generator(self.x_train, y_train)

    def valid

#    def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
#        for i in range(num_image):
#            img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
#            img = img / 255
#            img = trans.resize(img,target_size)
#            img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
#            img = np.reshape(img,(1,)+img.shape)
#            yield img