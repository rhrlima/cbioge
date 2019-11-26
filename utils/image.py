import numpy as np

import skimage.io as io
import skimage.transform as trans

import matplotlib.pyplot as plt


def load_image(path, npy=False):

    return io.imread(path) if not npy else np.load(path)


def write_image(path, img, npy=False):

    io.imsave(path, img) if not npy else np.save(path, img)


def binarize(img, threshold=0.5):

    img[img > threshold ] = 1
    img[img <= threshold] = 0
    return img


def normalize(img):

    return (img - img.min()) / (img.max() - img.min())


def resize(img, size):

    return trans.resize(img, size)


def iou_accuracy(true, pred):
    intersection = true * pred
    union = true + ((1. - true) * pred)
    return np.sum(intersection) / np.sum(union)


def calculate_output_size(img_shape, k, s, p):
    ''' width, height, kernel, padding, stride'''
    index = 1 if len(img_shape) == 4 else 0
    w = img_shape[index]
    h = img_shape[index+1]

    p = 0 if p == 'valid' else (k-1) / 2
    ow = ((w - k + 2 * p) // s) + 1
    oh = ((h - k + 2 * p) // s) + 1
    return (int(ow), int(oh))


def plot(imgs):
    items = len(imgs)
    for i in range(items):
        plt.subplot(1, items, i+1)
        plt.imshow(imgs[i])
    plt.show()