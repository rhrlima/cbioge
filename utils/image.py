import numpy as np
import skimage.transform as trans

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