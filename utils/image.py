import numpy as np

def normalize(img):

	return (img - img.min()) / (img.max() - img.min())


def binarize(img, threshold=0.5):

	img[img > threshold ] = 1
	img[img <= threshold] = 0
	return img

def iou_accuracy(true, pred):
    intersection = true * pred
    union = true + ((1. - true) * pred)
    return np.sum(intersection) / np.sum(union)