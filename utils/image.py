import os

import numpy as np

import keras
import skimage.io as io


def normalize(img):

	return (img - img.min()) / (img.max() - img.min())


def binarize(img, threshold=0.5):

	img[img > threshold ] = 1
	img[img <= threshold] = 0
	return img
