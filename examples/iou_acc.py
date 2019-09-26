import os

import numpy as np

import skimage.io as io


def iou_accuracy(true, pred):
	intersection = true * pred
	union = true + ((1. - true) * pred)
	return np.sum(intersection) / np.sum(union)


def iou_loss(true, pred):
	return - iou_accuracy(true, pred)


def adjust_image(img, threshold=0.5):
	img = (img - img.min()) / (img.max() - img.min())
	img[img > threshold ] = 1.
	img[img <= threshold] = 0.
	return img

if __name__ == '__main__':
	
	path = 'datasets/membrane/test/'
	ids = [f'{i}.png' for i in range(30)]

	cum_iou = 0.0
	for id in ids:
		true = io.imread(os.path.join(path, 'label', id), as_gray = True)
		pred = io.imread(os.path.join(path, 'pred', id), as_gray = True)

		#thresholds = np.arange(.05, .95, .05)
		thresholds = [0.5]

		temp_iou = 0.0
		for t in thresholds:
			true = adjust_image(true, t)
			pred = adjust_image(pred, t)
			iou = iou_accuracy(true, pred)
			temp_iou += iou
		temp_iou = temp_iou / len(thresholds)
		cum_iou += temp_iou
		print('id', temp_iou)

	print('avg:', cum_iou/len(ids))