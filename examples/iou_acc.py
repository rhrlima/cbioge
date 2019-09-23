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
	# print(img.min(), img.max())
	img[img > threshold ] = 1.
	img[img <= threshold] = 0.
	return img


#img = np.array(io.imread('datasets/membrane/test/0.png'))
gt = io.imread('datasets/membrane/test/0_true.png', as_gray = True)
pred = io.imread('datasets/membrane/test/1_predict.png', as_gray = True)

#thresholds = [.4, .45, .5, .55, .6]
thresholds = np.arange(.05, .95, .05)

cum_iou = 0.0
for t in thresholds:
	gt2 = adjust_image(gt, t)
	pred2 = adjust_image(pred, t)
	iou = iou_accuracy(gt2, pred2)
	cum_iou += iou
	print(iou)

print('avg:', cum_iou/len(thresholds))