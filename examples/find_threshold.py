import pickle

import numpy as np

from utils.image import *

pickle_file = 'datasets/membrane_train_train.pickle'
dataset = pickle.load(open(pickle_file, 'rb'))


img = dataset['train_dataset'][0,:,:,0]
mask = dataset['train_labels'][0,:,:,0]
pred = load_image('trtr1/0.png')

pred = normalize(pred)

maskb = binarize(mask)
predb = binarize(pred)

print(mask.min(), mask.max(), pred.min(), pred.max())

print('mask mask', iou_accuracy(mask, mask))
print('pred pred', iou_accuracy(pred, pred))

print('mask predb', iou_accuracy(mask, predb))
print('maskb predb', iou_accuracy(maskb, predb))

thrs = np.arange(0.4, 0.65, 0.05)
print(thrs)
print('IoU', iou_accuracy(mask, pred))
for t in thrs:
	maskb = binarize(mask, t)
	predb = binarize(pred, t)
	print('IoU', iou_accuracy(maskb, predb), t)
	#print('IoU', iou_accuracy(maskb, predb))

#plot([img, mask, pred])