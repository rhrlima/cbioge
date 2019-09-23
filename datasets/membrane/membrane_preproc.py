import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import skimage.io as io
import skimage.transform as trans


def load_images(path):
    images = []
    files = glob.glob(os.path.join(path, '*.png'))
    for f in files:
        img = io.imread(f, as_gray=True)
        img = trans.resize(img, (256, 256))
        img = normalize(img)
        img = np.reshape(img, (256, 256, 1))
        print(f, img.shape, img.min(), img.max())
        images.append(img)
    return images

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def binarize(mask, threshold=0.5):
    mask[mask > threshold ] = 1
    mask[mask <= threshold] = 0
    return mask

def save(path, items, force=True):

    if not os.path.exists(path):
        os.makedirs(path)

    for i, item in enumerate(items):
        file_name = os.path.join(path, f'{i}.npy')
        if not os.path.exists(file_name) or force:
            print('saving', file_name)
            np.save(os.path.join(path, f'{i}.npy'), item)

if __name__ == '__main__':
    
    images = load_images('train/image')
    masks = load_images('train/label')
    masks = [binarize(m) for m in masks]

    save('npy/train/image', images)
    save('npy/train/label', masks)

    images = load_images('test/image')
    masks = load_images('test/label')
    masks = [binarize(m) for m in masks]

    save('npy/test/image', images)
    save('npy/test/label', masks)